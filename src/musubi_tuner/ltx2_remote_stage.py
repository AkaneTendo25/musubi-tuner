"""Experimental TCP remote-stage runner for LTX-2 transformer blocks.

This is an experimental transport for splitting the transformer block stack
between the local trainer and one or more remote GPU processes. It keeps the
integration default-off and deliberately narrow: each remote side owns a
contiguous transformer-block range, returns stage outputs during forward, then
receives output gradients and returns boundary activation gradients during
backward.

This is intentionally different from DeepSpeed/PyTorch pipeline parallelism.
Those systems usually launch all stages as ranks in one distributed job and
delegate scheduling/collectives to the distributed runtime. This module keeps
the existing single-process trainer as the coordinator and talks to independent
stage servers over a small TCP RPC protocol. That makes it useful for ad-hoc
remote PCs and SSH tunnels, but it does not provide distributed runtime
scheduling, fault tolerance, or high-performance collectives.

Operator notes for a multi-PC split:

1. Start one ``musubi_tuner.ltx2_remote_stage_server`` process per remote
   GPU/PC. Each server must use the same model checkpoint and compatible repo
   revision, but each server owns only its ``--split`` to ``--end`` range. Use
   ``--prune_non_stage_blocks`` so non-owned blocks are replaced after load.
2. On the trainer, pass ordered contiguous specs:
   ``--ltx2_remote_stage_specs "pc-a:17810:12:24;pc-b:17810:24:36;pc-c:17810:36:48"``.
   The trainer runs the prefix before the first remote start index. The last
   remote stage should normally end at the final transformer block.
   This implementation currently coordinates the chain from the trainer process; remote
   stages do not yet stream directly to the next remote stage. For real training
   over slow links, prefer one large remote suffix, or at most a few coarse
   stages, until direct stage-to-stage transport is implemented.
3. The current transport is plain pickle-over-TCP. Run it only on trusted LAN,
   Tailscale/WireGuard, or SSH-tunneled links. Do not expose the port to the
   open internet.
4. For training experiments, start with ``--ltx2_remote_stage_codec aq-int8``
   and ``--ltx2_remote_stage_grad_codec none``. Then test ``grad_codec=int8``.
   AQ codecs follow the AQ-SGD communication pattern: first keyed activations
   are sent exactly, later visits send stochastic quantized activation deltas,
   and optional activation-gradient compression uses the same stochastic
   quantizer. ``aq-int4`` saves more wire bytes, but it is a more aggressive
   approximation and should be treated as a convergence experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import contextlib
import gc
import logging
import os
import pickle
import socket
import struct
import threading
import uuid
import time
from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from musubi_tuner.ltx2_model_parallel import (
    MP_CODEC_INT4,
    MP_CODEC_INT8,
    MP_CODEC_NONE,
    MP_CODECS,
    dequantize_int4_blocks_for_ltx2_mp,
    dequantize_int8_blocks_for_ltx2_mp,
    quantize_int4_blocks_for_ltx2_mp,
    quantize_int8_blocks_for_ltx2_mp,
)
from musubi_tuner.ltx_2.model.transformer.transformer_args import TransformerArgs

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_STAGE_PORT = 7788
TRAINABLE_PARAM_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
REMOTE_STAGE_CODEC_AQ_INT8 = "aq-int8"
REMOTE_STAGE_CODEC_AQ_INT4 = "aq-int4"
REMOTE_STAGE_CODECS = (*MP_CODECS, REMOTE_STAGE_CODEC_AQ_INT8, REMOTE_STAGE_CODEC_AQ_INT4)
REMOTE_STAGE_AQ_CODECS = (REMOTE_STAGE_CODEC_AQ_INT8, REMOTE_STAGE_CODEC_AQ_INT4)
REMOTE_STAGE_AQ_KEY_MODES = ("sample", "sample_timestep", "sample_timestep_noise", "off")
REMOTE_STAGE_TRAINABLE_SCOPE_AUTO = "auto"
REMOTE_STAGE_TRAINABLE_SCOPE_LORA = "lora"
REMOTE_STAGE_TRAINABLE_SCOPE_BLOCKS = "blocks"
REMOTE_STAGE_TRAINABLE_SCOPE_FROZEN = "frozen"
REMOTE_STAGE_TRAINABLE_SCOPES = (
    REMOTE_STAGE_TRAINABLE_SCOPE_AUTO,
    REMOTE_STAGE_TRAINABLE_SCOPE_LORA,
    REMOTE_STAGE_TRAINABLE_SCOPE_BLOCKS,
)
_ARG_STATIC_FIELDS = {
    "context",
    "context_mask",
    "positional_embeddings",
    "cross_positional_embeddings",
    "self_attention_mask",
    "a2v_cross_attention_mask",
    "v2a_cross_attention_mask",
}


class RemoteStageConnectionError(RuntimeError):
    """Raised when a remote stage transport request cannot complete."""


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def _normalize_trainable_scope(scope: str | None) -> str:
    scope = str(scope or REMOTE_STAGE_TRAINABLE_SCOPE_AUTO).strip().lower()
    if scope == "network":
        return REMOTE_STAGE_TRAINABLE_SCOPE_LORA
    return scope


def add_ltx2_remote_stage_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if _parser_has_option(parser, "--ltx2_remote_stage"):
        return parser

    parser.add_argument(
        "--ltx2_remote_stage",
        action="store_true",
        help=(
            "Experimental: run a suffix of LTX-2 transformer blocks on a remote TCP stage server. "
            "Default off. Intended for controlled experiments, not production training."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_host",
        type=str,
        default="127.0.0.1",
        help="Remote LTX-2 stage server host. Used only with --ltx2_remote_stage.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_port",
        type=int,
        default=DEFAULT_REMOTE_STAGE_PORT,
        help=f"Remote LTX-2 stage server port. Default: {DEFAULT_REMOTE_STAGE_PORT}.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_split",
        type=int,
        default=-1,
        help=(
            "First transformer block index to run remotely. Local trainer runs blocks before this index. "
            "Required with --ltx2_remote_stage."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_specs",
        type=str,
        default=None,
        help=(
            "Experimental multi-stage spec for --ltx2_remote_stage. Format: "
            "'host:port:start:end;host:port:start:end'. Stages must be contiguous and usually end at "
            "the final transformer block. Overrides --ltx2_remote_stage_host/port/split."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_timeout",
        type=float,
        default=600.0,
        help="Socket timeout in seconds for remote stage requests. Default: 600.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_codec",
        type=str,
        default=MP_CODEC_NONE,
        choices=REMOTE_STAGE_CODECS,
        help=(
            "Boundary activation codec for remote forward tensors. Choices: none, int8, int4, "
            "aq-int8, aq-int4. AQ codecs quantize activation deltas keyed by sample identity. Default: none."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_grad_codec",
        type=str,
        default=MP_CODEC_NONE,
        choices=MP_CODECS,
        help="Boundary activation-gradient codec for remote backward tensors. Choices: none, int8, int4. Default: none.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_int8_block_size",
        type=int,
        default=256,
        help="Block size for remote-stage low-bit tensor codecs. Default: 256.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_metadata_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache static TransformerArgs fields on each remote stage and resend only dynamic fields on cache hits. "
            "Enabled by default when remote-stage mode is explicitly enabled."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_metadata_cache_size",
        type=int,
        default=8,
        help="Per-connection LRU size for cached remote-stage TransformerArgs static metadata. Default: 8.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_aq_key_mode",
        type=str,
        default="sample",
        choices=REMOTE_STAGE_AQ_KEY_MODES,
        help=(
            "How to build AQ-style delta-cache keys from a batch. sample reuses by cached item key/path; "
            "sample_timestep also hashes model timesteps; sample_timestep_noise also hashes noise; off disables keys."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_aq_stochastic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use stochastic unbiased rounding for AQ delta and AQ-controlled gradient codecs. "
            "This matches AQ-SGD's quantizer assumption. Default: enabled."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_aq_cache_size",
        type=int,
        default=0,
        help=(
            "Maximum activation entries per AQ encode/decode cache. 0 keeps all keyed entries, matching AQ-SGD's "
            "per-sample activation buffer but using CPU RAM. Default: 0."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_trainable",
        action="store_true",
        help=(
            "Ask remote stage servers to own trainable parameters. The local trainer sends optimizer_step "
            "messages after backward. Server processes must also be started with --trainable."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_trainable_scope",
        type=str,
        default=REMOTE_STAGE_TRAINABLE_SCOPE_AUTO,
        choices=REMOTE_STAGE_TRAINABLE_SCOPES,
        help=(
            "Expected remote trainable scope when --ltx2_remote_stage_trainable is set. "
            "'lora' means the remote server must train a LoRA/network adapter; 'blocks' means it may train "
            "owned base block weights; 'auto' accepts whichever scope the server reports. Default: auto."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_learning_rate",
        type=float,
        default=None,
        help="Learning rate recorded for remote-stage training metadata. Server --learning_rate controls updates.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay recorded for remote-stage training metadata. Server --weight_decay controls updates.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_max_grad_norm",
        type=float,
        default=0.0,
        help="Gradient clip norm recorded for remote-stage training metadata. Server --max_grad_norm controls clipping.",
    )
    parser.add_argument(
        "--ltx2_remote_stage_checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Server-local directory used for remote trainable stage checkpoints. If omitted, the trainer output_dir "
            "string is sent to the server, which only works when both sides share that filesystem path."
        ),
    )
    parser.add_argument(
        "--ltx2_remote_stage_prune_local_blocks",
        action="store_true",
        help=(
            "Experimental memory-saving mode: after the remote split is configured, replace transformer blocks "
            "owned by remote stages with lightweight placeholders on the local trainer. This keeps global block "
            "indices stable but prevents unused local suffix blocks and their LoRA modules from occupying VRAM."
        ),
    )
    return parser


def is_ltx2_remote_stage_enabled(args: argparse.Namespace | None = None) -> bool:
    return bool(getattr(args, "ltx2_remote_stage", False))


@dataclass(frozen=True)
class LTX2RemoteStageSpec:
    host: str
    port: int
    start_index: int
    end_index: int | None = None

    @property
    def split_index(self) -> int:
        return self.start_index

    def resolved_end(self, num_blocks: int | None = None) -> int | None:
        if self.end_index is not None:
            return self.end_index
        return num_blocks


def parse_ltx2_remote_stage_specs(
    args: argparse.Namespace,
    *,
    num_blocks: int | None = None,
) -> list[LTX2RemoteStageSpec]:
    raw_specs = getattr(args, "ltx2_remote_stage_specs", None)
    specs: list[LTX2RemoteStageSpec] = []
    if raw_specs is not None and str(raw_specs).strip():
        for raw_spec in str(raw_specs).split(";"):
            raw_spec = raw_spec.strip()
            if not raw_spec:
                continue
            parts = raw_spec.rsplit(":", 3)
            if len(parts) != 4:
                raise ValueError(
                    "ltx2_remote_stage_specs entries must be 'host:port:start:end', "
                    f"got {raw_spec!r}"
                )
            host, raw_port, raw_start, raw_end = parts
            end_index = int(raw_end)
            specs.append(
                LTX2RemoteStageSpec(
                    host=host,
                    port=int(raw_port),
                    start_index=int(raw_start),
                    end_index=None if end_index < 0 else end_index,
                )
            )
    else:
        split_index = int(getattr(args, "ltx2_remote_stage_split", -1))
        if split_index >= 0:
            specs.append(
                LTX2RemoteStageSpec(
                    host=getattr(args, "ltx2_remote_stage_host", "127.0.0.1"),
                    port=int(getattr(args, "ltx2_remote_stage_port", DEFAULT_REMOTE_STAGE_PORT)),
                    start_index=split_index,
                    end_index=num_blocks,
                )
            )

    _validate_remote_stage_specs(specs, num_blocks=num_blocks)
    return specs


def _validate_remote_stage_specs(
    specs: list[LTX2RemoteStageSpec],
    *,
    num_blocks: int | None = None,
) -> None:
    if not specs:
        raise ValueError("--ltx2_remote_stage requires --ltx2_remote_stage_split or --ltx2_remote_stage_specs")
    previous_end: int | None = None
    for idx, spec in enumerate(specs):
        if not spec.host:
            raise ValueError(f"remote stage {idx} host must not be empty")
        if not (0 < int(spec.port) < 65536):
            raise ValueError(f"remote stage {idx} port must be in 1..65535, got {spec.port}")
        if int(spec.start_index) < 0:
            raise ValueError(f"remote stage {idx} start index must be >= 0, got {spec.start_index}")
        resolved_end = spec.resolved_end(num_blocks)
        if resolved_end is None:
            if num_blocks is None and idx == len(specs) - 1:
                if previous_end is not None and int(spec.start_index) != previous_end:
                    raise ValueError(
                        "remote stages must be contiguous; "
                        f"stage {idx - 1} ended at {previous_end}, stage {idx} starts at {spec.start_index}"
                    )
                previous_end = None
                continue
            raise ValueError(
                f"remote stage {idx} end index is unknown; provide an end value in --ltx2_remote_stage_specs"
            )
        if resolved_end <= int(spec.start_index):
            raise ValueError(
                f"remote stage {idx} end index must be > start index, got {spec.start_index}:{resolved_end}"
            )
        if num_blocks is not None and resolved_end > num_blocks:
            raise ValueError(f"remote stage {idx} end index {resolved_end} exceeds block count {num_blocks}")
        if previous_end is not None and int(spec.start_index) != previous_end:
            raise ValueError(
                "remote stages must be contiguous; "
                f"stage {idx - 1} ended at {previous_end}, stage {idx} starts at {spec.start_index}"
            )
        previous_end = resolved_end
    if num_blocks is not None and previous_end != num_blocks:
        raise ValueError(
            f"remote stages must currently cover the transformer suffix through block {num_blocks}; "
            f"last stage ends at {previous_end}"
        )


def get_ltx2_remote_stage_local_keep_range(args: argparse.Namespace) -> tuple[int, int] | None:
    if not is_ltx2_remote_stage_enabled(args):
        return None
    if not bool(getattr(args, "ltx2_remote_stage_prune_local_blocks", False)):
        return None
    specs = parse_ltx2_remote_stage_specs(args, num_blocks=None)
    return 0, int(specs[0].start_index)


class LTX2RemoteStageClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        split_index: int,
        end_index: int | None = None,
        timeout: float = 600.0,
        codec: str = MP_CODEC_NONE,
        grad_codec: str = MP_CODEC_NONE,
        int8_block_size: int = 256,
        metadata_cache_enabled: bool = True,
        metadata_cache_size: int = 8,
        aq_stochastic: bool = True,
        aq_cache_size: int = 0,
        expected_trainable_scope: str = REMOTE_STAGE_TRAINABLE_SCOPE_AUTO,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.start_index = int(split_index)
        self.split_index = self.start_index
        self.end_index = None if end_index is None or int(end_index) < 0 else int(end_index)
        self.timeout = float(timeout)
        self.codec = str(codec or MP_CODEC_NONE).lower()
        self.grad_codec = str(grad_codec or MP_CODEC_NONE).lower()
        self.int8_block_size = int(int8_block_size)
        self.metadata_cache_enabled = bool(metadata_cache_enabled)
        self.metadata_cache_size = int(metadata_cache_size)
        self.aq_stochastic = bool(aq_stochastic)
        self.aq_cache_size = int(aq_cache_size)
        self.expected_trainable_scope = _normalize_trainable_scope(expected_trainable_scope)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._metadata_cache_keys: OrderedDict[str, None] = OrderedDict()
        self._aq_encode_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._aq_decode_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.stats: dict[str, float | int] = _new_remote_stage_transport_stats()
        self.validate()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LTX2RemoteStageClient":
        return cls(
            host=getattr(args, "ltx2_remote_stage_host", "127.0.0.1"),
            port=int(getattr(args, "ltx2_remote_stage_port", DEFAULT_REMOTE_STAGE_PORT)),
            split_index=int(getattr(args, "ltx2_remote_stage_split", -1)),
            end_index=None,
            timeout=float(getattr(args, "ltx2_remote_stage_timeout", 600.0)),
            codec=getattr(args, "ltx2_remote_stage_codec", MP_CODEC_NONE),
            grad_codec=getattr(args, "ltx2_remote_stage_grad_codec", MP_CODEC_NONE),
            int8_block_size=int(getattr(args, "ltx2_remote_stage_int8_block_size", 256)),
            metadata_cache_enabled=bool(getattr(args, "ltx2_remote_stage_metadata_cache", True)),
            metadata_cache_size=int(getattr(args, "ltx2_remote_stage_metadata_cache_size", 8)),
            aq_stochastic=bool(getattr(args, "ltx2_remote_stage_aq_stochastic", True)),
            aq_cache_size=int(getattr(args, "ltx2_remote_stage_aq_cache_size", 0)),
            expected_trainable_scope=getattr(args, "ltx2_remote_stage_trainable_scope", REMOTE_STAGE_TRAINABLE_SCOPE_AUTO),
        )

    def validate(self) -> None:
        if self.split_index < 0:
            raise ValueError("--ltx2_remote_stage_split must be set to a non-negative block index")
        if self.end_index is not None and self.end_index <= self.start_index:
            raise ValueError(f"remote stage end_index must be > start_index, got {self.start_index}:{self.end_index}")
        if not (0 < self.port < 65536):
            raise ValueError(f"ltx2_remote_stage_port must be in 1..65535, got {self.port}")
        if self.timeout <= 0:
            raise ValueError("ltx2_remote_stage_timeout must be > 0")
        if self.codec not in REMOTE_STAGE_CODECS:
            raise ValueError(f"ltx2_remote_stage_codec must be one of {REMOTE_STAGE_CODECS}, got {self.codec!r}")
        if self.grad_codec not in MP_CODECS:
            raise ValueError(f"ltx2_remote_stage_grad_codec must be one of {MP_CODECS}, got {self.grad_codec!r}")
        if self.int8_block_size <= 0:
            raise ValueError("ltx2_remote_stage_int8_block_size must be > 0")
        if self.metadata_cache_size <= 0:
            raise ValueError("ltx2_remote_stage_metadata_cache_size must be > 0")
        if self.aq_cache_size < 0:
            raise ValueError("ltx2_remote_stage_aq_cache_size must be >= 0")
        if self.expected_trainable_scope not in REMOTE_STAGE_TRAINABLE_SCOPES:
            raise ValueError(
                f"ltx2_remote_stage_trainable_scope must be one of {REMOTE_STAGE_TRAINABLE_SCOPES}, "
                f"got {self.expected_trainable_scope!r}"
            )

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._metadata_cache_keys.clear()
                self._aq_encode_cache.clear()
                self._aq_decode_cache.clear()

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        kind = str(message.get("kind", "unknown"))
        with self._lock:
            start_time = time.perf_counter()
            try:
                sock = self._ensure_socket()
                sent_bytes = _send_message(sock, message)
                response, received_bytes = _recv_message_with_size(sock)
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                self.stats["tcp_request_failures"] = int(self.stats.get("tcp_request_failures", 0)) + 1
                self._close_unlocked()
                raise RemoteStageConnectionError(
                    "LTX2 remote stage request "
                    f"{kind!r} failed for {self.host}:{self.port} "
                    f"blocks {self.start_index}:{self.end_index} after {elapsed:.1f}s "
                    f"(timeout={self.timeout:.1f}s): {exc}"
                ) from exc
            elapsed = time.perf_counter() - start_time
            _accumulate_transport_stats(
                self.stats,
                kind=kind,
                sent_bytes=sent_bytes,
                received_bytes=received_bytes,
                elapsed=elapsed,
            )
        if not isinstance(response, dict):
            raise RuntimeError(f"Remote stage returned non-dict response: {type(response)}")
        if response.get("ok") is not True:
            raise RuntimeError(f"Remote stage error: {response.get('error', 'unknown error')}")
        return response

    def get_stats(self) -> dict[str, Any]:
        response = self.request({"kind": "get_stats"})
        return dict(response.get("stats", {}))

    def reset_stats(self) -> None:
        self.request({"kind": "reset_stats"})
        self.stats = _new_remote_stage_transport_stats()

    def _ensure_socket(self) -> socket.socket:
        if self._sock is None:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            _configure_remote_stage_socket(sock, self.timeout)
            try:
                self._hello(sock)
            except Exception:
                try:
                    sock.close()
                finally:
                    pass
                raise
            self._sock = sock
        return self._sock

    def _hello(self, sock: socket.socket) -> None:
        _send_message(
            sock,
            {
                "kind": "hello",
                "split_index": self.start_index,
                "start_index": self.start_index,
                "end_index": self.end_index,
                "codec": self.codec,
                "grad_codec": self.grad_codec,
                "int8_block_size": self.int8_block_size,
                "metadata_cache_enabled": self.metadata_cache_enabled,
                "metadata_cache_size": self.metadata_cache_size,
                "aq_stochastic": self.aq_stochastic,
                "aq_cache_size": self.aq_cache_size,
            },
        )
        response = _recv_message(sock)
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"Remote stage hello failed: {response}")
        server_split = int(response.get("split_index", -1))
        if server_split != self.start_index:
            raise RuntimeError(
                f"Remote stage split mismatch: client split={self.start_index}, server split={server_split}"
            )
        server_end = response.get("end_index")
        if self.end_index is not None and server_end is not None and int(server_end) != self.end_index:
            raise RuntimeError(
                f"Remote stage end mismatch: client end={self.end_index}, server end={int(server_end)}"
            )
        if self.expected_trainable_scope != REMOTE_STAGE_TRAINABLE_SCOPE_AUTO:
            server_scope = _normalize_trainable_scope(response.get("trainable_scope"))
            if server_scope != self.expected_trainable_scope:
                raise RuntimeError(
                    "Remote stage trainable scope mismatch: "
                    f"client expected {self.expected_trainable_scope!r}, server reported {server_scope!r}"
                )


class LTX2RemoteStageGroup:
    def __init__(self, clients: list[LTX2RemoteStageClient]) -> None:
        if not clients:
            raise ValueError("LTX2RemoteStageGroup requires at least one client")
        self.clients = tuple(clients)
        previous_end: int | None = None
        for idx, client in enumerate(self.clients):
            start_index = int(client.start_index)
            end_index = client.end_index
            if end_index is not None and int(end_index) <= start_index:
                raise ValueError(
                    f"remote stage client {idx} end index must be > start index, got {start_index}:{end_index}"
                )
            if previous_end is not None and start_index != previous_end:
                raise ValueError(
                    "remote stage clients must be ordered and contiguous; "
                    f"client {idx - 1} ended at {previous_end}, client {idx} starts at {start_index}"
                )
            previous_end = None if end_index is None else int(end_index)
        self.start_index = self.clients[0].start_index
        self.end_index = self.clients[-1].end_index

    @property
    def split_index(self) -> int:
        return self.start_index

    def close(self) -> None:
        for client in self.clients:
            client.close()

    def zero_grad(self) -> list[dict[str, Any]]:
        return [client.request({"kind": "zero_grad"}) for client in self.clients]

    def optimizer_step(self) -> list[dict[str, Any]]:
        return [client.request({"kind": "optimizer_step"}) for client in self.clients]

    def save_state(self, checkpoint_dir: str, checkpoint_name: str) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        for idx, client in enumerate(self.clients):
            responses.append(
                client.request(
                    {
                        "kind": "save_state",
                        "checkpoint_dir": checkpoint_dir,
                        "checkpoint_name": checkpoint_name,
                        "stage_index": idx,
                    }
                )
            )
        return responses

    def load_state(self, checkpoint_dir: str, checkpoint_name: str) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        for idx, client in enumerate(self.clients):
            responses.append(
                client.request(
                    {
                        "kind": "load_state",
                        "checkpoint_dir": checkpoint_dir,
                        "checkpoint_name": checkpoint_name,
                        "stage_index": idx,
                    }
                )
            )
        return responses

    def get_stats(self) -> list[dict[str, Any]]:
        return [client.get_stats() for client in self.clients]

    def reset_stats(self) -> None:
        for client in self.clients:
            client.reset_stats()

    def describe(self) -> str:
        return ";".join(
            f"{client.host}:{client.port}:{client.start_index}:{client.end_index if client.end_index is not None else -1}"
            for client in self.clients
        )


def build_ltx2_remote_stage_group(
    args: argparse.Namespace,
    *,
    num_blocks: int | None = None,
) -> LTX2RemoteStageGroup:
    specs = parse_ltx2_remote_stage_specs(args, num_blocks=num_blocks)
    clients = [
        LTX2RemoteStageClient(
            host=spec.host,
            port=spec.port,
            split_index=spec.start_index,
            end_index=spec.resolved_end(num_blocks),
            timeout=float(getattr(args, "ltx2_remote_stage_timeout", 600.0)),
            codec=getattr(args, "ltx2_remote_stage_codec", MP_CODEC_NONE),
            grad_codec=getattr(args, "ltx2_remote_stage_grad_codec", MP_CODEC_NONE),
            int8_block_size=int(getattr(args, "ltx2_remote_stage_int8_block_size", 256)),
            metadata_cache_enabled=bool(getattr(args, "ltx2_remote_stage_metadata_cache", True)),
            metadata_cache_size=int(getattr(args, "ltx2_remote_stage_metadata_cache_size", 8)),
            aq_stochastic=bool(getattr(args, "ltx2_remote_stage_aq_stochastic", True)),
            aq_cache_size=int(getattr(args, "ltx2_remote_stage_aq_cache_size", 0)),
            expected_trainable_scope=getattr(args, "ltx2_remote_stage_trainable_scope", REMOTE_STAGE_TRAINABLE_SCOPE_AUTO),
        )
        for spec in specs
    ]
    return LTX2RemoteStageGroup(clients)


def validate_ltx2_remote_stage_setup(args: argparse.Namespace, transformer: torch.nn.Module | None = None) -> None:
    if not is_ltx2_remote_stage_enabled(args):
        return
    scope = _normalize_trainable_scope(getattr(args, "ltx2_remote_stage_trainable_scope", REMOTE_STAGE_TRAINABLE_SCOPE_AUTO))
    if scope != REMOTE_STAGE_TRAINABLE_SCOPE_AUTO and not bool(getattr(args, "ltx2_remote_stage_trainable", False)):
        raise ValueError("--ltx2_remote_stage_trainable_scope requires --ltx2_remote_stage_trainable")
    if transformer is not None:
        base_model = transformer.model if hasattr(transformer, "model") else transformer
        blocks = getattr(base_model, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError("LTX2 remote stage requires transformer_blocks on the base model")
        build_ltx2_remote_stage_group(args, num_blocks=len(blocks))
    else:
        build_ltx2_remote_stage_group(args, num_blocks=None)


def enable_ltx2_remote_stage(model: torch.nn.Module, args: argparse.Namespace) -> LTX2RemoteStageGroup:
    base_model = model.model if hasattr(model, "model") else model
    validate_ltx2_remote_stage_setup(args, base_model)
    blocks = getattr(base_model, "transformer_blocks", None)
    group = build_ltx2_remote_stage_group(args, num_blocks=len(blocks))
    base_model._ltx2_remote_stage_group = group
    base_model._ltx2_remote_stage_client = group.clients[0]
    base_model._ltx2_remote_stage_split = group.start_index
    logger.info(
        "LTX2 remote stage enabled: stages=%s codec=%s grad_codec=%s int8_block_size=%d trainable=%s scope=%s",
        group.describe(),
        group.clients[0].codec,
        group.clients[0].grad_codec,
        group.clients[0].int8_block_size,
        bool(getattr(args, "ltx2_remote_stage_trainable", False)),
        getattr(args, "ltx2_remote_stage_trainable_scope", REMOTE_STAGE_TRAINABLE_SCOPE_AUTO),
    )
    if not bool(getattr(args, "ltx2_remote_stage_trainable", False)):
        logger.warning(
            "LTX2 remote stage is running in frozen-remote mode. Use --ltx2_remote_stage_trainable and start "
            "servers with --trainable to test remote-owned parameter updates."
        )
    return group


def prune_ltx2_blocks_to_range(
    model: torch.nn.Module,
    *,
    keep_start: int,
    keep_end: int,
    label: str = "stage",
    clear_cuda_cache: bool = True,
) -> int:
    """Replace non-owned transformer blocks while preserving global block indices."""

    base_model = model.model if hasattr(model, "model") else model
    blocks = getattr(base_model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("LTX2 remote block pruning requires transformer_blocks on the base model")
    keep_start = int(keep_start)
    keep_end = int(keep_end)
    if keep_start < 0 or keep_end < keep_start or keep_end > len(blocks):
        raise ValueError(f"invalid keep range {keep_start}:{keep_end} for {len(blocks)} transformer blocks")

    replaced = 0
    for idx in range(len(blocks)):
        if keep_start <= idx < keep_end:
            continue
        if isinstance(blocks[idx], torch.nn.Identity):
            continue
        blocks[idx] = torch.nn.Identity()
        replaced += 1

    if replaced:
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "LTX2 remote stage %s pruning: kept blocks %d:%d, replaced %d non-owned blocks",
            label,
            keep_start,
            keep_end,
            replaced,
        )
    return replaced


def prune_ltx2_remote_stage_local_blocks(model: torch.nn.Module, args: argparse.Namespace) -> int:
    if not bool(getattr(args, "ltx2_remote_stage_prune_local_blocks", False)):
        return 0
    base_model = model.model if hasattr(model, "model") else model
    group = getattr(base_model, "_ltx2_remote_stage_group", None)
    if not isinstance(group, LTX2RemoteStageGroup):
        blocks = getattr(base_model, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError("LTX2 remote local pruning requires transformer_blocks on the base model")
        group = build_ltx2_remote_stage_group(args, num_blocks=len(blocks))
    return prune_ltx2_blocks_to_range(
        base_model,
        keep_start=0,
        keep_end=group.start_index,
        label="local",
    )


def close_ltx2_remote_stage(model: torch.nn.Module) -> None:
    base_model = model.model if hasattr(model, "model") else model
    group = getattr(base_model, "_ltx2_remote_stage_group", None)
    if isinstance(group, LTX2RemoteStageGroup):
        group.close()
        return
    client = getattr(base_model, "_ltx2_remote_stage_client", None)
    if isinstance(client, LTX2RemoteStageClient):
        client.close()


def zero_grad_ltx2_remote_stage(model: torch.nn.Module) -> list[dict[str, Any]]:
    base_model = model.model if hasattr(model, "model") else model
    group = getattr(base_model, "_ltx2_remote_stage_group", None)
    if isinstance(group, LTX2RemoteStageGroup):
        return group.zero_grad()
    client = getattr(base_model, "_ltx2_remote_stage_client", None)
    if isinstance(client, LTX2RemoteStageClient):
        return [client.request({"kind": "zero_grad"})]
    return []


def optimizer_step_ltx2_remote_stage(model: torch.nn.Module) -> list[dict[str, Any]]:
    base_model = model.model if hasattr(model, "model") else model
    group = getattr(base_model, "_ltx2_remote_stage_group", None)
    if isinstance(group, LTX2RemoteStageGroup):
        return group.optimizer_step()
    client = getattr(base_model, "_ltx2_remote_stage_client", None)
    if isinstance(client, LTX2RemoteStageClient):
        return [client.request({"kind": "optimizer_step"})]
    return []


def save_ltx2_remote_stage_state(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str,
    checkpoint_name: str,
) -> list[dict[str, Any]]:
    base_model = model.model if hasattr(model, "model") else model
    group = getattr(base_model, "_ltx2_remote_stage_group", None)
    if isinstance(group, LTX2RemoteStageGroup):
        return group.save_state(checkpoint_dir, checkpoint_name)
    client = getattr(base_model, "_ltx2_remote_stage_client", None)
    if isinstance(client, LTX2RemoteStageClient):
        return [
            client.request(
                {
                    "kind": "save_state",
                    "checkpoint_dir": checkpoint_dir,
                    "checkpoint_name": checkpoint_name,
                    "stage_index": 0,
                }
            )
        ]
    return []


def load_ltx2_remote_stage_state(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str,
    checkpoint_name: str,
) -> list[dict[str, Any]]:
    base_model = model.model if hasattr(model, "model") else model
    group = getattr(base_model, "_ltx2_remote_stage_group", None)
    if isinstance(group, LTX2RemoteStageGroup):
        return group.load_state(checkpoint_dir, checkpoint_name)
    client = getattr(base_model, "_ltx2_remote_stage_client", None)
    if isinstance(client, LTX2RemoteStageClient):
        return [
            client.request(
                {
                    "kind": "load_state",
                    "checkpoint_dir": checkpoint_dir,
                    "checkpoint_name": checkpoint_name,
                    "stage_index": 0,
                }
            )
        ]
    return []


def build_ltx2_remote_stage_cache_key(
    args: argparse.Namespace,
    batch: dict[str, Any] | None,
    *,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    mode = str(getattr(args, "ltx2_remote_stage_aq_key_mode", "sample") or "sample").lower()
    if mode == "off":
        return None
    if mode not in REMOTE_STAGE_AQ_KEY_MODES:
        raise ValueError(f"ltx2_remote_stage_aq_key_mode must be one of {REMOTE_STAGE_AQ_KEY_MODES}, got {mode!r}")
    if not isinstance(batch, dict):
        return None

    tokens = _resolve_batch_identity_tokens(batch)
    if not tokens:
        return None

    payload: dict[str, Any] = {
        "mode": mode,
        "items": tokens,
    }
    if mode in {"sample_timestep", "sample_timestep_noise"} and timesteps is not None:
        payload["timesteps"] = timesteps.detach()
    if mode == "sample_timestep_noise" and noise is not None:
        payload["noise"] = noise.detach()
    batch_key = _stable_tree_digest(payload)
    sample_keys = _build_ltx2_remote_stage_sample_cache_keys(
        mode,
        tokens,
        timesteps=timesteps,
        noise=noise,
    )
    return {
        "batch": batch_key,
        "samples": sample_keys,
    }


def set_ltx2_remote_stage_cache_key(model: torch.nn.Module, cache_key: Any | None) -> None:
    base_model = model.model if hasattr(model, "model") else model
    base_model._ltx2_remote_stage_cache_key = _normalize_remote_stage_cache_key(cache_key)


def run_remote_ltx2_stage(
    client: LTX2RemoteStageClient,
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
    perturbations: Any,
    cache_key: Any | None = None,
) -> tuple[TransformerArgs | None, TransformerArgs | None]:
    if video is None and audio is None:
        return video, audio

    request_id = uuid.uuid4().hex
    cache_key_state = _normalize_remote_stage_cache_key(cache_key)
    batch_cache_key = _remote_stage_batch_cache_key(cache_key_state)
    sample_cache_keys = _remote_stage_sample_cache_keys(cache_key_state)
    activation_cache_key: Any | None = (
        {"batch": batch_cache_key, "samples": sample_cache_keys}
        if sample_cache_keys
        else batch_cache_key
    )
    payload = {
        "kind": "forward",
        "request_id": request_id,
        "split_index": client.start_index,
        "start_index": client.start_index,
        "end_index": client.end_index,
        "cache_key": batch_cache_key,
        "sample_cache_keys": sample_cache_keys,
        "video": _prepare_transformer_args_payload(client, "video", batch_cache_key, video),
        "audio": _prepare_transformer_args_payload(client, "audio", batch_cache_key, audio),
        "video_x": (
            _encode_activation_tensor(
                video.x,
                client.codec,
                client.int8_block_size,
                cache=client._aq_encode_cache,
                cache_key=_activation_cache_key(client, activation_cache_key, "forward_in", "video"),
                max_cache_size=client.aq_cache_size,
                stochastic=client.aq_stochastic,
                stats=client.stats,
            )
            if video is not None
            else None
        ),
        "audio_x": (
            _encode_activation_tensor(
                audio.x,
                client.codec,
                client.int8_block_size,
                cache=client._aq_encode_cache,
                cache_key=_activation_cache_key(client, activation_cache_key, "forward_in", "audio"),
                max_cache_size=client.aq_cache_size,
                stochastic=client.aq_stochastic,
                stats=client.stats,
            )
            if audio is not None
            else None
        ),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "perturbations": _to_cpu_tree(perturbations),
        "codec": client.codec,
        "grad_codec": client.grad_codec,
        "int8_block_size": client.int8_block_size,
        "aq_stochastic": client.aq_stochastic,
        "aq_cache_size": client.aq_cache_size,
    }
    response = client.request(payload)
    video_x_out = (
        _decode_activation_tensor(
            response.get("video_x"),
            _first_arg_device(video, audio),
            cache=client._aq_decode_cache,
            max_cache_size=client.aq_cache_size,
            stats=client.stats,
        )
        if video is not None
        else None
    )
    audio_x_out = (
        _decode_activation_tensor(
            response.get("audio_x"),
            _first_arg_device(audio, video),
            cache=client._aq_decode_cache,
            max_cache_size=client.aq_cache_size,
            stats=client.stats,
        )
        if audio is not None
        else None
    )

    video_x_in = video.x if video is not None else torch.empty(0)
    audio_x_in = audio.x if audio is not None else torch.empty(0)
    video_x_wrapped, audio_x_wrapped = _RemoteStageBackward.apply(
        video_x_in,
        audio_x_in,
        video_x_out if video_x_out is not None else torch.empty(0, device=video_x_in.device),
        audio_x_out if audio_x_out is not None else torch.empty(0, device=audio_x_in.device),
        client,
        request_id,
        bool(video is not None),
        bool(audio is not None),
    )

    if video is not None:
        video = replace(video, x=video_x_wrapped)
    if audio is not None:
        audio = replace(audio, x=audio_x_wrapped)
    return video, audio


def run_remote_ltx2_stage_chain(
    group: LTX2RemoteStageGroup,
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
    perturbations: Any,
    cache_key: Any | None = None,
) -> tuple[TransformerArgs | None, TransformerArgs | None]:
    for client in group.clients:
        video, audio = run_remote_ltx2_stage(client, video, audio, perturbations, cache_key=cache_key)
    return video, audio


class _RemoteStageBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        video_x_in: torch.Tensor,
        audio_x_in: torch.Tensor,
        video_x_out: torch.Tensor,
        audio_x_out: torch.Tensor,
        client: LTX2RemoteStageClient,
        request_id: str,
        has_video: bool,
        has_audio: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.client = client
        ctx.request_id = request_id
        ctx.has_video = bool(has_video)
        ctx.has_audio = bool(has_audio)
        ctx.video_input_device = video_x_in.device
        ctx.audio_input_device = audio_x_in.device
        return video_x_out, audio_x_out

    @staticmethod
    def backward(ctx, grad_video_out: torch.Tensor | None, grad_audio_out: torch.Tensor | None):
        client: LTX2RemoteStageClient = ctx.client
        payload = {
            "kind": "backward",
            "request_id": ctx.request_id,
            "video_grad": (
                _encode_tensor(
                    grad_video_out,
                    client.grad_codec,
                    client.int8_block_size,
                    stochastic=client.codec in REMOTE_STAGE_AQ_CODECS and client.aq_stochastic,
                )
                if ctx.has_video and grad_video_out is not None
                else None
            ),
            "audio_grad": (
                _encode_tensor(
                    grad_audio_out,
                    client.grad_codec,
                    client.int8_block_size,
                    stochastic=client.codec in REMOTE_STAGE_AQ_CODECS and client.aq_stochastic,
                )
                if ctx.has_audio and grad_audio_out is not None
                else None
            ),
            "grad_codec": client.grad_codec,
            "int8_block_size": client.int8_block_size,
            "grad_stochastic": client.codec in REMOTE_STAGE_AQ_CODECS and client.aq_stochastic,
        }
        response = client.request(payload)
        video_grad = (
            _decode_tensor(response.get("video_grad"), torch.device(ctx.video_input_device))
            if ctx.has_video
            else None
        )
        audio_grad = (
            _decode_tensor(response.get("audio_grad"), torch.device(ctx.audio_input_device))
            if ctx.has_audio
            else None
        )
        return video_grad, audio_grad, None, None, None, None, None, None


class LTX2RemoteStageServer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        split_index: int,
        end_index: int | None = None,
        device: torch.device | str,
        bind: str = "0.0.0.0",
        port: int = DEFAULT_REMOTE_STAGE_PORT,
        int8_block_size: int = 256,
        trainable: bool = False,
        learning_rate: float | None = None,
        weight_decay: float = 0.01,
        max_grad_norm: float = 0.0,
        autocast_dtype: torch.dtype | None = None,
        trainable_network: torch.nn.Module | None = None,
        network_optimizer_params: list[dict[str, Any]] | None = None,
        prune_non_stage_blocks: bool = False,
        trainable_scope: str = REMOTE_STAGE_TRAINABLE_SCOPE_AUTO,
    ) -> None:
        self.model = model.model if hasattr(model, "model") else model
        self.start_index = int(split_index)
        self.split_index = self.start_index
        self.end_index = None if end_index is None or int(end_index) < 0 else int(end_index)
        self.device = torch.device(device)
        self.bind = str(bind)
        self.port = int(port)
        self.int8_block_size = int(int8_block_size)
        self.trainable = bool(trainable)
        self.learning_rate = learning_rate
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.autocast_dtype = autocast_dtype
        self.network = trainable_network
        self.network_optimizer_params = network_optimizer_params
        self.prune_non_stage_blocks = bool(prune_non_stage_blocks)
        self.trainable_scope = _normalize_trainable_scope(trainable_scope)
        self.optimizer: torch.optim.Optimizer | None = None
        self.contexts: dict[str, dict[str, Any]] = {}
        self.metadata_cache_enabled = True
        self.metadata_cache_size = 8
        self.aq_stochastic = True
        self.aq_cache_size = 0
        self._metadata_arg_cache: dict[str, dict[str, Any]] = {}
        self._metadata_cache_keys: OrderedDict[str, None] = OrderedDict()
        self._aq_encode_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._aq_decode_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.stats: dict[str, float | int] = _new_remote_stage_server_stats()
        self._validate()
        if self.prune_non_stage_blocks:
            prune_ltx2_blocks_to_range(
                self.model,
                keep_start=self.start_index,
                keep_end=int(self.end_index),
                label="server",
                clear_cuda_cache=True,
            )
        self._configure_trainable_params()

    def _validate(self) -> None:
        blocks = getattr(self.model, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError("Remote stage server model must expose transformer_blocks")
        if self.end_index is None:
            self.end_index = len(blocks)
        if self.start_index < 0 or self.start_index >= len(blocks):
            raise ValueError(f"split_index must be inside 0..{len(blocks) - 1}, got {self.start_index}")
        if self.end_index <= self.start_index or self.end_index > len(blocks):
            raise ValueError(
                f"end_index must be inside {self.start_index + 1}..{len(blocks)}, got {self.end_index}"
            )
        if self.int8_block_size <= 0:
            raise ValueError("int8_block_size must be > 0")
        if self.trainable_scope not in REMOTE_STAGE_TRAINABLE_SCOPES:
            raise ValueError(f"remote trainable scope must be one of {REMOTE_STAGE_TRAINABLE_SCOPES}, got {self.trainable_scope!r}")
        if self.trainable_scope == REMOTE_STAGE_TRAINABLE_SCOPE_AUTO:
            self.trainable_scope = (
                REMOTE_STAGE_TRAINABLE_SCOPE_LORA if self.network is not None else REMOTE_STAGE_TRAINABLE_SCOPE_BLOCKS
            )
        if self.trainable_scope == REMOTE_STAGE_TRAINABLE_SCOPE_LORA and self.network is None:
            raise ValueError("--trainable_scope lora requires --network_module on the remote stage server")
        if self.trainable_scope == REMOTE_STAGE_TRAINABLE_SCOPE_BLOCKS and self.network is not None:
            raise ValueError("--trainable_scope blocks cannot be combined with --network_module; use --trainable_scope lora")

    def _reported_trainable_scope(self) -> str:
        if not self.trainable:
            return REMOTE_STAGE_TRAINABLE_SCOPE_FROZEN
        return self.trainable_scope

    def _autocast_context(self):
        if self.device.type == "cuda" and self.autocast_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _configure_trainable_params(self) -> None:
        blocks = getattr(self.model, "transformer_blocks")
        for block in blocks:
            block.requires_grad_(False)
        if self.network is not None:
            self.network.train()
            self.network.requires_grad_(bool(self.trainable))
        stage_blocks = blocks[self.start_index : self.end_index]
        if not self.trainable:
            return
        if self.learning_rate is None or float(self.learning_rate) <= 0:
            raise ValueError("--trainable remote stage server requires --learning_rate > 0")
        if self.trainable_scope == REMOTE_STAGE_TRAINABLE_SCOPE_LORA:
            if self.network_optimizer_params:
                params_or_groups: list[dict[str, Any]] | list[torch.nn.Parameter] = self.network_optimizer_params
                params = [
                    param
                    for group in self.network_optimizer_params
                    for param in group.get("params", [])
                    if isinstance(param, torch.nn.Parameter) and param.requires_grad
                ]
            else:
                params = [param for param in self.network.parameters() if param.requires_grad]
                params_or_groups = params
            optimizer_scope = REMOTE_STAGE_TRAINABLE_SCOPE_LORA
        else:
            for param in stage_blocks.parameters():
                param.requires_grad_(param.dtype in TRAINABLE_PARAM_DTYPES)
            params = [
                param
                for param in stage_blocks.parameters()
                if param.requires_grad and param.dtype in TRAINABLE_PARAM_DTYPES
            ]
            params_or_groups = params
            optimizer_scope = REMOTE_STAGE_TRAINABLE_SCOPE_BLOCKS
        if not params:
            raise RuntimeError(f"remote stage has no trainable {optimizer_scope} parameters")
        self.optimizer = torch.optim.AdamW(params_or_groups, lr=float(self.learning_rate), weight_decay=self.weight_decay)
        logger.info(
            "LTX2 remote stage optimizer enabled: scope=%s blocks=%d:%d params=%d lr=%g weight_decay=%g max_grad_norm=%g",
            optimizer_scope,
            self.start_index,
            self.end_index,
            sum(param.numel() for param in params),
            float(self.learning_rate),
            self.weight_decay,
            self.max_grad_norm,
        )

    def serve_forever(self) -> None:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.bind, self.port))
        server_sock.listen(1)
        logger.info(
            "LTX2 remote stage server listening on %s:%d split=%d device=%s",
            self.bind,
            self.port,
            self.start_index,
            self.device,
        )
        try:
            while True:
                conn, addr = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info("LTX2 remote stage client connected: %s", addr)
                try:
                    self._serve_connection(conn)
                except Exception:
                    logger.exception("LTX2 remote stage connection failed")
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        finally:
            server_sock.close()

    def _serve_connection(self, conn: socket.socket) -> None:
        self._clear_session_caches()
        while True:
            try:
                message, received_bytes = _recv_message_with_size(conn)
            except EOFError:
                return
            if not isinstance(message, dict):
                sent_bytes = _send_message(conn, {"ok": False, "error": f"expected dict message, got {type(message)}"})
                _accumulate_transport_stats(
                    self.stats,
                    kind="invalid",
                    sent_bytes=sent_bytes,
                    received_bytes=received_bytes,
                )
                continue
            kind = str(message.get("kind", "unknown"))
            try:
                response = self._handle_message(message)
            except Exception as exc:
                logger.exception("Remote stage request failed: %s", message.get("kind"))
                response = {"ok": False, "error": str(exc)}
            sent_bytes = _send_message(conn, response)
            if kind != "reset_stats":
                _accumulate_transport_stats(
                    self.stats,
                    kind=kind,
                    sent_bytes=sent_bytes,
                    received_bytes=received_bytes,
                )

    def _handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        kind = message.get("kind")
        if kind == "hello":
            requested_split = int(message.get("split_index", -1))
            requested_end = message.get("end_index")
            if requested_split != self.start_index:
                return {
                    "ok": False,
                    "error": f"split mismatch: client={requested_split}, server={self.start_index}",
                    "split_index": self.start_index,
                    "end_index": self.end_index,
                }
            if requested_end is not None and int(requested_end) != int(self.end_index):
                return {
                    "ok": False,
                    "error": f"end mismatch: client={requested_end}, server={self.end_index}",
                    "split_index": self.start_index,
                    "end_index": self.end_index,
                }
            self.metadata_cache_enabled = bool(message.get("metadata_cache_enabled", True))
            self.metadata_cache_size = int(message.get("metadata_cache_size", self.metadata_cache_size) or self.metadata_cache_size)
            self.aq_stochastic = bool(message.get("aq_stochastic", self.aq_stochastic))
            aq_cache_size = message.get("aq_cache_size", self.aq_cache_size)
            self.aq_cache_size = int(0 if aq_cache_size is None else aq_cache_size)
            if self.metadata_cache_size <= 0:
                return {
                    "ok": False,
                    "error": "metadata_cache_size must be > 0",
                    "split_index": self.start_index,
                    "end_index": self.end_index,
                }
            if self.aq_cache_size < 0:
                return {
                    "ok": False,
                    "error": "aq_cache_size must be >= 0",
                    "split_index": self.start_index,
                    "end_index": self.end_index,
                }
            return {
                "ok": True,
                "kind": "hello_ack",
                "split_index": self.start_index,
                "start_index": self.start_index,
                "end_index": self.end_index,
                "trainable": self.trainable,
                "trainable_scope": self._reported_trainable_scope(),
                "metadata_cache_enabled": self.metadata_cache_enabled,
                "metadata_cache_size": self.metadata_cache_size,
                "aq_stochastic": self.aq_stochastic,
                "aq_cache_size": self.aq_cache_size,
            }
        if kind == "forward":
            return self._handle_forward(message)
        if kind == "backward":
            return self._handle_backward(message)
        if kind == "zero_grad":
            return self._handle_zero_grad()
        if kind == "optimizer_step":
            return self._handle_optimizer_step()
        if kind == "save_state":
            return self._handle_save_state(message)
        if kind == "load_state":
            return self._handle_load_state(message)
        if kind == "get_stats":
            return {"ok": True, "kind": "stats_response", "stats": self._stats_snapshot()}
        if kind == "reset_stats":
            self.stats = _new_remote_stage_server_stats()
            self._clear_session_caches()
            return {"ok": True, "kind": "reset_stats_response"}
        if kind == "shutdown":
            raise EOFError("shutdown requested")
        return {"ok": False, "error": f"unknown message kind: {kind!r}"}

    def _clear_session_caches(self) -> None:
        self.contexts.clear()
        self._metadata_arg_cache.clear()
        self._metadata_cache_keys.clear()
        self._aq_encode_cache.clear()
        self._aq_decode_cache.clear()

    def _stats_snapshot(self) -> dict[str, Any]:
        stats = dict(self.stats)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            stats.update(
                {
                    "cuda_device": str(self.device),
                    "cuda_name": torch.cuda.get_device_name(self.device),
                    "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                }
            )
        return stats

    def _handle_forward(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id = str(message["request_id"])
        codec = str(message.get("codec", MP_CODEC_NONE) or MP_CODEC_NONE)
        block_size = int(message.get("int8_block_size", self.int8_block_size) or self.int8_block_size)
        cache_key = _normalize_cache_key(message.get("cache_key"))
        sample_cache_keys = _normalize_sample_cache_keys(message.get("sample_cache_keys"))
        activation_cache_key: Any | None = (
            {"batch": cache_key, "samples": sample_cache_keys}
            if sample_cache_keys
            else cache_key
        )
        aq_stochastic = bool(message.get("aq_stochastic", self.aq_stochastic))
        aq_cache_size_value = message.get("aq_cache_size", self.aq_cache_size)
        aq_cache_size = int(0 if aq_cache_size_value is None else aq_cache_size_value)
        if aq_cache_size < 0:
            raise ValueError("aq_cache_size must be >= 0")
        video = _unpack_transformer_args_without_x(
            self._resolve_transformer_args_payload(message.get("video")),
            self.device,
        )
        audio = _unpack_transformer_args_without_x(
            self._resolve_transformer_args_payload(message.get("audio")),
            self.device,
        )
        if video is not None:
            video_x = _decode_activation_tensor(
                message["video_x"],
                self.device,
                cache=self._aq_decode_cache,
                max_cache_size=aq_cache_size,
                stats=self.stats,
            ).detach().requires_grad_(True)
            video = replace(video, x=video_x)
        else:
            video_x = None
        if audio is not None:
            audio_x = _decode_activation_tensor(
                message["audio_x"],
                self.device,
                cache=self._aq_decode_cache,
                max_cache_size=aq_cache_size,
                stats=self.stats,
            ).detach().requires_grad_(True)
            audio = replace(audio, x=audio_x)
        else:
            audio_x = None
        perturbations = _to_device_tree(message.get("perturbations"), self.device)

        start_time = time.perf_counter()
        with torch.enable_grad(), self._autocast_context():
            for block in self.model.transformer_blocks[self.start_index : self.end_index]:
                video, audio = block(video, audio, perturbations)

        video_out_x = video.x if video is not None else None
        audio_out_x = audio.x if audio is not None else None
        self.contexts[request_id] = {
            "video_x": video_x,
            "audio_x": audio_x,
            "video_out_x": video_out_x,
            "audio_out_x": audio_out_x,
        }
        elapsed = time.perf_counter() - start_time
        self.stats["forward_requests"] = int(self.stats["forward_requests"]) + 1
        self.stats["forward_seconds"] = float(self.stats["forward_seconds"]) + elapsed
        logger.debug("Remote stage forward stored context %s codec=%s elapsed=%.4fs", request_id, codec, elapsed)
        return {
            "ok": True,
            "kind": "forward_response",
            "request_id": request_id,
            "video_x": (
                _encode_activation_tensor(
                    video_out_x.detach(),
                    codec,
                    block_size,
                    cache=self._aq_encode_cache,
                    cache_key=_activation_cache_key(self, activation_cache_key, "forward_out", "video"),
                    max_cache_size=aq_cache_size,
                    stochastic=aq_stochastic,
                    stats=self.stats,
                )
                if video_out_x is not None
                else None
            ),
            "audio_x": (
                _encode_activation_tensor(
                    audio_out_x.detach(),
                    codec,
                    block_size,
                    cache=self._aq_encode_cache,
                    cache_key=_activation_cache_key(self, activation_cache_key, "forward_out", "audio"),
                    max_cache_size=aq_cache_size,
                    stochastic=aq_stochastic,
                    stats=self.stats,
                )
                if audio_out_x is not None
                else None
            ),
        }

    def _resolve_transformer_args_payload(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        cache_key = payload.get("__cache_key__") if isinstance(payload, dict) else None
        if not cache_key:
            return payload
        cache_key = str(cache_key)
        static_payload = payload.get("static")
        if static_payload is not None:
            self._metadata_arg_cache[cache_key] = static_payload
            _remember_lru_key(self._metadata_cache_keys, cache_key, max_size=self.metadata_cache_size)
            _trim_mapping_to_lru(self._metadata_arg_cache, self._metadata_cache_keys)
            self.stats["metadata_cache_misses"] = int(self.stats.get("metadata_cache_misses", 0)) + 1
        elif cache_key in self._metadata_arg_cache:
            _remember_lru_key(self._metadata_cache_keys, cache_key, max_size=self.metadata_cache_size)
            self.stats["metadata_cache_hits"] = int(self.stats.get("metadata_cache_hits", 0)) + 1
        else:
            raise RuntimeError(f"missing remote-stage TransformerArgs metadata cache entry: {cache_key}")
        merged = dict(self._metadata_arg_cache[cache_key])
        merged.update(payload.get("dynamic") or {})
        return merged

    def _handle_backward(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id = str(message["request_id"])
        context = self.contexts.pop(request_id, None)
        if context is None:
            raise RuntimeError(f"unknown or already-freed remote stage context: {request_id}")
        grad_codec = str(message.get("grad_codec", MP_CODEC_NONE) or MP_CODEC_NONE)
        block_size = int(message.get("int8_block_size", self.int8_block_size) or self.int8_block_size)
        grad_stochastic = bool(message.get("grad_stochastic", False))
        outputs: list[torch.Tensor] = []
        grad_outputs: list[torch.Tensor] = []
        if context["video_out_x"] is not None and message.get("video_grad") is not None:
            outputs.append(context["video_out_x"])
            grad_outputs.append(_decode_tensor(message["video_grad"], self.device))
        if context["audio_out_x"] is not None and message.get("audio_grad") is not None:
            outputs.append(context["audio_out_x"])
            grad_outputs.append(_decode_tensor(message["audio_grad"], self.device))
        start_time = time.perf_counter()
        if outputs:
            torch.autograd.backward(outputs, grad_outputs)
        video_grad = context["video_x"].grad if context["video_x"] is not None else None
        audio_grad = context["audio_x"].grad if context["audio_x"] is not None else None
        elapsed = time.perf_counter() - start_time
        self.stats["backward_requests"] = int(self.stats["backward_requests"]) + 1
        self.stats["backward_seconds"] = float(self.stats["backward_seconds"]) + elapsed
        logger.debug("Remote stage backward completed context %s codec=%s elapsed=%.4fs", request_id, grad_codec, elapsed)
        return {
            "ok": True,
            "kind": "backward_response",
            "request_id": request_id,
            "video_grad": (
                _encode_tensor(video_grad, grad_codec, block_size, stochastic=grad_stochastic)
                if video_grad is not None
                else None
            ),
            "audio_grad": (
                _encode_tensor(audio_grad, grad_codec, block_size, stochastic=grad_stochastic)
                if audio_grad is not None
                else None
            ),
        }

    def _handle_zero_grad(self) -> dict[str, Any]:
        if self.optimizer is None:
            return {"ok": True, "kind": "zero_grad_response", "trainable": False}
        self.optimizer.zero_grad(set_to_none=True)
        return {"ok": True, "kind": "zero_grad_response", "trainable": True}

    def _handle_optimizer_step(self) -> dict[str, Any]:
        if self.optimizer is None:
            return {"ok": True, "kind": "optimizer_step_response", "trainable": False, "stepped": False}
        start_time = time.perf_counter()
        grad_norm = None
        if self.max_grad_norm and self.max_grad_norm > 0:
            params = [param for group in self.optimizer.param_groups for param in group["params"] if param.grad is not None]
            if params:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                grad_norm = float(grad_norm_tensor.detach().cpu().item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.perf_counter() - start_time
        self.stats["optimizer_steps"] = int(self.stats["optimizer_steps"]) + 1
        self.stats["optimizer_seconds"] = float(self.stats["optimizer_seconds"]) + elapsed
        return {
            "ok": True,
            "kind": "optimizer_step_response",
            "trainable": True,
            "stepped": True,
            "grad_norm": grad_norm,
            "elapsed_seconds": elapsed,
        }

    def _stage_checkpoint_path(self, message: dict[str, Any]) -> str:
        checkpoint_dir = str(message.get("checkpoint_dir") or "")
        checkpoint_name = str(message.get("checkpoint_name") or "remote_stage.pt")
        stage_index = int(message.get("stage_index", 0))
        if not checkpoint_dir:
            raise ValueError("save_state/load_state requires checkpoint_dir")
        base_name = os.path.basename(checkpoint_name)
        if base_name.endswith(".safetensors"):
            base_name = base_name[: -len(".safetensors")]
        elif base_name.endswith(".pt"):
            base_name = base_name[: -len(".pt")]
        filename = f"{base_name}.remote_stage_{stage_index}_{self.start_index}_{self.end_index}.pt"
        return os.path.join(checkpoint_dir, filename)

    def _stage_adapter_path(self, message: dict[str, Any]) -> str:
        path = self._stage_checkpoint_path(message)
        return path[: -len(".pt")] + ".safetensors" if path.endswith(".pt") else f"{path}.safetensors"

    def _stage_state_dict(self) -> dict[str, Any]:
        blocks = self.model.transformer_blocks[self.start_index : self.end_index]
        has_network = self.network is not None
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "trainable": self.trainable,
            "trainable_scope": self._reported_trainable_scope(),
            "model_state_dict": None if has_network else blocks.state_dict(),
            "network_state_dict": self.network.state_dict() if has_network else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "stats": dict(self.stats),
        }

    def _handle_save_state(self, message: dict[str, Any]) -> dict[str, Any]:
        path = self._stage_checkpoint_path(message)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._stage_state_dict(), path)
        adapter_path = None
        if self.network is not None and hasattr(self.network, "save_weights"):
            adapter_path = self._stage_adapter_path(message)
            self.network.save_weights(adapter_path, torch.float32, {})
        return {
            "ok": True,
            "kind": "save_state_response",
            "path": path,
            "adapter_path": adapter_path,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "trainable": self.trainable,
            "trainable_scope": self._reported_trainable_scope(),
        }

    def _handle_load_state(self, message: dict[str, Any]) -> dict[str, Any]:
        path = self._stage_checkpoint_path(message)
        state = torch.load(path, map_location=self.device)
        if int(state.get("start_index", -1)) != self.start_index or int(state.get("end_index", -1)) != self.end_index:
            raise RuntimeError(
                f"remote stage checkpoint range mismatch: file={state.get('start_index')}:{state.get('end_index')} "
                f"server={self.start_index}:{self.end_index}"
            )
        network_state = state.get("network_state_dict")
        model_state = state.get("model_state_dict")
        if network_state is not None:
            if self.network is None:
                raise RuntimeError("remote stage checkpoint contains a network state, but this server has no network")
            self.network.load_state_dict(network_state)
        elif model_state is not None:
            blocks = self.model.transformer_blocks[self.start_index : self.end_index]
            blocks.load_state_dict(model_state)
        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)
        loaded_stats = state.get("stats")
        if isinstance(loaded_stats, dict):
            self.stats.update(loaded_stats)
        return {
            "ok": True,
            "kind": "load_state_response",
            "path": path,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "trainable": self.trainable,
            "trainable_scope": self._reported_trainable_scope(),
        }


def _new_remote_stage_transport_stats() -> dict[str, float | int]:
    return {
        "tcp_messages_sent": 0,
        "tcp_messages_received": 0,
        "tcp_bytes_sent": 0,
        "tcp_bytes_received": 0,
        "tcp_request_seconds": 0.0,
        "tcp_request_failures": 0,
    }


def _new_remote_stage_server_stats() -> dict[str, float | int]:
    stats = _new_remote_stage_transport_stats()
    stats.update(
        {
            "forward_requests": 0,
            "backward_requests": 0,
            "optimizer_steps": 0,
            "forward_seconds": 0.0,
            "backward_seconds": 0.0,
            "optimizer_seconds": 0.0,
            "metadata_cache_hits": 0,
            "metadata_cache_misses": 0,
            "aq_refreshes": 0,
            "aq_deltas": 0,
            "aq_fallbacks": 0,
        }
    )
    return stats


def _accumulate_transport_stats(
    stats: dict[str, float | int],
    *,
    kind: str,
    sent_bytes: int,
    received_bytes: int,
    elapsed: float | None = None,
) -> None:
    stats["tcp_messages_sent"] = int(stats.get("tcp_messages_sent", 0)) + 1
    stats["tcp_messages_received"] = int(stats.get("tcp_messages_received", 0)) + 1
    stats["tcp_bytes_sent"] = int(stats.get("tcp_bytes_sent", 0)) + int(sent_bytes)
    stats["tcp_bytes_received"] = int(stats.get("tcp_bytes_received", 0)) + int(received_bytes)
    if elapsed is not None:
        stats["tcp_request_seconds"] = float(stats.get("tcp_request_seconds", 0.0)) + float(elapsed)
    safe_kind = str(kind or "unknown").replace(" ", "_").replace("-", "_")
    sent_key = f"{safe_kind}_bytes_sent"
    received_key = f"{safe_kind}_bytes_received"
    count_key = f"{safe_kind}_tcp_messages"
    stats[sent_key] = int(stats.get(sent_key, 0)) + int(sent_bytes)
    stats[received_key] = int(stats.get(received_key, 0)) + int(received_bytes)
    stats[count_key] = int(stats.get(count_key, 0)) + 1


def _send_message(sock: socket.socket, message: Any) -> int:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)
    return len(payload) + 8


def _configure_remote_stage_socket(sock: socket.socket, timeout: float) -> None:
    sock.settimeout(float(timeout))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, "SIO_KEEPALIVE_VALS"):
        idle_ms = int(max(5.0, min(60.0, float(timeout) / 3.0)) * 1000)
        interval_ms = int(max(1.0, min(10.0, float(timeout) / 12.0)) * 1000)
        with contextlib.suppress(OSError):
            sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, idle_ms, interval_ms))
    else:
        for option_name, value in (
            ("TCP_KEEPIDLE", int(max(5.0, min(60.0, float(timeout) / 3.0)))),
            ("TCP_KEEPINTVL", int(max(1.0, min(10.0, float(timeout) / 12.0)))),
            ("TCP_KEEPCNT", 3),
        ):
            option = getattr(socket, option_name, None)
            if option is not None:
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_TCP, option, value)


def _recv_message(sock: socket.socket) -> Any:
    message, _wire_bytes = _recv_message_with_size(sock)
    return message


def _recv_message_with_size(sock: socket.socket) -> tuple[Any, int]:
    header = _recv_exact(sock, 8)
    if not header:
        raise EOFError("connection closed")
    (size,) = struct.unpack("!Q", header)
    if size <= 0:
        raise EOFError("empty message")
    return pickle.loads(_recv_exact(sock, size)), size + 8


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            if chunks:
                raise EOFError("connection closed mid-message")
            return b""
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _normalize_cache_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_sample_cache_keys(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    keys = [_normalize_cache_key(item) for item in value]
    return [key for key in keys if key is not None]


def _normalize_remote_stage_cache_key(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        batch_key = _normalize_cache_key(value.get("batch"))
        sample_keys = _normalize_sample_cache_keys(value.get("samples"))
        if batch_key is None and sample_keys:
            batch_key = _stable_tree_digest({"samples": sample_keys})
        if batch_key is None:
            return None
        return {"batch": batch_key, "samples": sample_keys}
    batch_key = _normalize_cache_key(value)
    if batch_key is None:
        return None
    return {"batch": batch_key, "samples": []}


def _remote_stage_batch_cache_key(value: Any) -> str | None:
    normalized = _normalize_remote_stage_cache_key(value)
    if normalized is None:
        return None
    return _normalize_cache_key(normalized.get("batch"))


def _remote_stage_sample_cache_keys(value: Any) -> list[str]:
    normalized = _normalize_remote_stage_cache_key(value)
    if normalized is None:
        return []
    return _normalize_sample_cache_keys(normalized.get("samples"))


def _remember_lru_key(cache_keys: OrderedDict[str, None], key: str, *, max_size: int) -> None:
    if key in cache_keys:
        cache_keys.move_to_end(key)
    else:
        cache_keys[key] = None
    while len(cache_keys) > int(max_size):
        cache_keys.popitem(last=False)


def _trim_mapping_to_lru(mapping: dict[str, Any], cache_keys: OrderedDict[str, None]) -> None:
    live = set(cache_keys.keys())
    for key in list(mapping.keys()):
        if key not in live:
            mapping.pop(key, None)


def _get_aq_cache_tensor(cache: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    value = cache.get(key)
    if value is not None and isinstance(cache, OrderedDict):
        cache.move_to_end(key)
    return value


def _store_aq_cache_tensor(
    cache: dict[str, torch.Tensor],
    key: str,
    tensor: torch.Tensor,
    *,
    max_size: int = 0,
) -> None:
    cache[key] = tensor
    if isinstance(cache, OrderedDict):
        cache.move_to_end(key)
    max_size = int(max_size)
    if max_size <= 0:
        return
    while len(cache) > max_size:
        if isinstance(cache, OrderedDict):
            cache.popitem(last=False)
        else:
            cache.pop(next(iter(cache)))


def _hash_update_tree(hasher: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor")
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        try:
            hasher.update(tensor.numpy().tobytes())
        except TypeError:
            hasher.update(tensor.to(torch.float32).numpy().tobytes())
        return
    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value.keys(), key=str):
            hasher.update(str(key).encode("utf-8"))
            _hash_update_tree(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"list" if isinstance(value, list) else b"tuple")
        for item in value:
            _hash_update_tree(hasher, item)
        return
    hasher.update(repr(value).encode("utf-8"))


def _stable_tree_digest(value: Any, *, digest_size: int = 16) -> str:
    hasher = hashlib.blake2b(digest_size=digest_size)
    _hash_update_tree(hasher, value)
    return hasher.hexdigest()


def _resolve_batch_identity_tokens(batch: dict[str, Any]) -> list[str]:
    for key in ("item_keys", "latent_cache_paths", "text_cache_paths", "captions"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)) and value:
            tokens = [str(item) for item in value if item is not None]
            if tokens:
                return tokens
    latents = batch.get("latents")
    if isinstance(latents, dict):
        for key in ("item_keys", "latent_cache_paths", "text_cache_paths"):
            value = latents.get(key)
            if isinstance(value, (list, tuple)) and value:
                tokens = [str(item) for item in value if item is not None]
            if tokens:
                return tokens
    return []


def _tensor_has_batch_items(tensor: torch.Tensor | None, count: int) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.ndim > 0 and int(tensor.shape[0]) == int(count)


def _build_ltx2_remote_stage_sample_cache_keys(
    mode: str,
    tokens: list[str],
    *,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> list[str]:
    count = len(tokens)
    if count <= 0:
        return []
    has_timestep_items = _tensor_has_batch_items(timesteps, count)
    has_noise_items = _tensor_has_batch_items(noise, count)
    if mode in {"sample_timestep", "sample_timestep_noise"} and timesteps is not None and not has_timestep_items:
        return []
    if mode == "sample_timestep_noise" and noise is not None and not has_noise_items:
        return []

    sample_keys: list[str] = []
    for idx, token in enumerate(tokens):
        payload: dict[str, Any] = {
            "mode": mode,
            "items": [token],
        }
        if mode in {"sample_timestep", "sample_timestep_noise"} and has_timestep_items:
            payload["timesteps"] = timesteps.detach()[idx]
        if mode == "sample_timestep_noise" and has_noise_items:
            payload["noise"] = noise.detach()[idx]
        sample_keys.append(_stable_tree_digest(payload))
    return sample_keys


def _split_transformer_args_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    static_payload: dict[str, Any] = {}
    dynamic_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _ARG_STATIC_FIELDS:
            static_payload[key] = value
        else:
            dynamic_payload[key] = value
    return static_payload, dynamic_payload


def _prepare_transformer_args_payload(
    client: LTX2RemoteStageClient,
    modality: str,
    cache_key: str | None,
    args: TransformerArgs | None,
) -> dict[str, Any] | None:
    if args is None:
        return None
    payload = _pack_transformer_args_without_x(args)
    if not client.metadata_cache_enabled or cache_key is None:
        return payload

    static_payload, dynamic_payload = _split_transformer_args_payload(payload)
    static_digest = _stable_tree_digest(static_payload)
    remote_key = f"{client.start_index}:{client.end_index}:{modality}:{cache_key}:{static_digest}"
    send_static = remote_key not in client._metadata_cache_keys
    _remember_lru_key(client._metadata_cache_keys, remote_key, max_size=client.metadata_cache_size)
    return {
        "__cache_key__": remote_key,
        "static": static_payload if send_static else None,
        "dynamic": dynamic_payload,
    }


def _aq_base_codec(codec: str) -> str | None:
    codec = str(codec or MP_CODEC_NONE).lower()
    if codec == REMOTE_STAGE_CODEC_AQ_INT8:
        return MP_CODEC_INT8
    if codec == REMOTE_STAGE_CODEC_AQ_INT4:
        return MP_CODEC_INT4
    return None


def _activation_cache_key(
    owner: Any,
    cache_key: Any | None,
    direction: str,
    name: str,
) -> Any | None:
    if isinstance(cache_key, dict):
        batch_key = _activation_cache_key(owner, cache_key.get("batch"), direction, name)
        sample_keys = [
            _activation_cache_key(owner, key, direction, name)
            for key in _normalize_sample_cache_keys(cache_key.get("samples"))
        ]
        return {"batch": batch_key, "samples": [key for key in sample_keys if key is not None]}
    normalized_key = _normalize_cache_key(cache_key)
    if normalized_key is None:
        return None
    start_index = int(getattr(owner, "start_index", getattr(owner, "split_index", -1)))
    end_index = getattr(owner, "end_index", None)
    return f"{start_index}:{end_index}:{direction}:{name}:{normalized_key}"


def _cache_compatible(cached: torch.Tensor | None, tensor: torch.Tensor) -> bool:
    return (
        isinstance(cached, torch.Tensor)
        and tuple(cached.shape) == tuple(tensor.shape)
        and cached.dtype == tensor.dtype
    )


def _encode_activation_tensor(
    tensor: torch.Tensor | None,
    codec: str,
    block_size: int,
    *,
    cache: dict[str, torch.Tensor],
    cache_key: str | None,
    max_cache_size: int = 0,
    stochastic: bool = False,
    stats: dict[str, float | int] | None = None,
) -> dict[str, Any] | None:
    if tensor is None:
        return None
    base_codec = _aq_base_codec(codec)
    if base_codec is None or not torch.is_floating_point(tensor):
        return _encode_tensor(tensor, codec, block_size)
    sample_cache_keys = _remote_stage_sample_cache_keys(cache_key)
    if sample_cache_keys and tensor.ndim > 0 and int(tensor.shape[0]) == len(sample_cache_keys):
        # AQ-SGD caches by training sample. For normal batched tensors, encode
        # each first-dimension sample against its own cached activation. If the
        # tensor layout does not expose batch as dim 0, fall back to the whole
        # batch key below.
        return {
            "codec": str(codec).lower(),
            "aq_mode": "batch",
            "items": [
                _encode_activation_tensor(
                    tensor[idx : idx + 1],
                    codec,
                    block_size,
                    cache=cache,
                    cache_key=sample_key,
                    max_cache_size=max_cache_size,
                    stochastic=stochastic,
                    stats=stats,
                )
                for idx, sample_key in enumerate(sample_cache_keys)
            ],
            "orig_shape": tuple(tensor.shape),
        }
    cache_key = _remote_stage_batch_cache_key(cache_key)
    if cache_key is None:
        _bump_aq_stats(stats, "fallback")
        return _encode_tensor(tensor, base_codec, block_size, stochastic=stochastic)

    cached = _get_aq_cache_tensor(cache, cache_key)
    if _cache_compatible(cached, tensor):
        previous = cached.to(device=tensor.device, dtype=tensor.dtype)
        inner = _encode_tensor(tensor.detach() - previous, base_codec, block_size, stochastic=stochastic)
        decoded_delta = _decode_tensor(inner, tensor.device)
        if decoded_delta is not None:
            _store_aq_cache_tensor(
                cache,
                cache_key,
                (previous + decoded_delta).detach().cpu(),
                max_size=max_cache_size,
            )
        mode = "delta"
    else:
        # AQ-SGD sends the first activation for a keyed sample without
        # quantization, then stores that exact value as the delta base.
        inner = _encode_tensor(tensor, MP_CODEC_NONE, block_size)
        decoded = _decode_tensor(inner, tensor.device)
        if decoded is not None:
            _store_aq_cache_tensor(cache, cache_key, decoded.detach().cpu(), max_size=max_cache_size)
        mode = "refresh"
    _bump_aq_stats(stats, mode)

    return {
        "codec": str(codec).lower(),
        "aq_mode": mode,
        "aq_key": cache_key,
        "inner": inner,
    }


def _decode_activation_tensor(
    payload: dict[str, Any] | None,
    device: torch.device | str,
    *,
    cache: dict[str, torch.Tensor],
    max_cache_size: int = 0,
    stats: dict[str, float | int] | None = None,
) -> torch.Tensor | None:
    if payload is None:
        return None
    codec = str(payload.get("codec", MP_CODEC_NONE)).lower()
    if codec not in REMOTE_STAGE_AQ_CODECS:
        return _decode_tensor(payload, device)
    if str(payload.get("aq_mode", "refresh")) == "batch":
        items = payload.get("items")
        if not isinstance(items, (list, tuple)):
            raise RuntimeError("AQ batch payload is missing item payloads")
        decoded_items = [
            _decode_activation_tensor(
                item,
                device,
                cache=cache,
                max_cache_size=max_cache_size,
                stats=stats,
            )
            for item in items
        ]
        if any(item is None for item in decoded_items):
            return None
        out = torch.cat([item for item in decoded_items if item is not None], dim=0)
        orig_shape = tuple(payload.get("orig_shape", tuple(out.shape)))
        return out.reshape(orig_shape)

    key = _normalize_cache_key(payload.get("aq_key"))
    inner = payload.get("inner")
    decoded = _decode_tensor(inner, device)
    if key is None or decoded is None:
        return decoded

    mode = str(payload.get("aq_mode", "refresh"))
    if mode == "delta":
        cached = _get_aq_cache_tensor(cache, key)
        if not _cache_compatible(cached, decoded):
            raise RuntimeError(f"missing or incompatible AQ-style activation cache entry: {key}")
        out = cached.to(device=device, dtype=decoded.dtype) + decoded
    else:
        out = decoded
    _store_aq_cache_tensor(cache, key, out.detach().cpu(), max_size=max_cache_size)
    _bump_aq_stats(stats, mode)
    return out


def _bump_aq_stats(stats: dict[str, float | int] | None, mode: str) -> None:
    if stats is None:
        return
    if mode == "delta":
        key = "aq_deltas"
    elif mode == "refresh":
        key = "aq_refreshes"
    else:
        key = "aq_fallbacks"
    stats[key] = int(stats.get(key, 0)) + 1


def _encode_tensor(
    tensor: torch.Tensor | None,
    codec: str,
    block_size: int,
    *,
    stochastic: bool = False,
) -> dict[str, Any] | None:
    if tensor is None:
        return None
    codec = str(codec or MP_CODEC_NONE).lower()
    if codec == MP_CODEC_INT8 and torch.is_floating_point(tensor):
        q, scale, orig_numel, orig_shape, dtype = quantize_int8_blocks_for_ltx2_mp(
            tensor.detach(),
            block_size,
            stochastic=stochastic,
        )
        return {
            "codec": MP_CODEC_INT8,
            "stochastic": bool(stochastic),
            "q": q.cpu(),
            "scale": scale.cpu(),
            "orig_numel": orig_numel,
            "orig_shape": orig_shape,
            "dtype": dtype,
        }
    if codec == MP_CODEC_INT4 and torch.is_floating_point(tensor):
        packed, scale, orig_numel, orig_shape, dtype = quantize_int4_blocks_for_ltx2_mp(
            tensor.detach(),
            block_size,
            stochastic=stochastic,
        )
        return {
            "codec": MP_CODEC_INT4,
            "stochastic": bool(stochastic),
            "packed": packed.cpu(),
            "scale": scale.cpu(),
            "orig_numel": orig_numel,
            "orig_shape": orig_shape,
            "dtype": dtype,
        }
    return {
        "codec": MP_CODEC_NONE,
        "tensor": tensor.detach().cpu(),
    }


def _decode_tensor(payload: dict[str, Any] | None, device: torch.device | str) -> torch.Tensor | None:
    if payload is None:
        return None
    device = torch.device(device)
    codec = str(payload.get("codec", MP_CODEC_NONE))
    if codec == MP_CODEC_INT8:
        return dequantize_int8_blocks_for_ltx2_mp(
            payload["q"].to(device),
            payload["scale"].to(device),
            int(payload["orig_numel"]),
            tuple(payload["orig_shape"]),
            payload["dtype"],
        )
    if codec == MP_CODEC_INT4:
        return dequantize_int4_blocks_for_ltx2_mp(
            payload["packed"].to(device),
            payload["scale"].to(device),
            int(payload["orig_numel"]),
            tuple(payload["orig_shape"]),
            payload["dtype"],
        )
    return payload["tensor"].to(device)


def _pack_transformer_args_without_x(args: TransformerArgs | None) -> dict[str, Any] | None:
    if args is None:
        return None
    out: dict[str, Any] = {}
    for field in fields(TransformerArgs):
        if field.name == "x":
            continue
        out[field.name] = _to_cpu_tree(getattr(args, field.name))
    return out


def _unpack_transformer_args_without_x(payload: dict[str, Any] | None, device: torch.device) -> TransformerArgs | None:
    if payload is None:
        return None
    kwargs = {field.name: _to_device_tree(payload.get(field.name), device) for field in fields(TransformerArgs) if field.name != "x"}
    kwargs["x"] = torch.empty(0, device=device)
    return TransformerArgs(**kwargs)


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    return value


def _to_device_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device_tree(item, device) for key, item in value.items()}
    return value


def _first_arg_device(primary: TransformerArgs | None, fallback: TransformerArgs | None) -> torch.device:
    if primary is not None:
        return primary.x.device
    if fallback is not None:
        return fallback.x.device
    return torch.device("cpu")
