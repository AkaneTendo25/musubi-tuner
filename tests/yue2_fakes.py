"""Shared YuE2 test doubles (no weights, no CUDA).

* ``FakeOffloader`` / ``FakeOffloaderFactory``: a layout-simulating stand-in for ``ModelOffloader``. It moves no weights;
  it tracks which block indices are resident on the device, ports the forward-only strategies and the training-mode
  forward/backward swap schedule of ``modules/custom_offloading_utils.py`` (``ModelOffloader.create_backward_hook``,
  ``submit_move_blocks_forward``, ``prepare_block_devices_before_forward``), and raises when a block is used while
  not resident or a swap would move a block that is not where the schedule expects it. The real offloader cannot run
  on CPU (``torch.cuda.set_device`` in its worker), so every CPU block-swap test uses this fake;
  ``tests/test_yue2_fakes.py`` checks it against the real ``ModelOffloader`` on CUDA.
* ``FakeYuE2VAE``: a deterministic VAE with the ``YuE2VAE`` interface and a +-1-frame receptive field, so encoding a
  segment alone differs from slicing a whole-record encode at the segment edges.
* ``FakeTokenizer``: byte-level tokenizer (ids < 256 < EOD) with the ``YuE2TextTokenizer`` interface.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HOP = 1920
LATENT_DIM = 64


class FakeOffloader:
    """Layout-simulating ``ModelOffloader`` (same public methods and attributes).

    ``resident`` is the set of block indices whose weights would be on the device. ``events`` records
    ``("prepare",)``, ``("forward_only", flag)``, ``("wait", i)``, ``("move", to_cpu, to_device, "fwd"|"bwd")`` and
    ``("pass", forward_only)`` (at the first block of every pass), for asserting call order in tests.
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device = torch.device("cpu"),
        register_hooks: bool = True,
    ):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = torch.device(device)
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward
        self.prepared = False
        self.resident: set[int] = set(range(num_blocks))
        self.events: list[tuple] = []
        self.remove_handles = []
        if supports_backward and register_hooks:
            for i, block in enumerate(blocks):
                hook = self._create_backward_hook(i)
                if hook is not None:
                    self.remove_handles.append(block.register_full_backward_hook(hook))

    @property
    def initial_layout(self) -> set[int]:
        return set(range(self.num_blocks - self.blocks_to_swap))

    def _swapping(self) -> bool:
        return bool(self.blocks_to_swap)

    def _move(self, to_cpu: int, to_device: int, phase: str) -> None:
        if to_cpu not in self.resident:
            raise AssertionError(
                f"[{self.block_type}] {phase} swap moves block {to_cpu} to CPU but it is not resident (resident {sorted(self.resident)})"
            )
        if to_device in self.resident:
            raise AssertionError(
                f"[{self.block_type}] {phase} swap moves block {to_device} to the device but it is already resident"
                f" (resident {sorted(self.resident)})"
            )
        self.resident.discard(to_cpu)
        self.resident.add(to_device)
        self.events.append(("move", to_cpu, to_device, phase))

    def set_forward_only(self, forward_only: bool) -> None:
        self.forward_only = forward_only
        self.events.append(("forward_only", forward_only))

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]) -> None:
        if not self._swapping():
            return
        self.resident = self.initial_layout
        self.prepared = True
        self.events.append(("prepare",))

    def wait_for_block(self, block_idx: int) -> None:
        if not self._swapping():
            return
        if not self.prepared:
            raise AssertionError(f"[{self.block_type}] forward before prepare_block_devices_before_forward")
        if block_idx == 0:
            self.events.append(("pass", self.forward_only))
        self.events.append(("wait", block_idx))
        if block_idx not in self.resident:
            raise AssertionError(
                f"[{self.block_type}] block {block_idx} used while not resident (resident {sorted(self.resident)},"
                f" forward_only={self.forward_only})"
            )

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int) -> None:
        if not self._swapping():
            return
        n, s = self.num_blocks, self.blocks_to_swap
        if not self.forward_only:
            if block_idx >= s:
                return
            self._move(block_idx, (n - s + block_idx) % n, "fwd")
            return
        if s < n // 2:
            if s <= block_idx < n - s:
                return
            if block_idx < s:
                to_device = (n - s + block_idx) % n
            else:
                to_device = block_idx - (n - s)
        else:
            to_device = (n - s + block_idx) % n
        self._move(block_idx, to_device, "fwd")

    def _create_backward_hook(self, block_index: int):
        n, s = self.num_blocks, self.blocks_to_swap
        propagated = n - block_index - 1
        swapping = 0 < propagated <= s
        waiting = 0 < block_index <= s
        if not swapping and not waiting:
            return None
        to_cpu, to_device, to_wait = n - propagated, s - propagated, block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if swapping:
                self._move(to_cpu, to_device, "bwd")
            if waiting and to_wait not in self.resident:
                raise AssertionError(f"[{self.block_type}] backward reached block {to_wait} while it is not resident")
            return None

        return backward_hook

    def remove_hooks(self) -> None:
        for handle in self.remove_handles:
            handle.remove()
        self.remove_handles = []

    def passes(self) -> list[bool]:
        """``forward_only`` flag of every pass started so far, in order."""
        return [e[1] for e in self.events if e[0] == "pass"]


class FakeOffloaderFactory:
    """Drop-in for ``custom_offloading_utils.create_offloader``; monkeypatch it where the model looks it up.

    ``created`` maps ``block_type`` to the last ``FakeOffloader`` built for it.
    """

    def __init__(self, register_hooks: bool = True):
        self.register_hooks = register_hooks
        self.created: dict[str, FakeOffloader] = {}

    def __call__(self, block_type, blocks, num_blocks, blocks_to_swap, config):
        off = FakeOffloader(
            block_type,
            blocks,
            num_blocks,
            blocks_to_swap,
            config.supports_backward,
            config.device,
            register_hooks=self.register_hooks,
        )
        self.created[block_type] = off
        return off


class FakeYuE2VAE(nn.Module):
    """Deterministic ``YuE2VAE`` stand-in (FP32).

    Encode: per-frame channel means of ``[2, T*1920]`` audio, then a fixed 3-tap ``conv1d`` (zero padded) to 64
    channels, so each latent frame depends on its neighbours. Decode: least-squares inverse of the centre tap, each
    frame repeated 1920 times, trimmed to the reference length ``1920*T - 64``. ``encode_calls`` records the sample
    length of every encoder call.
    """

    hop = HOP
    latent_dim = LATENT_DIM

    def __init__(self, source_dtype: str = "float32", seed: int = 0, decoder_only: bool = False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(LATENT_DIM, 2, 3, generator=g) * 0.5)
        self.register_buffer("unproj", torch.linalg.pinv(self.proj[:, :, 1]))
        self.source_dtype = source_dtype
        self.fingerprint = f"fake-yue2-vae:{seed}:{source_dtype}"
        self.decoder_only = decoder_only
        self.encode_calls: list[int] = []

    def encode_mean(self, audio: torch.Tensor) -> torch.Tensor:
        if self.decoder_only:
            raise RuntimeError("encoder not loaded")
        if audio.ndim != 3 or audio.shape[1] != 2 or audio.shape[-1] % HOP != 0 or audio.shape[-1] == 0:
            raise ValueError(f"expected audio [B, 2, T*{HOP}], got {tuple(audio.shape)}")
        self.encode_calls.append(int(audio.shape[-1]))
        b, _, s = audio.shape
        frames = audio.float().reshape(b, 2, s // HOP, HOP).mean(-1)
        return F.conv1d(frames, self.proj, padding=1)

    def encode_mean_chunked(self, audio: torch.Tensor, chunk_frames: int = 750, overlap_frames: int = 50) -> torch.Tensor:
        if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[-1] % HOP != 0:
            raise ValueError(f"expected audio [2, T*{HOP}], got {tuple(audio.shape)}")
        total = audio.shape[-1] // HOP
        out = torch.empty(total, LATENT_DIM, dtype=torch.float32)
        for c in range(0, total, chunk_frames):
            lo, hi = max(0, c - overlap_frames), min(total, c + chunk_frames + overlap_frames)
            lat = self.encode_mean(audio[None, :, lo * HOP : hi * HOP])[0].T
            n = min(chunk_frames, total - c)
            out[c : c + n] = lat[c - lo : c - lo + n]
        return out

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1] != LATENT_DIM:
            raise ValueError(f"expected latents [B, {LATENT_DIM}, T], got {tuple(latent.shape)}")
        frames = torch.einsum("cl,blt->bct", self.unproj, latent.float())
        return frames.repeat_interleave(HOP, dim=-1)[..., : latent.shape[-1] * HOP - 64]

    def decode_tiled(self, latent, core_frames: int = 1024, halo_frames: int = 16, output_device="cpu") -> torch.Tensor:
        if halo_frames < self.required_halo(core_frames):
            raise ValueError("halo_frames too small")
        return self.decode(latent).to(output_device)

    def required_halo(self, core_frames: int) -> int:
        return 1


def load_fake_yue2_vae(
    vae_path: Optional[str] = None,
    dit_path: Optional[str] = None,
    *,
    device="cpu",
    decoder_only: bool = False,
    allow_fp16_source: bool = True,
    source_dtype: str = "float32",
) -> FakeYuE2VAE:
    """Same signature as ``yue2_checkpoint.load_yue2_vae`` (plus ``source_dtype``) for monkeypatching."""
    if source_dtype != "float32" and not allow_fp16_source:
        raise ValueError(f"YuE2 VAE source is {source_dtype}, not float32")
    return FakeYuE2VAE(source_dtype=source_dtype, decoder_only=decoder_only).to(device)


class FakeTokenizer:
    """Byte-level ``YuE2TextTokenizer`` stand-in: ``encode`` returns the UTF-8 bytes (all ids < EOD)."""

    backend = "fake"

    def __init__(self, salt: str = ""):
        self.fingerprint = "sha256:" + hashlib.sha256(("fake-tokenizer" + salt).encode()).hexdigest()

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", errors="replace")


# region tiny checkpoints (reference model -> HF / ComfyUI bf16 / ComfyUI int8 files)


def ref_config(tiny) -> "object":
    """Official ``YuE2Config`` with the dimensions of a ``yue2_model.YuE2Config`` (needs transformers)."""
    from yue2_ref.modeling_yue2 import YuE2Config as RefConfig

    return RefConfig(
        hidden_size=tiny.hidden_size,
        num_hidden_layers=tiny.num_layers,
        num_attention_heads=tiny.num_heads,
        num_key_value_heads=tiny.num_kv_heads,
        head_dim=tiny.head_dim,
        intermediate_size=tiny.intermediate_size,
        vocab_size=tiny.vocab_size,
        rms_norm_eps=tiny.rms_norm_eps,
        rope_theta=tiny.rope_theta,
        max_position_embeddings=tiny.max_position_embeddings,
        latent_dim=tiny.latent_dim,
        max_latent_frames=tiny.max_latent_frames,
    )


def make_ref_model(tiny, seed: int = 0):
    """Official ``YuE2ForCausalLM`` (fp32, eval) with random unit-gain weights and non-trivial norm scales."""
    from yue2_ref.modeling_yue2 import YuE2ForCausalLM

    torch.manual_seed(seed)
    model = YuE2ForCausalLM(ref_config(tiny)).float().eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5)
            elif name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return model


def write_safetensors_ordered(path, tensors: dict, metadata: Optional[dict] = None) -> str:
    """Write a safetensors file whose header lists the keys in ``tensors`` order (``save_file`` sorts them)."""
    import json
    import struct

    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    names = {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.int64: "I64",
    }
    for key, t in tensors.items():
        t = t.detach().contiguous().cpu()
        data = t.view(torch.uint8).numpy().tobytes() if t.numel() else b""
        header[key] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for data in blobs:
            f.write(data)
    return str(path)


def save_ref_hf(path, ref_model, dtype=torch.float32) -> str:
    """HF-layout file from the reference model (``save_file``: alphabetical header, so k_proj precedes q_proj)."""
    from safetensors.torch import save_file

    save_file({k: v.detach().to(dtype).contiguous() for k, v in ref_model.state_dict().items()}, str(path))
    return str(path)


def tiny_vae_configs() -> tuple[dict, dict]:
    strides = [2, 2, 4, 4, 5, 6]
    encoder = dict(in_channels=2, channels=4, c_mults=[1, 1, 1, 1, 1, 2], strides=strides, latent_dim=128, use_snake=True)
    decoder = dict(
        out_channels=2, channels=4, c_mults=[1, 1, 1, 1, 1, 2], strides=strides, latent_dim=64, use_snake=True, final_tanh=False
    )
    return encoder, decoder


def tiny_vae_state_dict(seed: int = 0) -> dict:
    """Random fp32 state dict of the tiny Oobleck VAE (legacy weight-norm keys)."""
    from musubi_tuner.yue2.yue2_vae import YuE2VAE

    torch.manual_seed(seed)
    vae = YuE2VAE(*tiny_vae_configs())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith(("alpha", "beta")):
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
            elif name.endswith("weight_g"):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            else:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
    return {k: v.clone() for k, v in vae.state_dict().items()}


def native_to_comfy(native_sd: dict, *, vae_sd: Optional[dict] = None, tokenizer_json: Optional[bytes] = None) -> dict:
    """ComfyUI all-in-one layout from native tensors, including the duplicate NAR final norm."""
    out = {}
    for key, t in native_sd.items():
        if key.startswith("ar.blocks."):
            out["text_encoders.model.layers." + key[len("ar.blocks.") :]] = t
        elif key.startswith("ar.embed_tokens."):
            out["text_encoders.model.embed_tokens." + key[len("ar.embed_tokens.") :]] = t
        elif key.startswith("ar.lm_head."):
            out["text_encoders.model.lm_head." + key[len("ar.lm_head.") :]] = t
        elif key == "norm.weight":
            out["text_encoders.model.norm.weight"] = t
            out["model.diffusion_model.model.norm.weight"] = t.clone()
        elif key.startswith("nar.blocks."):
            out["model.diffusion_model.model.layers." + key[len("nar.blocks.") :]] = t
        elif key.startswith("nar."):
            out["model.diffusion_model." + key[len("nar.") :]] = t
        else:
            raise KeyError(key)
    for key, t in (vae_sd or {}).items():
        out["vae." + key] = t
    if tokenizer_json is not None:
        out["text_encoders.yue2_tokenizer_json"] = torch.tensor(list(tokenizer_json), dtype=torch.uint8)
    return out


COMFY_INT8_MODULES = ("vae2llm", "llm2vae", "time_embedder.mlp.0", "time_embedder.mlp.2", "embed_tokens", "lm_head")


def comfy_quant_payload(group_size: int) -> torch.Tensor:
    import json

    raw = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}).encode()
    return torch.tensor(list(raw), dtype=torch.uint8)


def quantize_comfy(comfy_sd: dict, group_size: int = 256) -> dict:
    """ComfyUI pre-quantized ConvRot int8 variant: every 2-D weight whose in_features divides ``group_size`` becomes
    int8 + ``.weight_scale`` + ``.comfy_quant`` (block linears, embed, lm_head, llm2vae, time embedder)."""
    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight

    out = {}
    for key, t in comfy_sd.items():
        if key.startswith(("text_encoders.", "model.diffusion_model.")) and key.endswith(".weight") and t.ndim == 2:
            if t.shape[1] % group_size == 0:
                module = key[: -len(".weight")]
                q, s = quantize_int8_convrot_weight(t.float(), group_size)
                out[key] = q
                out[module + ".weight_scale"] = s.float()
                out[module + ".comfy_quant"] = comfy_quant_payload(group_size)
                continue
        out[key] = t.half() if key.startswith("vae.") else t
    return out


# endregion


TOKENIZER_CORPUS = [
    "[verse]\nhello darkness my old friend\nI've come to talk with you again",
    "[chorus]\nla la la, we're singing all night long! 123 456",
    "dream pop, female vocal, 92 bpm, reverb guitar",
    'X:1\nT:tune\nM:4/4\nK:C\n"C"CDEF|"G"GABc|',
    "你好世界 音乐 歌词 さくら 桜",
] * 20


def make_tiny_tokenizer_json(vocab_size: int = 500) -> bytes:
    """Byte-level BPE with the official pre-tokenizer regex, NFC, and the YuE2 added tokens (tokenizers JSON)."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    from musubi_tuner.yue2.yue2_tokenizer import TIKTOKEN_PATTERN

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(TIKTOKEN_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(TOKENIZER_CORPUS, trainer)
    tok.add_special_tokens([AddedToken("<|endoftext|>", special=True, normalized=False)])
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in ("<abc>", "</abc>", "<extra_0>")])
    return tok.to_str().encode("utf-8")
