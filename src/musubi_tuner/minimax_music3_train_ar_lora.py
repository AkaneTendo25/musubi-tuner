"""Train the MiniMax Music 3 autoregressive model from distilled RVQ codes."""

from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from musubi_tuner.minimax_music3.ar import AUDIO_END, AUDIO_OFFSET, SEMANTIC_VOCAB, build_prompt, load_depth_decoder
from musubi_tuner.minimax_music3.ar_lora import disable_adapters, inject_lora, load_lora_weights, save_lora


@dataclass
class TrainingItem:
    codes: torch.Tensor
    prompt_ids: torch.Tensor
    caption_dropped_prompt_ids: torch.Tensor
    name: str


class NextLatPredictor(torch.nn.Module):
    def __init__(self, hidden_size: int, block_index: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, eps=1e-6)
        self.up = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.down = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.register_buffer("block_index", torch.tensor(block_index, dtype=torch.int64), persistent=True)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(self.norm(hidden_states))))


def _text(path: Path, default: str) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else default


def _load_items(codes_dir: Path, caption_dir: Path, tokenizer) -> list[TrainingItem]:
    items = []
    for code_path in sorted(codes_dir.glob("*_mm3_rvq.safetensors")):
        stem = code_path.name.removesuffix("_mm3_rvq.safetensors")
        codes = load_file(str(code_path))["rvq_codes"].long()
        if codes.ndim != 2 or codes.shape[1] != 8:
            raise ValueError(f"Expected [frames,8] RVQ codes in {code_path}, got {tuple(codes.shape)}")
        caption = _text(caption_dir / f"{stem}.txt", stem)
        lyrics = _text(caption_dir / f"{stem}.lyrics.txt", "[instrumental]")
        prompt_ids = tokenizer(build_prompt(caption, lyrics), return_tensors="pt").input_ids[0]
        caption_dropped_prompt_ids = tokenizer(build_prompt("", lyrics), return_tensors="pt").input_ids[0]
        items.append(
            TrainingItem(
                codes=codes,
                prompt_ids=prompt_ids,
                caption_dropped_prompt_ids=caption_dropped_prompt_ids,
                name=stem,
            )
        )
    if not items:
        raise ValueError(f"No *_mm3_rvq.safetensors files found in {codes_dir}")
    return items


def _training_sequence(
    item: TrainingItem,
    frames: int,
    prompt_start_probability: float,
    continuation_context_frames: int,
    caption_dropout_probability: float,
    embedding: torch.nn.Module,
    depth_embedding: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    available = item.codes.shape[0]
    if random.random() < prompt_start_probability:
        target_start = 0
    else:
        target_start = random.randrange(1, available) if available > 1 else 0
    count = min(frames, available - target_start)
    # A continuation always needs at least the immediately preceding frame;
    # larger values retain a longer prefix while masking it from the loss.
    retained_context = max(1, continuation_context_frames) if target_start else 0
    context_start = max(0, target_start - retained_context)
    supervised_offset = target_start - context_start
    targets = item.codes[target_start : target_start + count].to(device)
    # Teacher forcing needs every preceding frame, including the unsupervised
    # continuation prefix, but never the final target itself.
    feedback_codes = item.codes[context_start : target_start + count - 1].to(device)
    prompt_ids = (
        item.caption_dropped_prompt_ids
        if random.random() < caption_dropout_probability
        else item.prompt_ids
    ).to(device)
    prompt = embedding(prompt_ids)
    if feedback_codes.shape[0]:
        semantic = embedding(feedback_codes[:, 0] + AUDIO_OFFSET)
        offsets = torch.arange(7, device=device) * 1024
        depth = F.embedding(feedback_codes[:, 1:] + offsets, depth_embedding).sum(dim=1)
        feedback = (semantic + depth) * (8**-0.5)
    else:
        feedback = prompt.new_empty((0, prompt.shape[-1]))
    return torch.cat((prompt, feedback), dim=0).unsqueeze(0), targets, prompt.shape[0], supervised_offset


def _distillation_cross_entropy(student: torch.Tensor, teacher: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Cross entropy against a frozen teacher distribution, optionally truncated to its top-k tokens."""
    teacher = teacher.float()
    student = student.float()
    if top_k > 0 and top_k < teacher.shape[-1]:
        values, indices = teacher.topk(top_k, dim=-1)
        probabilities = values.softmax(dim=-1)
        student_log_probs = student.gather(-1, indices).log_softmax(dim=-1)
    else:
        probabilities = teacher.softmax(dim=-1)
        student_log_probs = student.log_softmax(dim=-1)
    return -(probabilities * student_log_probs).sum(dim=-1).mean()


def _distillation_token_loss(student: torch.Tensor, teacher: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Per-token frozen-teacher loss; student may have an extra candidate dimension."""
    teacher = teacher.float()
    student = student.float()
    if teacher.ndim == student.ndim - 1:
        teacher = teacher.unsqueeze(0)
    if top_k > 0 and top_k < teacher.shape[-1]:
        values, indices = teacher.topk(top_k, dim=-1)
        probabilities = values.softmax(dim=-1)
        indices = indices.expand(*student.shape[:-1], indices.shape[-1])
        probabilities = probabilities.expand_as(indices)
        student_piece = student.gather(-1, indices)
        return -(probabilities * (student_piece - student.logsumexp(dim=-1, keepdim=True))).sum(dim=-1)
    probabilities = teacher.softmax(dim=-1)
    return -(probabilities * student.log_softmax(dim=-1)).sum(dim=-1)


def _blockwise_token_loss(token_loss: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reduce [candidate, token] losses with equal weighting per non-empty block."""
    if block_size <= 0:
        return token_loss.mean(dim=1)
    pad = (-token_loss.shape[1]) % block_size
    valid = torch.ones_like(token_loss, dtype=torch.bool)
    if pad:
        token_loss = F.pad(token_loss, (0, pad))
        valid = F.pad(valid, (0, pad), value=False)
    pieces = token_loss.reshape(token_loss.shape[0], -1, block_size)
    piece_valid = valid.reshape(valid.shape[0], -1, block_size)
    return (pieces * piece_valid).sum(dim=2).div(piece_valid.sum(dim=2).clamp_min(1)).mean(dim=1)


def _xm_select_loss(token_loss: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_losses = _blockwise_token_loss(token_loss, block_size)
    loss, winner = candidate_losses.min(dim=0)
    return loss, winner


def _auxiliary_state(route_embeddings, nextlat_predictor) -> dict[str, torch.Tensor]:
    state = {}
    if route_embeddings is not None:
        state["xm_route_embeddings.weight"] = route_embeddings.weight.detach().cpu().contiguous()
    if nextlat_predictor is not None:
        for key, value in nextlat_predictor.state_dict().items():
            state[f"nextlat_predictor.{key}"] = value.detach().cpu().contiguous()
    return state


@torch.no_grad()
def _load_auxiliary_state(state: dict[str, torch.Tensor], route_embeddings, nextlat_predictor) -> None:
    if route_embeddings is not None:
        route_embeddings.weight.copy_(state["xm_route_embeddings.weight"].to(route_embeddings.weight))
    if nextlat_predictor is not None:
        prefix = "nextlat_predictor."
        nextlat_predictor.load_state_dict(
            {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        )


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step-"))
    except ValueError:
        return -1


def _resolve_resume_checkpoint(output_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    if value != "latest":
        return Path(value)
    candidates = [path for path in output_dir.glob("step-*") if path.is_dir() and (path / "training_state.pt").exists()]
    if not candidates:
        raise ValueError(f"No resumable step-* checkpoints found in {output_dir}")
    return max(candidates, key=_checkpoint_step)


def _save_checkpoint(
    adapters,
    optimizer,
    scheduler,
    output_dir: Path,
    step: int,
    rank: int,
    alpha: int,
    route_embeddings=None,
    nextlat_predictor=None,
) -> Path:
    checkpoint = output_dir / f"step-{step}"
    save_lora(adapters, checkpoint, rank, alpha)
    auxiliary = _auxiliary_state(route_embeddings, nextlat_predictor)
    if auxiliary:
        save_file(auxiliary, checkpoint / "training_auxiliary.safetensors")
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "auxiliary": auxiliary,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint / "training_state.pt")
    return checkpoint


def _prune_checkpoints(output_dir: Path, limit: int | None) -> None:
    if limit is None or limit <= 0:
        return
    checkpoints = sorted(
        (path for path in output_dir.glob("step-*") if path.is_dir()), key=_checkpoint_step
    )
    for path in checkpoints[:-limit]:
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes_dir", type=Path, required=True)
    parser.add_argument("--caption_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ar_model", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=64, help="Teacher-forced frames per optimizer step")
    parser.add_argument("--prompt_start_probability", type=float, default=0.5)
    parser.add_argument("--caption_dropout_probability", type=float, default=0.0)
    parser.add_argument(
        "--continuation_context_frames",
        type=int,
        default=0,
        help="preceding RVQ frames retained as loss-masked continuation context; 0 uses one preceding frame",
    )
    parser.add_argument("--regularization_codes_dir", type=Path)
    parser.add_argument("--regularization_caption_dir", type=Path)
    parser.add_argument("--regularization_probability", type=float, default=0.0)
    parser.add_argument("--regularization_top_k", type=int, default=64)
    parser.add_argument("--xm_enabled", action="store_true")
    parser.add_argument("--xm_candidate_count", type=int, default=2)
    parser.add_argument("--xm_block_size", type=int, default=16)
    parser.add_argument("--nextlat_enabled", action="store_true")
    parser.add_argument("--nextlat_weight", type=float, default=0.1)
    parser.add_argument("--nextlat_block_index", type=int, default=-1)
    parser.add_argument("--nextlat_state_loss", choices=("smooth_l1", "mse"), default="smooth_l1")
    parser.add_argument("--nextlat_kl_weight", type=float, default=0.0)
    parser.add_argument("--depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--resume_from_checkpoint", help="checkpoint directory or 'latest'")
    parser.add_argument("--checkpoints_total_limit", type=int)
    parser.add_argument("--log_every_seconds", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not 0.0 <= args.prompt_start_probability <= 1.0:
        parser.error("--prompt_start_probability must be in [0,1]")
    if not 0.0 <= args.caption_dropout_probability <= 1.0:
        parser.error("--caption_dropout_probability must be in [0,1]")
    if args.continuation_context_frames < 0:
        parser.error("--continuation_context_frames must be non-negative")
    if not 0.0 <= args.regularization_probability <= 1.0:
        parser.error("--regularization_probability must be in [0,1]")
    if args.regularization_probability and not args.regularization_codes_dir:
        parser.error("--regularization_probability requires --regularization_codes_dir")
    if args.xm_enabled and args.xm_candidate_count < 2:
        parser.error("--xm_candidate_count must be at least 2")
    if args.xm_block_size < 0 or args.xm_block_size == 1:
        parser.error("--xm_block_size must be 0 or at least 2")
    if args.nextlat_enabled and args.nextlat_weight <= 0:
        parser.error("--nextlat_weight must be positive")
    if args.nextlat_kl_weight < 0:
        parser.error("--nextlat_kl_weight must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.ar_model, subfolder="tokenizer")
    items = _load_items(args.codes_dir, args.caption_dir, tokenizer)
    regularization_items = []
    if args.regularization_codes_dir:
        regularization_items = _load_items(
            args.regularization_codes_dir,
            args.regularization_caption_dir or args.regularization_codes_dir,
            tokenizer,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.ar_model, subfolder="language_model", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    adapters = inject_lora(model, args.rank, args.alpha)
    model = model.to(device)
    hidden_size = int(model.config.hidden_size)
    route_embeddings = None
    if args.xm_enabled:
        route_embeddings = torch.nn.Embedding(args.xm_candidate_count, hidden_size, device=device, dtype=dtype)
        torch.nn.init.normal_(route_embeddings.weight, mean=0.0, std=float(getattr(model.config, "initializer_range", 0.02)))
        model.add_module("xm_route_embeddings", route_embeddings)
    nextlat_predictor = None
    nextlat_block_index = args.nextlat_block_index
    if args.nextlat_enabled:
        layer_count = len(model.model.layers)
        nextlat_block_index = layer_count - 1 if nextlat_block_index < 0 else nextlat_block_index
        if not 0 <= nextlat_block_index < layer_count:
            parser.error(f"--nextlat_block_index must be within [-1,{layer_count - 1}]")
        nextlat_predictor = NextLatPredictor(hidden_size, nextlat_block_index).to(device=device, dtype=dtype)
        model.add_module("nextlat_predictor", nextlat_predictor)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    backbone = model.model
    embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    depth_decoder = load_depth_decoder(args.ar_model, device=device, dtype=dtype)
    depth_decoder.requires_grad_(False)
    depth_embedding = depth_decoder.audio_embeddings.weight
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    resume_path = _resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_path is not None:
        load_lora_weights(adapters, resume_path)
        state = torch.load(resume_path / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("auxiliary"):
            _load_auxiliary_state(state["auxiliary"], route_embeddings, nextlat_predictor)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        start_step = int(state["step"])
        print(f"resumed={resume_path} step={start_step}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"items={len(items)} trainable_parameters={trainable:,} dtype={dtype}")

    started = last_log = time.monotonic()
    interval_steps = 0
    interval_loss = interval_semantic_loss = interval_depth_loss = 0.0
    interval_nextlat_loss = 0.0
    interval_route_usage = [0] * (args.xm_candidate_count if args.xm_enabled else 1)
    interval_semantic_accuracy = interval_depth_accuracy = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step + 1, args.steps + 1):
        regularizing = bool(regularization_items) and random.random() < args.regularization_probability
        item = random.choice(regularization_items if regularizing else items)
        inputs, targets, prompt_length, supervised_offset = _training_sequence(
            item,
            args.frames,
            args.prompt_start_probability,
            args.continuation_context_frames,
            args.caption_dropout_probability,
            embedding,
            depth_embedding,
            device,
        )
        teacher_hidden = None
        if regularizing:
            with torch.no_grad(), disable_adapters(adapters):
                teacher_output = backbone(inputs_embeds=inputs, use_cache=False, return_dict=True)
                teacher_hidden = teacher_output.last_hidden_state[
                    :, prompt_length - 1 + supervised_offset : prompt_length - 1 + supervised_offset + targets.shape[0]
                ].detach()
        candidate_count = args.xm_candidate_count if args.xm_enabled else 1
        model_inputs = inputs.repeat(candidate_count, 1, 1)
        if route_embeddings is not None:
            supervised_start = prompt_length - 1 + supervised_offset
            supervised_end = supervised_start + targets.shape[0]
            route_ids = torch.arange(candidate_count, device=device)
            routes = route_embeddings(route_ids).to(model_inputs.dtype)
            model_inputs[:, supervised_start:supervised_end] += routes[:, None, :]
        outputs = backbone(
            inputs_embeds=model_inputs,
            use_cache=False,
            return_dict=True,
            output_hidden_states=args.nextlat_enabled,
        )
        hidden = outputs.last_hidden_state[
            :, prompt_length - 1 + supervised_offset : prompt_length - 1 + supervised_offset + targets.shape[0]
        ]
        audio_weight = torch.cat(
            (lm_head.weight[AUDIO_END : AUDIO_END + 1], lm_head.weight[AUDIO_OFFSET : AUDIO_OFFSET + SEMANTIC_VOCAB])
        ).float()
        semantic_logits = F.linear(hidden.float(), audio_weight)
        semantic_targets = targets[:, 0] + 1
        if regularizing:
            teacher_semantic_logits = F.linear(teacher_hidden.float(), audio_weight)
            semantic_token_loss = _distillation_token_loss(
                semantic_logits, teacher_semantic_logits, args.regularization_top_k
            )
        else:
            semantic_token_loss = F.cross_entropy(
                semantic_logits.reshape(-1, semantic_logits.shape[-1]),
                semantic_targets.repeat(candidate_count),
                reduction="none",
            ).reshape(candidate_count, -1)

        ar_hidden = hidden.reshape(candidate_count * targets.shape[0], -1)
        semantic_embedding = embedding(targets[:, 0] + AUDIO_OFFSET)
        offsets = torch.arange(6, device=device) * 1024
        teacher_depth = depth_decoder.audio_embeddings(targets[:, 1:7] + offsets)
        semantic_embedding_expanded = semantic_embedding.repeat(candidate_count, 1)
        teacher_depth_expanded = teacher_depth.repeat(candidate_count, 1, 1)
        depth_inputs = torch.cat(
            (
                depth_decoder.projection(ar_hidden).unsqueeze(1),
                depth_decoder.projection(semantic_embedding_expanded).unsqueeze(1),
                depth_decoder.projection(teacher_depth_expanded),
            ),
            dim=1,
        )
        depth_hidden = depth_decoder(depth_inputs)[:, 1:].reshape(candidate_count, targets.shape[0], 7, -1)
        teacher_depth_hidden = None
        if regularizing:
            teacher_depth_inputs = torch.cat(
                (
                    depth_decoder.projection(teacher_hidden.squeeze(0)).unsqueeze(1),
                    depth_decoder.projection(semantic_embedding).unsqueeze(1),
                    depth_decoder.projection(teacher_depth),
                ),
                dim=1,
            )
            with torch.no_grad():
                teacher_depth_hidden = depth_decoder(teacher_depth_inputs)[:, 1:]
        depth_token_losses = []
        depth_correct = 0.0
        for index, head in enumerate(depth_decoder.audio_heads):
            depth_logits = head(depth_hidden[:, :, index]).float()
            depth_target = targets[:, index + 1]
            if regularizing:
                teacher_depth_logits = head(teacher_depth_hidden[:, index]).float()
                depth_token_losses.append(
                    _distillation_token_loss(depth_logits, teacher_depth_logits, args.regularization_top_k)
                )
            else:
                depth_token_losses.append(
                    F.cross_entropy(
                        depth_logits.reshape(-1, depth_logits.shape[-1]),
                        depth_target.repeat(candidate_count),
                        reduction="none",
                    ).reshape(candidate_count, -1)
                )
        depth_token_loss = torch.stack(depth_token_losses).mean(dim=0)
        combined_token_loss = semantic_token_loss + args.depth_loss_weight * depth_token_loss
        if args.xm_enabled:
            loss, winner = _xm_select_loss(combined_token_loss, args.xm_block_size)
            winner_index = int(winner.item())
        else:
            loss = combined_token_loss.mean()
            winner_index = 0
        semantic_loss = semantic_token_loss[winner_index].mean()
        depth_loss = depth_token_loss[winner_index].mean()
        selected_semantic_logits = semantic_logits[winner_index]
        for index, head in enumerate(depth_decoder.audio_heads):
            selected_depth_logits = head(depth_hidden[winner_index, :, index]).float()
            depth_correct += (selected_depth_logits.argmax(dim=-1) == targets[:, index + 1]).float().mean().item()

        nextlat_loss = loss.new_zeros(())
        if nextlat_predictor is not None:
            layer_hidden = outputs.hidden_states[nextlat_block_index + 1][
                winner_index,
                prompt_length - 1 + supervised_offset : prompt_length - 1 + supervised_offset + targets.shape[0],
            ]
            if layer_hidden.shape[0] < 2:
                raise ValueError("NextLat requires at least two supervised audio frames")
            nextlat_prediction = nextlat_predictor(layer_hidden[:-1])
            nextlat_target = layer_hidden[1:].detach()
            if args.nextlat_state_loss == "smooth_l1":
                nextlat_state_loss = F.smooth_l1_loss(nextlat_prediction.float(), nextlat_target.float())
            else:
                nextlat_state_loss = F.mse_loss(nextlat_prediction.float(), nextlat_target.float())
            nextlat_loss = nextlat_state_loss * args.nextlat_weight
            if args.nextlat_kl_weight > 0:
                pred_logits = F.linear(nextlat_prediction.float(), audio_weight)
                with torch.no_grad():
                    target_probs = F.linear(nextlat_target.float(), audio_weight).softmax(dim=-1)
                nextlat_kl = F.kl_div(pred_logits.log_softmax(dim=-1), target_probs, reduction="batchmean")
                nextlat_loss = nextlat_loss + args.nextlat_kl_weight * nextlat_kl
            loss = loss + nextlat_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        interval_steps += 1
        interval_loss += loss.item()
        interval_semantic_loss += semantic_loss.item()
        interval_depth_loss += depth_loss.item()
        interval_nextlat_loss += nextlat_loss.item()
        interval_route_usage[winner_index] += 1
        interval_semantic_accuracy += (selected_semantic_logits.argmax(dim=-1) == semantic_targets).float().mean().item()
        interval_depth_accuracy += depth_correct / 7

        now = time.monotonic()
        if now - last_log >= args.log_every_seconds or step == args.steps:
            elapsed = now - last_log
            peak_allocated = peak_reserved = 0.0
            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
                peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            print(
                f"step={step}/{args.steps} loss={interval_loss / interval_steps:.4f} "
                f"semantic_loss={interval_semantic_loss / interval_steps:.4f} "
                f"depth_loss={interval_depth_loss / interval_steps:.4f} "
                f"nextlat_loss={interval_nextlat_loss / interval_steps:.4f} "
                f"semantic_accuracy={interval_semantic_accuracy / interval_steps:.4f} "
                f"depth_accuracy={interval_depth_accuracy / interval_steps:.4f} "
                f"steps_per_second={interval_steps / elapsed:.3f} "
                f"peak_allocated_gib={peak_allocated:.2f} peak_reserved_gib={peak_reserved:.2f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} regularization={int(regularizing)} "
                f"xm_routes={interval_route_usage} item={item.name}",
                flush=True,
            )
            interval_steps = 0
            interval_loss = interval_semantic_loss = interval_depth_loss = 0.0
            interval_nextlat_loss = 0.0
            interval_route_usage = [0] * len(interval_route_usage)
            interval_semantic_accuracy = interval_depth_accuracy = 0.0
            last_log = now
        if args.save_every > 0 and step % args.save_every == 0:
            _save_checkpoint(
                adapters,
                optimizer,
                scheduler,
                args.output_dir,
                step,
                args.rank,
                args.alpha,
                route_embeddings,
                nextlat_predictor,
            )
            _prune_checkpoints(args.output_dir, args.checkpoints_total_limit)

    save_lora(adapters, args.output_dir, args.rank, args.alpha)
    auxiliary = _auxiliary_state(route_embeddings, nextlat_predictor)
    if auxiliary:
        save_file(auxiliary, args.output_dir / "training_auxiliary.safetensors")
    print(f"saved={args.output_dir} elapsed_seconds={time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
