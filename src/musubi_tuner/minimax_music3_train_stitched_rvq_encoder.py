"""Train a MiniMax Music 3 RVQ encoder on exact stitched generation traces."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset

from musubi_tuner.minimax_music3.stitched_rvq_encoder import (
    StitchedRVQEncoder,
    StitchedRVQEncoderConfig,
    frame_pool,
    stitched_frame_boundaries,
)


@lru_cache(maxsize=16)
def _trace(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


class TraceWindowDataset(Dataset):
    def __init__(self, paths: list[Path], frames: int = 128, padded_latents: int = 448):
        self.items = []
        self.frames = frames
        self.padded_latents = padded_latents
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                code_count = stream.get_slice("codes").get_shape()[0]
                latent_count = stream.get_slice("flow_vae_latents").get_shape()[-1]
            frame_count = code_count - 1
            boundaries = stitched_frame_boundaries(frame_count)
            while frame_count and int(boundaries[frame_count]) > latent_count:
                frame_count -= 1
            for start in range(0, frame_count - frames + 1, frames):
                window_boundaries = boundaries[start : start + frames + 1]
                self.items.append(
                    (str(path), start, window_boundaries, frame_pool(window_boundaries, padded_latents))
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, start_frame, boundaries, pool = self.items[index]
        start_latent, end_latent = int(boundaries[0]), int(boundaries[-1])
        length = end_latent - start_latent
        tensors = _trace(path)
        latents = tensors["flow_vae_latents"][:, start_latent:end_latent].float()
        latents = F.pad(latents, (0, self.padded_latents - length))
        targets = tensors["codes"][start_frame + 1 : start_frame + 1 + self.frames].long()
        return latents, pool, targets


def _losses(output, targets):
    semantic, depth = output
    losses = [F.cross_entropy(semantic.flatten(0, 1), targets[..., 0].flatten())]
    losses.extend(
        F.cross_entropy(logits.flatten(0, 1), targets[..., index + 1].flatten())
        for index, logits in enumerate(depth)
    )
    return losses


@torch.inference_mode()
def _evaluate(model, loader, device, dtype):
    model.eval()
    loss_sum = semantic_correct = depth_correct = frames = 0.0
    for latents, pool, targets in loader:
        latents, pool, targets = latents.to(device), pool.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            semantic, depth = model(latents, pool)
            losses = _losses((semantic, depth), targets)
        count = targets.shape[0] * targets.shape[1]
        loss_sum += sum(loss.item() for loss in losses) * count / len(losses)
        semantic_correct += (semantic.argmax(-1) == targets[..., 0]).sum().item()
        depth_correct += sum(
            (logits.argmax(-1) == targets[..., index + 1]).sum().item()
            for index, logits in enumerate(depth)
        )
        frames += count
    model.train()
    return {
        "loss": loss_sum / frames,
        "semantic_accuracy": semantic_correct / frames,
        "depth_accuracy": depth_correct / (7 * frames),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_seconds", type=float, default=10.0)
    args = parser.parse_args()

    paths = sorted(args.trace_dir.rglob("prediction.safetensors"))
    if len(paths) < 2:
        parser.error("At least two trace files are required")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    holdout_count = max(1, min(len(paths) - 1, round(len(paths) * args.holdout)))
    eval_paths, train_paths = paths[:holdout_count], paths[holdout_count:]
    config = StitchedRVQEncoderConfig(width=args.width, layers=args.layers, heads=args.heads)
    train_set = TraceWindowDataset(train_paths, config.window_frames, config.padded_latents)
    eval_set = TraceWindowDataset(eval_paths, config.window_frames, config.padded_latents)
    if not train_set or not eval_set:
        parser.error("Trace split does not contain complete encoder windows")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    model = StitchedRVQEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.steps

    def schedule(step):
        warmup = min(1.0, (step + 1) / max(1, args.warmup_steps))
        progress = min(1.0, step / max(1, total_steps))
        return warmup * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    iterator = iter(train_loader)
    started = last_log = time.monotonic()
    peak_memory = 0
    for step in range(1, args.steps + 1):
        try:
            latents, pool, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            latents, pool, targets = next(iterator)
        latents, pool, targets = latents.to(device), pool.to(device), targets.to(device)
        model.train()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            losses = _losses(model(latents, pool), targets)
            loss = sum(losses) / len(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        now = time.monotonic()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        if step == 1 or now - last_log >= args.log_every_seconds:
            elapsed = max(now - started, 1e-6)
            print(
                f"step={step}/{args.steps} loss={loss.item():.4f} semantic_loss={losses[0].item():.4f} "
                f"depth_loss={sum(item.item() for item in losses[1:]) / 7:.4f} "
                f"steps_per_second={step / elapsed:.3f} peak_allocated_gib={peak_memory / 2**30:.2f}",
                flush=True,
            )
            last_log = now
        if step % args.eval_every == 0 or step == args.steps:
            metrics = _evaluate(model, eval_loader, device, dtype)
            print("eval " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()), flush=True)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                model.save(
                    args.output.with_name(f"{args.output.stem}-best{args.output.suffix}"),
                    {"step": str(step), "train_traces": str(len(train_paths)), "eval_traces": str(len(eval_paths))},
                )
        if step % args.save_every == 0 or step == args.steps:
            model.save(
                args.output.with_name(f"{args.output.stem}-step{step:08d}{args.output.suffix}"),
                {"step": str(step), "train_traces": str(len(train_paths)), "eval_traces": str(len(eval_paths))},
            )
    model.save(
        args.output,
        {"step": str(args.steps), "train_traces": str(len(train_paths)), "eval_traces": str(len(eval_paths))},
    )


if __name__ == "__main__":
    main()
