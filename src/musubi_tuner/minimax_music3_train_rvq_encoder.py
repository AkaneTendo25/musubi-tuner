"""Train the MiniMax Music 3 distilled DAV-latent to RVQ encoder."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from musubi_tuner.minimax_music3.rvq_distill import load_distill_cache
from musubi_tuner.minimax_music3.rvq_encoder import MiniMaxMusic3RVQEncoder, RVQEncoderConfig


@lru_cache(maxsize=8)
def _cached_item(path: str) -> dict[str, torch.Tensor]:
    return load_distill_cache(path)[0]


@lru_cache(maxsize=16)
def _cached_features(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


class DistillDataset(Dataset):
    def __init__(self, paths: list[Path], feature_dir: Path | None = None, input_key: str = "dav_latents"):
        self.items = []
        self.feature_dir = feature_dir
        self.input_key = input_key
        for path in paths:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as stream:
                variants = stream.get_slice("dav_latents").get_shape()[0]
            for variant in range(variants):
                self.items.append((str(path), variant))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, variant = self.items[index]
        tensors = _cached_item(path)
        if self.input_key == "dav_latents":
            inputs = tensors[self.input_key][variant]
        else:
            if self.feature_dir is None:
                raise ValueError("feature_dir is required for cached acoustic features")
            cached_features = _cached_features(str(self.feature_dir / Path(path).name))
            mel_features = cached_features["mel_features"][variant]
            if self.input_key == "mel_features":
                inputs = mel_features
            else:
                dav_latents = tensors["dav_latents"][variant]
                acoustic = mel_features
                if self.input_key == "dav_ssl_features":
                    acoustic = cached_features["ssl_features"][variant]
                elif self.input_key == "dav_mel_ssl_features":
                    ssl_features = cached_features["ssl_features"][variant]
                    ssl_features = F.interpolate(
                        ssl_features.float().unsqueeze(0),
                        size=mel_features.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )[0].to(mel_features)
                    acoustic = torch.cat((mel_features, ssl_features), dim=0)
                acoustic = F.interpolate(
                    acoustic.float().unsqueeze(0),
                    size=dav_latents.shape[-1],
                    mode="linear",
                    align_corners=False,
                )[0].to(dav_latents)
                inputs = torch.cat((dav_latents, acoustic), dim=0)
        return inputs, tensors["rvq_codes"].long(), None, None


class TraceDistillDataset(Dataset):
    """Read exact generation latents and post-guidance RVQ distributions."""

    def __init__(self, paths: list[Path], frames: int):
        from safetensors import safe_open

        self.items = []
        self.frames = frames
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                code_frames = stream.get_slice("codes").get_shape()[0] - 1
            for start in range(0, code_frames, frames):
                self.items.append((str(path), start, min(code_frames, start + frames)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, start, end = self.items[index]
        tensors = _cached_features(path)
        codes = tensors["codes"][1:].long()
        topk_ids = tensors["teacher_topk_ids"][1:].long()
        topk_logits = tensors["teacher_topk_logits"][1:].float()
        latents = tensors["flow_vae_latents"].float()
        total_frames = codes.shape[0]
        latent_start = start * latents.shape[-1] // total_frames
        latent_end = (end * latents.shape[-1] + total_frames - 1) // total_frames
        return (
            latents[:, latent_start:latent_end],
            codes[start:end],
            topk_ids[start:end],
            topk_logits[start:end],
        )


def _collate(batch):
    max_latents = max(item[0].shape[-1] for item in batch)
    max_frames = max(item[1].shape[0] for item in batch)
    latent_batch, code_batch, id_batch, logit_batch = [], [], [], []
    has_teacher = batch[0][2] is not None
    for latents, codes, topk_ids, topk_logits in batch:
        latent_batch.append(F.pad(latents, (0, max_latents - latents.shape[-1])))
        code_batch.append(F.pad(codes, (0, 0, 0, max_frames - codes.shape[0]), value=-100))
        if has_teacher:
            id_batch.append(F.pad(topk_ids, (0, 0, 0, 0, 0, max_frames - codes.shape[0]), value=-1))
            logit_batch.append(
                F.pad(topk_logits, (0, 0, 0, 0, 0, max_frames - codes.shape[0]), value=float("-inf"))
            )
    return (
        torch.stack(latent_batch),
        torch.stack(code_batch),
        torch.stack(id_batch) if has_teacher else None,
        torch.stack(logit_batch) if has_teacher else None,
    )


def _soft_topk_loss(student, teacher_ids, teacher_logits):
    valid = (teacher_ids >= 0) & (teacher_ids < student.shape[1])
    safe_ids = teacher_ids.clamp(0, student.shape[1] - 1)
    teacher_logits = teacher_logits.masked_fill(~valid, float("-inf"))
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1).masked_fill(~valid, 0)
    student_log_probs = F.log_softmax(student.float(), dim=1)
    selected = student_log_probs.permute(0, 2, 1).gather(-1, safe_ids)
    frame_loss = -(teacher_probs * selected).sum(-1)
    frame_mask = valid.any(-1)
    return frame_loss[frame_mask].mean()


def _losses(model, semantic_features, depth_features, codes, teacher_ids=None, teacher_logits=None, soft_weight=0.0):
    semantic, depth = model.classify(semantic_features, depth_features)
    semantic_loss = F.cross_entropy(semantic.transpose(1, 2).reshape(-1, semantic.shape[1]), codes[..., 0].reshape(-1))
    depth_losses = [
        F.cross_entropy(logits.transpose(1, 2).reshape(-1, logits.shape[1]), codes[..., index + 1].reshape(-1))
        for index, logits in enumerate(depth)
    ]
    mask = codes[..., 0] != -100
    semantic_targets = model.semantic_codebook[codes[..., 0].clamp_min(0)]
    semantic_cosine = (
        1 - F.cosine_similarity(semantic_features.transpose(1, 2).float(), semantic_targets.float(), dim=-1)
    )[mask].mean()
    depth_cosines = []
    for index, features in enumerate(depth_features):
        target_codes = codes[..., index + 1]
        target_mask = target_codes != -100
        targets = model.depth_codebooks[index, target_codes.clamp_min(0)]
        depth_cosines.append(
            (1 - F.cosine_similarity(features.transpose(1, 2).float(), targets.float(), dim=-1))[target_mask].mean()
        )
    semantic_loss = semantic_loss + semantic_cosine
    depth_loss = torch.stack(depth_losses).mean() + torch.stack(depth_cosines).mean()
    if teacher_ids is not None and soft_weight:
        semantic_soft = _soft_topk_loss(semantic, teacher_ids[..., 0, :], teacher_logits[..., 0, :])
        depth_soft = torch.stack(
            [
                _soft_topk_loss(logits, teacher_ids[..., index + 1, :], teacher_logits[..., index + 1, :])
                for index, logits in enumerate(depth)
            ]
        ).mean()
        semantic_loss = (1 - soft_weight) * semantic_loss + soft_weight * semantic_soft
        depth_loss = (1 - soft_weight) * depth_loss + soft_weight * depth_soft
    return semantic_loss, depth_loss


@torch.inference_mode()
def _evaluate(model, loader, device, amp_dtype, max_batches, soft_weight=0.0):
    model.eval()
    totals = {
        "semantic_loss": 0.0,
        "depth_loss": 0.0,
        "semantic_top1": 0,
        "semantic_top10": 0,
        "semantic_top50": 0,
        "depth_top1": 0,
        "depth_top10": 0,
        "depth_teacher_forced_top1": 0,
        "semantic_cosine": 0.0,
        "depth_cosine": 0.0,
        "frames": 0,
    }
    semantic_ranks = []
    depth_ranks = []
    depth_stream_ranks = [[] for _ in range(7)]
    depth_stream_correct = torch.zeros(7, dtype=torch.long)
    depth_teacher_forced_ranks = []
    for batch_index, (latents, codes, teacher_ids, teacher_logits) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        latents, codes = latents.to(device), codes.to(device)
        if teacher_ids is not None:
            teacher_ids, teacher_logits = teacher_ids.to(device), teacher_logits.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            semantic_features, depth_features = model.features(latents, codes.shape[1])
            semantic, depth = model.classify(semantic_features, depth_features)
            _, teacher_forced_features = model.features(latents, codes.shape[1], codes)
            _, teacher_forced_depth = model.classify(semantic_features, teacher_forced_features)
            semantic_loss, depth_loss = _losses(
                model, semantic_features, depth_features, codes, teacher_ids, teacher_logits, soft_weight
            )
        mask = codes[..., 0] != -100
        count = mask.sum().item()
        totals["semantic_loss"] += semantic_loss.item() * count
        totals["depth_loss"] += depth_loss.item() * count
        semantic_targets = codes[..., 0].clamp_min(0)
        semantic_target_logits = semantic.gather(1, semantic_targets.unsqueeze(1)).squeeze(1)
        semantic_ranks.append(((semantic > semantic_target_logits.unsqueeze(1)).sum(1) + 1)[mask].cpu())
        totals["semantic_top1"] += ((semantic.argmax(1) == semantic_targets) & mask).sum().item()
        for k, key in ((10, "semantic_top10"), (50, "semantic_top50")):
            matches = semantic.topk(k, dim=1).indices == semantic_targets.unsqueeze(1)
            totals[key] += (matches.any(1) & mask).sum().item()
        target_features = model.semantic_codebook[semantic_targets]
        cosine = F.cosine_similarity(semantic_features.transpose(1, 2).float(), target_features.float(), dim=-1)
        totals["semantic_cosine"] += (cosine * mask).sum().item()
        for index, logits in enumerate(depth):
            depth_mask = codes[..., index + 1] != -100
            depth_targets = codes[..., index + 1].clamp_min(0)
            depth_target_logits = logits.gather(1, depth_targets.unsqueeze(1)).squeeze(1)
            ranks = ((logits > depth_target_logits.unsqueeze(1)).sum(1) + 1)[depth_mask].cpu()
            depth_ranks.append(ranks)
            depth_stream_ranks[index].append(ranks)
            correct = ((logits.argmax(1) == depth_targets) & depth_mask).sum().item()
            totals["depth_top1"] += correct
            depth_stream_correct[index] += correct
            matches = logits.topk(10, dim=1).indices == depth_targets.unsqueeze(1)
            totals["depth_top10"] += (matches.any(1) & depth_mask).sum().item()
            target_features = model.depth_codebooks[index, depth_targets]
            cosine = F.cosine_similarity(depth_features[index].transpose(1, 2).float(), target_features.float(), dim=-1)
            totals["depth_cosine"] += (cosine * depth_mask).sum().item()
            teacher_logits_for_stream = teacher_forced_depth[index]
            teacher_target_logits = teacher_logits_for_stream.gather(1, depth_targets.unsqueeze(1)).squeeze(1)
            depth_teacher_forced_ranks.append(
                ((teacher_logits_for_stream > teacher_target_logits.unsqueeze(1)).sum(1) + 1)[depth_mask].cpu()
            )
            totals["depth_teacher_forced_top1"] += (
                (teacher_logits_for_stream.argmax(1) == depth_targets) & depth_mask
            ).sum().item()
        totals["frames"] += count
    count = max(1, totals["frames"])
    metrics = {
        "semantic_loss": totals["semantic_loss"] / count,
        "depth_loss": totals["depth_loss"] / count,
        "semantic_accuracy": totals["semantic_top1"] / count,
        "depth_accuracy": totals["depth_top1"] / (7 * count),
        "semantic_top10": totals["semantic_top10"] / count,
        "semantic_top50": totals["semantic_top50"] / count,
        "depth_top10": totals["depth_top10"] / (7 * count),
        "depth_teacher_forced_accuracy": totals["depth_teacher_forced_top1"] / (7 * count),
        "semantic_cosine": totals["semantic_cosine"] / count,
        "depth_cosine": totals["depth_cosine"] / (7 * count),
        "semantic_median_rank": torch.cat(semantic_ranks).float().median().item(),
        "depth_median_rank": torch.cat(depth_ranks).float().median().item(),
        "depth_teacher_forced_median_rank": torch.cat(depth_teacher_forced_ranks).float().median().item(),
    }
    for index in range(7):
        metrics[f"depth_{index + 1}_accuracy"] = depth_stream_correct[index].item() / count
        metrics[f"depth_{index + 1}_median_rank"] = torch.cat(depth_stream_ranks[index]).float().median().item()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache_dir", type=Path)
    source.add_argument("--trace_dir", type=Path, help="Directory containing extracted prediction.safetensors traces")
    parser.add_argument("--feature_dir", type=Path)
    parser.add_argument(
        "--input_key",
        choices=("dav_latents", "mel_features", "dav_mel_features", "dav_ssl_features", "dav_mel_ssl_features"),
        default="dav_latents",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--trace_frames", type=int, default=256, help="Frames per exact-latent training segment")
    parser.add_argument("--soft_target_weight", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--code_dim", type=int, default=256)
    parser.add_argument("--classifier", choices=("direct", "cosine"), default="cosine")
    parser.add_argument("--dual_trunk", action="store_true", help="Use a separate acoustic trunk for DAV+mel inputs")
    parser.add_argument("--depth_loss_weight", type=float, default=2.0)
    parser.add_argument("--codebooks", type=Path)
    parser.add_argument("--init_checkpoint", type=Path)
    parser.add_argument("--semantic_checkpoint", type=Path, help="Initialize the DAV semantic trunk of a dual-trunk model")
    parser.add_argument("--freeze_semantic", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--teacher_forcing_probability", type=float, default=0.75)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.teacher_forcing_probability <= 1.0:
        parser.error("teacher_forcing_probability must be between 0 and 1")
    if args.dual_trunk and not args.input_key.startswith("dav_"):
        parser.error("dual_trunk requires a DAV plus acoustic input")
    if (args.semantic_checkpoint is not None or args.freeze_semantic) and not args.dual_trunk:
        parser.error("semantic checkpoint options require --dual_trunk")

    if not 0.0 <= args.soft_target_weight <= 1.0:
        parser.error("soft_target_weight must be between 0 and 1")
    paths = sorted(
        args.cache_dir.glob("*_mm3_rvq_distill.safetensors")
        if args.cache_dir is not None
        else args.trace_dir.rglob("prediction.safetensors")
    )
    if len(paths) < 2:
        parser.error("At least two sequence caches are required")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    holdout_count = max(1, min(len(paths) - 1, round(len(paths) * args.holdout)))
    eval_paths, train_paths = paths[:holdout_count], paths[holdout_count:]
    if args.trace_dir is not None:
        train_dataset = TraceDistillDataset(train_paths, args.trace_frames)
        eval_dataset = TraceDistillDataset(eval_paths, args.trace_frames)
    else:
        train_dataset = DistillDataset(train_paths, args.feature_dir, args.input_key)
        eval_dataset = DistillDataset(eval_paths, args.feature_dir, args.input_key)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    codebook_path = args.codebooks or (args.cache_dir / "rvq_codebooks.safetensors" if args.cache_dir else None)
    if codebook_path is None:
        parser.error("--codebooks is required with --trace_dir")
    if not codebook_path.exists():
        parser.error(f"Projected teacher codebooks not found: {codebook_path}")
    from safetensors.torch import load_file

    if args.init_checkpoint is not None:
        model = MiniMaxMusic3RVQEncoder.load(args.init_checkpoint, device)
    else:
        codebook_tensors = load_file(str(codebook_path))
        model = MiniMaxMusic3RVQEncoder(
            RVQEncoderConfig(
                input_channels=train_dataset[0][0].shape[0],
                acoustic_channels=train_dataset[0][0].shape[0] - 128 if args.dual_trunk else 0,
                width=args.width,
                layers=args.layers,
                code_dim=args.code_dim,
                classifier=args.classifier,
            )
        ).to(device)
        model.set_codebooks(codebook_tensors["semantic_codebook"], codebook_tensors["depth_codebooks"])
    if args.semantic_checkpoint is not None:
        checkpoint = load_file(str(args.semantic_checkpoint))
        current = model.state_dict()
        semantic_prefixes = ("input.", "blocks.", "output_norm.", "semantic_head.", "semantic_codebook")
        semantic_state = {
            key: value for key, value in checkpoint.items() if key.startswith(semantic_prefixes) and key in current
        }
        model.load_state_dict(semantic_state, strict=False)
    if args.freeze_semantic:
        semantic_prefixes = ("input.", "blocks.", "output_norm.", "semantic_head.")
        for name, parameter in model.named_parameters():
            if name.startswith(semantic_prefixes):
                parameter.requires_grad_(False)
    if args.eval_only:
        metrics = _evaluate(model, eval_loader, device, amp_dtype, args.eval_batches, args.soft_target_weight)
        print("eval " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    generator = iter(train_loader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    stale_evaluations = 0

    final_step = 0
    for step in range(1, args.steps + 1):
        final_step = step
        try:
            latents, codes, teacher_ids, teacher_logits = next(generator)
        except StopIteration:
            generator = iter(train_loader)
            latents, codes, teacher_ids, teacher_logits = next(generator)
        latents, codes = latents.to(device), codes.to(device)
        if teacher_ids is not None:
            teacher_ids, teacher_logits = teacher_ids.to(device), teacher_logits.to(device)
        model.train()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            target_codes = codes if rng.random() < args.teacher_forcing_probability else None
            semantic_features, depth_features = model.features(latents, codes.shape[1], target_codes)
            semantic_loss, depth_loss = _losses(
                model,
                semantic_features,
                depth_features,
                codes,
                teacher_ids,
                teacher_logits,
                args.soft_target_weight,
            )
            loss = semantic_loss + args.depth_loss_weight * depth_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 25 == 0:
            print(f"step={step} loss={loss.item():.6f} semantic={semantic_loss.item():.6f} depth={depth_loss.item():.6f}")
        if step % args.eval_every == 0 or step == args.steps:
            metrics = _evaluate(model, eval_loader, device, amp_dtype, args.eval_batches, args.soft_target_weight)
            print("eval " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
            score = metrics["semantic_accuracy"] + args.depth_loss_weight * metrics["depth_accuracy"]
            if score > best_score:
                best_score = score
                stale_evaluations = 0
                model.save(
                    args.output.with_name(f"{args.output.stem}-best{args.output.suffix}"),
                    {"step": str(step), "score": str(score), "train_sequences": str(len(train_paths)), "eval_sequences": str(len(eval_paths))},
                )
            else:
                stale_evaluations += 1
                if args.early_stopping_patience and stale_evaluations >= args.early_stopping_patience:
                    print(f"early_stop step={step} best_score={best_score:.6f}")
                    break
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.output.with_name(f"{args.output.stem}-step{step:08d}{args.output.suffix}")
            model.save(checkpoint, {"step": str(step), "train_sequences": str(len(train_paths)), "eval_sequences": str(len(eval_paths))})
    model.save(args.output, {"step": str(final_step), "train_sequences": str(len(train_paths)), "eval_sequences": str(len(eval_paths))})


if __name__ == "__main__":
    main()
