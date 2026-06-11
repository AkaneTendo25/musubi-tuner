"""Step-level trainer logging helpers."""

from __future__ import annotations

import argparse


def generate_step_logs(
    self,
    args: argparse.Namespace,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    video_loss=None,
    audio_loss=None,
    mask_metrics=None,
):
    network_train_unet_only = True
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if video_loss is not None:
        logs["loss/video"] = video_loss
    if audio_loss is not None:
        logs["loss/audio"] = audio_loss

    if mask_metrics:
        for k, v in mask_metrics.items():
            logs[f"loss/{k}"] = v

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/average_key_norm"] = mean_norm
        logs["max_norm/max_key_norm"] = maximum_norm

    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None and i < len(lr_descriptions):
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if network_train_unet_only else 1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                if len(lrs) > 2:
                    lr_desc = f"group{i}"
                else:
                    lr_desc = "unet"

        logs[f"lr/{lr_desc}"] = lr

        if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().endswith("Prodigy".lower()):
            logs[f"lr/d*lr/{lr_desc}"] = (
                lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
            )

        if args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
            logs[f"lr/d*lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
            if "effective_lr" in optimizer.param_groups[i]:
                logs[f"lr/d*eff_lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["effective_lr"]

        if args.optimizer_type.lower() == "automagic" and optimizer is not None:
            logs["lr/automagic_avg"] = optimizer.get_avg_learning_rate()
            lr_tensor = optimizer.get_lr_tensor()
            if lr_tensor is not None and len(lr_tensor) > 1:
                logs["lr/automagic_min"] = float(lr_tensor.min())
                logs["lr/automagic_max"] = float(lr_tensor.max())
                logs["lr/automagic_std"] = float(lr_tensor.std())

    return logs
