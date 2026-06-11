"""Architecture-neutral sample generation runtime for NetworkTrainer."""

from __future__ import annotations

import logging
import os
import time

import torch
from accelerate import Accelerator, PartialState

from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.sampling_prompts import should_sample_images

logger = logging.getLogger(__name__)


def sample_images(self, accelerator: Accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
    """Generate configured training samples using architecture-specific inference hooks."""
    if not should_sample_images(args, steps, epoch):
        return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if sample_parameters is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # Use the unwrapped model
    transformer = accelerator.unwrap_model(transformer)
    transformer.switch_block_swap_for_inference()

    # Create a directory to save the samples
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad(), accelerator.autocast():
            for sample_parameter in sample_parameters:
                self.sample_image_inference(
                    accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                )
                clean_memory_on_device(accelerator.device)
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_params = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                for sample_parameter in sample_parameter_lists[0]:
                    self.sample_image_inference(
                        accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                    )
                    clean_memory_on_device(accelerator.device)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    transformer.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
    """Run one architecture-specific sample inference and save the resulting image/video."""
    sample_steps = sample_parameter.get("sample_steps", 20)
    width = sample_parameter.get("width", 256)  # make smaller for faster and memory saving inference
    height = sample_parameter.get("height", 256)
    frame_count = sample_parameter.get("frame_count", 1)
    guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
    discrete_flow_shift = sample_parameter.get("discrete_flow_shift", self.default_discrete_flow_shift)
    seed = sample_parameter.get("seed")
    prompt: str = sample_parameter.get("prompt", "")
    cfg_scale = sample_parameter.get("cfg_scale", None)  # None for architecture default
    negative_prompt = sample_parameter.get("negative_prompt", None)

    # round width and height to multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # 1, 5, 9, 13, ... For HunyuanVideo and Wan2.1
    frame_count = (frame_count - 1) // self.vae_frame_stride * self.vae_frame_stride + 1

    if self.i2v_training:
        image_path = sample_parameter.get("image_path", None)
        if image_path is None:
            logger.error("No image_path for i2v model / i2vモデルのサンプル画像生成にはimage_pathが必要です")
            return
    else:
        image_path = None

    if self.control_training:
        control_video_path = sample_parameter.get("control_video_path", None)
        if control_video_path is None:
            logger.error(
                "No control_video_path for control model / controlモデルのサンプル画像生成にはcontrol_video_pathが必要です"
            )
            return
    else:
        control_video_path = None

    device = accelerator.device
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()
        generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

    logger.info(f"prompt: {prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"frame count: {frame_count}")
    logger.info(f"sample steps: {sample_steps}")
    logger.info(f"guidance scale: {guidance_scale}")
    logger.info(f"discrete flow shift: {discrete_flow_shift}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    do_classifier_free_guidance = False
    if negative_prompt is not None:
        do_classifier_free_guidance = True
        logger.info(f"negative prompt: {negative_prompt}")
        logger.info(f"cfg scale: {cfg_scale}")

    if self.i2v_training:
        logger.info(f"image path: {image_path}")
    if self.control_training:
        logger.info(f"control video path: {control_video_path}")

    # inference: architecture dependent
    # Check if transformer has self-referencing _orig_mod (compiled model hack)
    # If so, skip eval/train to avoid infinite recursion
    has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
    was_train = transformer.training if not has_self_ref_orig_mod else True
    if not has_self_ref_orig_mod:
        transformer.eval()

    video = self.do_inference(
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=image_path,
        control_video_path=control_video_path,
    )

    if not has_self_ref_orig_mod:
        transformer.train(was_train)

    # Save video
    if video is None:
        logger.error("No video generated / 生成された動画がありません")
        return

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    prompt_idx = sample_parameter.get("enum", 0)
    save_path = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"

    wandb_tracker = None
    try:
        wandb_tracker = accelerator.get_tracker("wandb")  # raises ValueError if wandb is not initialized
        try:
            import wandb
        except ImportError:
            raise ImportError("No wandb / wandb がインストールされていないようです")
    except:  # wandb 無効時
        wandb = None

    if video.shape[2] == 1:
        # In Qwen-Image-Layered, video is (N, C, 1, H, W) where N=Layers, otherwise (1, C, 1, H, W)
        image_paths = save_images_grid(video, save_dir, save_path, n_rows=video.shape[0], create_subdir=False)
        if wandb_tracker is not None and wandb is not None:
            for image_path in image_paths:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
    else:
        video_path = os.path.join(save_dir, save_path) + ".mp4"
        save_videos_grid(video, video_path)
        if wandb_tracker is not None and wandb is not None:
            wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)

    # Move models back to initial state
    vae.to("cpu")
    clean_memory_on_device(device)
