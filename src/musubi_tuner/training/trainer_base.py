"""NetworkTrainer base class shared by architecture-specific training scripts."""

import argparse
from typing import Any, Optional

import torch
from accelerate import Accelerator

from musubi_tuner.training import batch_state as _batch_state
from musubi_tuner.training import model_helpers as _model_helpers
from musubi_tuner.training import optimizer_setup as _optimizer_setup
from musubi_tuner.training import resume_utils as _resume_utils
from musubi_tuner.training import sampling_runtime as _sampling_runtime
from musubi_tuner.training import timestep_logging as _timestep_logging
from musubi_tuner.training import timestep_sampling as _timestep_sampling
from musubi_tuner.training import trainer_hooks as _trainer_hooks
from musubi_tuner.training import trainer_logging as _trainer_logging
from musubi_tuner.training import training_loop as _training_loop
from musubi_tuner.training.metadata import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_MINIMUM_KEYS,
)
from musubi_tuner.training.outputs import DiTOutput as DiTOutput


class NetworkTrainer:
    def __init__(self):
        self.blocks_to_swap = None
        self.timestep_range_pool = []
        self.num_timestep_buckets: Optional[int] = None
        self.vae_frame_stride = 4
        self.default_discrete_flow_shift = 14.5
        self._current_batch_latents_info: Optional[dict[str, Any]] = None
        self.training = False

    @property
    def architecture(self) -> str:
        raise NotImplementedError

    @property
    def architecture_full_name(self) -> str:
        raise NotImplementedError

    def handle_model_specific_args(self, args: argparse.Namespace):
        pass

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        raise NotImplementedError

    def do_inference(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        sample_parameter: dict,
        vae,
        dit_dtype: torch.dtype,
        transformer,
        discrete_flow_shift: float,
        sample_steps: int,
        width: int,
        height: int,
        frame_count: int,
        generator: torch.Generator,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        cfg_scale: float | None,
        image_path: str | None = None,
        control_video_path: str | None = None,
    ):
        raise NotImplementedError

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        raise NotImplementedError

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        raise NotImplementedError

    def compile_transformer(self, args, transformer):
        if args.compile:
            transformer = torch.compile(transformer, backend=args.dynamo_backend, mode=args.dynamo_mode)
        return transformer

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> DiTOutput:
        raise NotImplementedError


NetworkTrainer.train = _training_loop.train
NetworkTrainer.set_current_batch_latents_info = _batch_state.set_current_batch_latents_info
NetworkTrainer.get_current_batch_latents_info = _batch_state.get_current_batch_latents_info
NetworkTrainer.get_checkpoint_metadata = _model_helpers.get_checkpoint_metadata
NetworkTrainer.post_save_checkpoint_hook = _model_helpers.post_save_checkpoint_hook
NetworkTrainer.i2v_training = property(_model_helpers.i2v_training)
NetworkTrainer.control_training = property(_model_helpers.control_training)
NetworkTrainer._resolve_network_module = _model_helpers.resolve_network_module
NetworkTrainer.convert_weight_keys = _model_helpers.convert_weight_keys
NetworkTrainer.load_network_weights = _model_helpers.load_network_weights
NetworkTrainer.resume_from_local_or_hf_if_specified = _resume_utils.resume_from_local_or_hf_if_specified
NetworkTrainer._recover_global_step = staticmethod(_resume_utils.recover_global_step)
NetworkTrainer._state_dir_matches_output_name = staticmethod(_resume_utils.state_dir_matches_output_name)
NetworkTrainer._find_latest_state_dir = staticmethod(_resume_utils.find_latest_state_dir)
NetworkTrainer.sample_images = _sampling_runtime.sample_images
NetworkTrainer.sample_image_inference = _sampling_runtime.sample_image_inference
NetworkTrainer.get_bucketed_timestep = _timestep_sampling.get_bucketed_timestep
NetworkTrainer.get_noisy_model_input_and_timesteps = _timestep_sampling.get_noisy_model_input_and_timesteps
NetworkTrainer.show_timesteps = _timestep_sampling.show_timesteps
NetworkTrainer._get_tensorboard_writer = _timestep_logging.get_tensorboard_writer
NetworkTrainer._should_log_timestep_distribution_to_tensorboard = _timestep_logging.should_log_timestep_distribution_to_tensorboard
NetworkTrainer._get_timestep_distribution_logging_payload = _timestep_logging.get_timestep_distribution_logging_payload
NetworkTrainer._prepare_timestep_distribution_values = _timestep_logging.prepare_timestep_distribution_values
NetworkTrainer._accumulate_timestep_distribution = _timestep_logging.accumulate_timestep_distribution
NetworkTrainer._log_timestep_distribution_histogram = _timestep_logging.log_timestep_distribution_histogram
NetworkTrainer.get_optimizer = _optimizer_setup.get_optimizer
NetworkTrainer.is_schedulefree_optimizer = _optimizer_setup.is_schedulefree_optimizer
NetworkTrainer.get_dummy_scheduler = _optimizer_setup.get_dummy_scheduler
NetworkTrainer.get_lr_scheduler = _optimizer_setup.get_lr_scheduler
NetworkTrainer._enable_lycoris_fp8_forward_compat = _optimizer_setup.enable_lycoris_fp8_forward_compat
NetworkTrainer._maybe_wrap_group_warmup_scheduler = _optimizer_setup.maybe_wrap_group_warmup_scheduler
NetworkTrainer._prepare_network_optimizer_params = _optimizer_setup.prepare_network_optimizer_params
NetworkTrainer._copy_optimizer_state_subset = staticmethod(_optimizer_setup.copy_optimizer_state_subset)
NetworkTrainer._get_prodigy_plus_split_groups = staticmethod(_optimizer_setup.get_prodigy_plus_split_groups)
NetworkTrainer._refresh_prodigy_plus_late_param_group_state = staticmethod(
    _optimizer_setup.refresh_prodigy_plus_late_param_group_state
)
NetworkTrainer._refresh_optimizer_after_adaptive_rank_prune = _optimizer_setup.refresh_optimizer_after_adaptive_rank_prune
NetworkTrainer._refresh_optimizer_param_groups_after_adaptive_rank_resume = (
    _optimizer_setup.refresh_optimizer_param_groups_after_adaptive_rank_resume
)
NetworkTrainer._register_optimizer_resume_safe_globals = staticmethod(_optimizer_setup.register_optimizer_resume_safe_globals)
NetworkTrainer.generate_step_logs = _trainer_logging.generate_step_logs
NetworkTrainer.is_model_parallel_enabled = _trainer_hooks.is_model_parallel_enabled
NetworkTrainer.validate_model_parallel_setup = _trainer_hooks.validate_model_parallel_setup
NetworkTrainer.enable_model_parallel_transformer = _trainer_hooks.enable_model_parallel_transformer
NetworkTrainer.place_network_for_model_parallel = _trainer_hooks.place_network_for_model_parallel
NetworkTrainer.clip_grad_norm_for_model_parallel = _trainer_hooks.clip_grad_norm_for_model_parallel
NetworkTrainer.pre_train_hook = _trainer_hooks.pre_train_hook
NetworkTrainer.compute_prior_divergence_addition = _trainer_hooks.compute_prior_divergence_addition
NetworkTrainer.preservation_backward = _trainer_hooks.preservation_backward
NetworkTrainer.compute_validation_extra_loss = _trainer_hooks.compute_validation_extra_loss
NetworkTrainer.modify_video_loss_per_element = _trainer_hooks.modify_video_loss_per_element
NetworkTrainer.modify_audio_loss_per_element = _trainer_hooks.modify_audio_loss_per_element
NetworkTrainer.compute_video_extra_loss = _trainer_hooks.compute_video_extra_loss
NetworkTrainer.apply_differential_guidance_target = _trainer_hooks.apply_differential_guidance_target

__all__ = [
    "DiTOutput",
    "NetworkTrainer",
    "SS_METADATA_KEY_BASE_MODEL_VERSION",
    "SS_METADATA_KEY_NETWORK_ALPHA",
    "SS_METADATA_KEY_NETWORK_ARGS",
    "SS_METADATA_KEY_NETWORK_DIM",
    "SS_METADATA_KEY_NETWORK_MODULE",
    "SS_METADATA_MINIMUM_KEYS",
]
