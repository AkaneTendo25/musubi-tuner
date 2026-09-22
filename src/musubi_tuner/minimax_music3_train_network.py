"""LoRA training entry point for MiniMax Music 3 ComfyUI DiT checkpoints."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_MUSIC3, ARCHITECTURE_MINIMAX_MUSIC3_FULL
from musubi_tuner.hv_train_network import DiTOutput, NetworkTrainer, read_config_from_file, setup_parser_common
from musubi_tuner.minimax_music3.model import MiniMaxMusic3DiT
from musubi_tuner.minimax_music3.sampling import sample_latents
from musubi_tuner.minimax_music3.ar import generate_conditioning, load_ar
from musubi_tuner.minimax_music3.utils import load_comfy_dit
from musubi_tuner.minimax_music3.vocoder import load_comfy_dav
from musubi_tuner.hv_train_network import load_prompts
from safetensors.torch import load_file, save_file
from musubi_tuner.utils import model_utils


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean flow loss over valid audio frames and all latent channels."""
    squared_error = (prediction.float() - target.float()).square()
    if valid_mask is None:
        return squared_error.mean()
    mask = valid_mask.to(device=squared_error.device, dtype=squared_error.dtype)
    return (squared_error * mask).sum() / (mask.sum() * squared_error.shape[1]).clamp_min(1)


def masked_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine feature distance over valid latent tokens."""
    distance = 1.0 - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1)
    if valid_mask is None:
        return distance.mean()
    mask = valid_mask.to(device=distance.device, dtype=distance.dtype)
    if mask.ndim == 3:
        mask = mask[:, 0]
    return (distance * mask).sum() / mask.sum().clamp_min(1)


def sample_music3_flow_times(
    batch_size: int,
    device: torch.device,
    sampling: str,
    mixflow: bool = False,
    mixflow_gamma: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return model noise-times and interpolation noise-times."""
    if mixflow:
        # MixFlow samples paper data-time from Beta(2,1), represented by
        # sqrt(U). Music 3's interpolation below uses the inverse noise-time.
        noise_time = 1.0 - torch.sqrt(torch.rand(batch_size, device=device))
        slowdown = torch.rand_like(noise_time)
        interpolation_time = noise_time + slowdown * mixflow_gamma * (1.0 - noise_time)
        return noise_time, interpolation_time
    if sampling == "uniform":
        noise_time = torch.rand(batch_size, device=device)
    elif sampling == "cubic":
        noise_time = torch.rand(batch_size, device=device).pow(3.0)
    else:
        noise_time = torch.sigmoid(torch.randn(batch_size, device=device))
    return noise_time, noise_time


def twinflow_rcgm_target(
    base_prediction: torch.Tensor,
    velocity_target: torch.Tensor,
    noisy_latents: torch.Tensor,
    noise_time: torch.Tensor,
    reference_time: torch.Tensor,
    teacher_forward,
    estimate_order: int = 2,
    target_clamp: float = 1.0,
) -> torch.Tensor:
    """Build a recursive consistency target for a data-ward velocity model."""
    sigma = noise_time.view(noise_time.shape[0], 1, 1)
    reference = reference_time.view(reference_time.shape[0], 1, 1)
    current = noisy_latents
    previous = sigma
    accumulated = torch.zeros_like(base_prediction)
    schedule = [reference] if estimate_order <= 1 else [
        sigma + (reference - sigma) * ((index + 1) / estimate_order)
        for index in range(estimate_order)
    ]
    for following in schedule:
        teacher = teacher_forward(current, previous.flatten(), following.flatten())
        clean_estimate = current + previous * teacher
        noise_estimate = current - (1.0 - previous) * teacher
        current = following * noise_estimate + (1.0 - following) * clean_estimate
        accumulated = accumulated + teacher * (previous - following)
        previous = following
    residual = base_prediction.detach() - accumulated - velocity_target
    return base_prediction.detach() - residual.clamp(min=-target_clamp, max=target_clamp)


def resolve_latest_music3_state(output_dir: str | os.PathLike, model_name: str) -> str:
    root = Path(output_dir)
    candidates = [
        path for path in root.glob(f"{model_name}*-state") if path.is_dir() and any(path.iterdir())
    ]
    if not candidates:
        raise ValueError(f"No saved Music 3 training states found for {model_name!r} in {root}")
    return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))


class MiniMaxMusic3NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1
        self._ema_shadow = None

    @property
    def architecture(self):
        return ARCHITECTURE_MINIMAX_MUSIC3

    @property
    def architecture_full_name(self):
        return ARCHITECTURE_MINIMAX_MUSIC3_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 1.0
        if args.convrot_int8 and (args.fp8_base or args.fp8_scaled):
            raise ValueError("--convrot_int8 cannot be combined with fp8 options")
        if args.convrot_int8_bwd == "int8" and not args.convrot_int8:
            raise ValueError("--convrot_int8_bwd int8 requires --convrot_int8")

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        parameters = load_prompts(sample_prompts)
        needs_generation = any(not parameter.get("conditioning_path") for parameter in parameters)
        ar_components = None
        if needs_generation:
            if not args.ar_model:
                raise ValueError("Validation prompts without conditioning_path require --ar_model")
            ar_components = load_ar(args.ar_model, device=accelerator.device, dtype=self.dit_dtype)
        for parameter in parameters:
            path = parameter.get("conditioning_path")
            if path:
                tensors = load_file(path)
                key = next(key for key in tensors if key.startswith("varlen_music3_hidden_"))
                parameter["music3_hidden"] = tensors[key]
                continue
            caption = parameter.get("caption") or parameter.get("prompt")
            lyrics = parameter.get("lyrics")
            if not caption or not lyrics:
                raise ValueError("Validation prompts need caption/prompt and lyrics when conditioning_path is absent")
            duration = float(parameter.get("duration", args.validation_audio_duration))
            frames = max(1, round(duration * 25))
            parameter["music3_hidden"] = generate_conditioning(
                *ar_components, caption, lyrics, frames, seed=int(parameter.get("seed", 0))
            )
        if ar_components is not None:
            del ar_components
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return parameters

    def do_inference(self, accelerator, args, sample_parameter, vae, dit_dtype, transformer,
                     discrete_flow_shift, sample_steps, width, height, frame_count, generator,
                     do_classifier_free_guidance, guidance_scale, cfg_scale, **kwargs):
        latents = sample_latents(
            transformer, sample_parameter["music3_hidden"], num_steps=sample_steps,
            generator=generator, dtype=dit_dtype,
        )
        vae.to(accelerator.device)
        return vae.decode(latents.to(vae.dtype)).float().cpu()

    def load_vae(self, args, vae_dtype, vae_path):
        return load_comfy_dav(vae_path, dtype=vae_dtype)

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir,
                               sample_parameter, epoch, steps):
        import soundfile as sf

        seed = sample_parameter.get("seed", 0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        waveform = self.do_inference(
            accelerator, args, sample_parameter, vae, dit_dtype, transformer, 1.0,
            sample_parameter.get("sample_steps", 30), 0, 0, 1, generator, False, 1.0, None,
        )
        suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        path = os.path.join(save_dir, f"{args.output_name or 'music3'}_{suffix}_{seed}.wav")
        sf.write(path, waveform[0].T.numpy(), 44100)
        vae.to("cpu")

    def load_transformer(
        self, accelerator: Accelerator, args: argparse.Namespace, dit_path: str, attn_mode: str,
        split_attn: bool, loading_device: str, dit_weight_dtype: Optional[torch.dtype],
    ):
        transformer = load_comfy_dit(
            dit_path, device=loading_device, dtype=dit_weight_dtype or self.dit_dtype,
            disable_mmap=args.disable_numpy_memmap, convrot_int8=args.convrot_int8,
            convrot_int8_bwd=args.convrot_int8_bwd, calc_device=accelerator.device,
        )
        transformer.gradient_checkpointing_interval = args.music3_gradient_checkpointing_interval
        transformer.gradient_checkpointing_segment_stride = args.music3_gradient_checkpointing_segment_stride
        if args.music3_flowmap:
            transformer.enable_flowmap_time_conditioning(args.music3_flowmap_gate, args.music3_flowmap_delta_type)
        if args.music3_signed_time:
            transformer.enable_time_sign_conditioning()
        return transformer

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(
            args, transformer, [transformer.diffusion_transformer.transformer.layers],
            disable_linear=self.blocks_to_swap > 0 or args.convrot_int8,
        )

    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params):
        if args.music3_signed_time:
            sign_embedding = transformer.diffusion_transformer.time_sign_embed
            trainable_params.append({"params": sign_embedding.parameters(), "lr": args.learning_rate})
        return trainable_params

    def on_post_save(
        self,
        args,
        accelerator,
        network,
        transformer,
        ckpt_name,
        save_dtype,
        metadata,
        force_sync_upload,
    ):
        del network, metadata, force_sync_upload
        transformer = accelerator.unwrap_model(transformer)
        state = {}
        sign_embedding = transformer.diffusion_transformer.time_sign_embed
        if sign_embedding is not None:
            state["time_sign_embed.weight"] = sign_embedding.weight.detach().to("cpu", dtype=save_dtype).contiguous()
        if self._ema_shadow is not None:
            for name, value in self._ema_shadow.items():
                state[f"ema.{name}"] = value.detach().to("cpu", dtype=save_dtype).contiguous()
        if state:
            stem = os.path.splitext(ckpt_name)[0]
            save_file(state, os.path.join(args.output_dir, f"{stem}_music3_aux.safetensors"))

    def on_train_start(self, args, accelerator, network, transformer, optimizer):
        del optimizer
        if args.music3_ema_decay > 0:
            raw_network = accelerator.unwrap_model(network)
            self._ema_shadow = {
                name: parameter.detach().float().clone()
                for name, parameter in raw_network.named_parameters()
                if parameter.requires_grad
            }
        auxiliary_path = args.music3_auxiliary
        if auxiliary_path is None and args.resume:
            state_path = Path(args.resume)
            if state_path.name.endswith("-state"):
                checkpoint_stem = state_path.name[: -len("-state")]
                candidate = state_path.parent / f"{checkpoint_stem}_music3_aux.safetensors"
                if candidate.exists():
                    auxiliary_path = str(candidate)
        if auxiliary_path:
            auxiliary = load_file(auxiliary_path, device="cpu")
            raw_transformer = accelerator.unwrap_model(transformer)
            sign_embedding = raw_transformer.diffusion_transformer.time_sign_embed
            if "time_sign_embed.weight" in auxiliary:
                if sign_embedding is None:
                    raise ValueError("Auxiliary checkpoint contains signed-time weights but signed-time conditioning is disabled")
                sign_embedding.weight.data.copy_(auxiliary["time_sign_embed.weight"].to(sign_embedding.weight))
            ema_values = {key[len("ema.") :]: value for key, value in auxiliary.items() if key.startswith("ema.")}
            if ema_values:
                if self._ema_shadow is None:
                    raise ValueError("Auxiliary checkpoint contains EMA weights but --music3_ema_decay is disabled")
                missing = set(self._ema_shadow) - set(ema_values)
                if missing:
                    raise ValueError(f"Auxiliary EMA checkpoint is missing {len(missing)} tensors")
                for name in self._ema_shadow:
                    self._ema_shadow[name].copy_(ema_values[name].to(self._ema_shadow[name]))

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        del transformer, global_step
        if not sync_gradients or self._ema_shadow is None:
            return
        decay = args.music3_ema_decay
        raw_network = accelerator.unwrap_model(network)
        with torch.no_grad():
            for name, parameter in raw_network.named_parameters():
                if name in self._ema_shadow:
                    self._ema_shadow[name].lerp_(parameter.detach().float(), 1.0 - decay)

    @contextmanager
    def _ema_weights(self, accelerator, network):
        if self._ema_shadow is None:
            raise RuntimeError("Music 3 EMA teacher is not initialized")
        raw_network = accelerator.unwrap_model(network)
        original = {}
        with torch.no_grad():
            for name, parameter in raw_network.named_parameters():
                if name in self._ema_shadow:
                    original[name] = parameter.detach().clone()
                    parameter.copy_(self._ema_shadow[name].to(parameter))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in raw_network.named_parameters():
                    if name in original:
                        parameter.copy_(original[name])

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input,
        timesteps, network_dtype, **kwargs,
    ) -> DiTOutput:
        contexts = batch["music3_hidden"]
        lengths = [context.shape[0] for context in contexts]
        max_frames = max(lengths)
        context = torch.stack([F.pad(value, (0, 0, 0, max_frames - value.shape[0])) for value in contexts])
        context = context.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        # Musubi supplies noise-time (1 = noise); the Music 3 Fourier embedding
        # uses data-time (0 = noise, 1 = data).
        t = (1.0 - timesteps / 1000.0).to(accelerator.device)
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            context.requires_grad_(True)
        with accelerator.autocast():
            conditioning_scale = torch.ones_like(t)[:, None, None]
            if transformer.training and args.music3_conditioning_dropout > 0:
                keep = torch.rand((t.shape[0], 1, 1), device=t.device) >= args.music3_conditioning_dropout
                conditioning_scale = keep.to(t.dtype)
            result = transformer(
                noisy_model_input,
                t,
                context,
                conditioning_scale,
                valid_mask=batch.get("latents_mask"),
                reference_timestep=kwargs.get("reference_timestep"),
                timestep_sign=kwargs.get("timestep_sign"),
                hidden_state_layer=kwargs.get("hidden_state_layer"),
            )
        if kwargs.get("hidden_state_layer") is not None:
            prediction, hidden_states = result
        else:
            prediction, hidden_states = result, None
        target = latents.to(device=accelerator.device, dtype=torch.float32) - noise.float()
        return DiTOutput(pred=prediction, target=target, extra={"hidden_states": hidden_states})

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise,
                      noise_scheduler, dit_dtype, network_dtype, vae, global_step):
        batch_size = latents.shape[0]
        if args.music3_mixflow or batch.get("timesteps") is None:
            # Music 3 is trained in data-time coordinates: 0 is pure noise and
            # 1 is the clean latent. Sigmoid sampling is the default schedule.
            t, interpolation_t = sample_music3_flow_times(
                batch_size,
                accelerator.device,
                args.music3_timestep_sampling,
                args.music3_mixflow,
                args.music3_mixflow_gamma,
            )
        else:
            t = torch.as_tensor(batch["timesteps"], device=accelerator.device, dtype=torch.float32)
            interpolation_t = t
        latents = latents.to(accelerator.device, dtype=torch.float32)
        noise = noise.to(accelerator.device, dtype=torch.float32)
        input_noise = noise
        if args.music3_input_perturbation > 0:
            input_noise = noise + args.music3_input_perturbation * torch.randn_like(noise)
        valid_mask = batch.get("latents_mask")
        if args.music3_self_flow:
            alternate_t, _ = sample_music3_flow_times(
                batch_size,
                accelerator.device,
                args.music3_timestep_sampling,
            )
            token_mask = torch.rand((batch_size, latents.shape[-1]), device=accelerator.device)
            token_mask = token_mask < args.music3_self_flow_mask_ratio
            token_t = torch.where(token_mask, alternate_t[:, None], t[:, None])
            student_noisy = (1 - token_t[:, None, :]) * latents + token_t[:, None, :] * input_noise
            output = self.call_dit(
                args,
                accelerator,
                transformer,
                latents,
                batch,
                noise,
                student_noisy,
                token_t * 1000,
                network_dtype,
                hidden_state_layer=args.music3_self_flow_student_layer,
            )
            teacher_t = torch.minimum(t, alternate_t)
            teacher_noisy = (1 - teacher_t[:, None, None]) * latents + teacher_t[:, None, None] * input_noise
            with self._ema_weights(accelerator, network), torch.no_grad():
                teacher_output = self.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents,
                    batch,
                    noise,
                    teacher_noisy,
                    teacher_t * 1000,
                    network_dtype,
                    hidden_state_layer=args.music3_self_flow_teacher_layer,
                )
            base_loss = masked_mse_loss(output.pred, output.target, valid_mask)
            feature_loss = masked_cosine_distance(
                output.extra["hidden_states"], teacher_output.extra["hidden_states"], valid_mask
            )
            loss = base_loss + args.music3_self_flow_weight * feature_loss
            return loss, {
                "self_flow/base": base_loss.detach().item(),
                "self_flow/features": feature_loss.detach().item(),
            }
        noisy = (1 - interpolation_t[:, None, None]) * latents + interpolation_t[:, None, None] * input_noise
        output = self.call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy, t * 1000, network_dtype
        )
        base_loss = masked_mse_loss(output.pred, output.target, valid_mask)
        if not args.music3_twinflow:
            return base_loss, {}

        reference_noise_time = t - torch.rand_like(t) * t

        def teacher_forward(current_latents, previous_noise_time, following_noise_time):
            with torch.no_grad():
                teacher_output = self.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents,
                    batch,
                    noise,
                    current_latents,
                    previous_noise_time * 1000,
                    network_dtype,
                    reference_timestep=1.0 - following_noise_time,
                )
            return teacher_output.pred.detach()

        with self._ema_weights(accelerator, network):
            consistency_target = twinflow_rcgm_target(
                output.pred,
                output.target,
                noisy,
                t,
                reference_noise_time,
                teacher_forward,
                args.music3_twinflow_estimate_order,
                args.music3_twinflow_target_clamp,
            )
        consistency_loss = masked_mse_loss(output.pred, consistency_target, valid_mask)
        loss = base_loss + args.music3_twinflow_weight * consistency_loss
        return loss, {
            "twinflow/base": base_loss.detach().item(),
            "twinflow/consistency": consistency_loss.detach().item(),
        }


def setup_music3_parser(parser):
    parser.add_argument("--convrot_int8", action="store_true", help="quantize frozen transformer block Linears to ConvRot int8")
    parser.add_argument("--convrot_int8_bwd", choices=("native", "int8"), default="native")
    parser.add_argument("--ar_model", help="MiniMax Music 3 repository/path used to encode validation caption and lyrics")
    parser.add_argument("--validation_audio_duration", type=float, default=30.0)
    parser.add_argument("--music3_input_perturbation", type=float, default=0.0)
    parser.add_argument("--music3_conditioning_dropout", type=float, default=0.0)
    parser.add_argument("--music3_mixflow", action="store_true")
    parser.add_argument("--music3_mixflow_gamma", type=float, default=0.8)
    parser.add_argument("--music3_flowmap", action="store_true")
    parser.add_argument("--music3_flowmap_gate", type=float, default=0.25)
    parser.add_argument("--music3_flowmap_delta_type", choices=("r", "t-r"), default="r")
    parser.add_argument("--music3_signed_time", action="store_true")
    parser.add_argument("--music3_ema_decay", type=float, default=0.0)
    parser.add_argument("--music3_auxiliary", help="signed-time and EMA companion checkpoint")
    parser.add_argument("--music3_twinflow", action="store_true")
    parser.add_argument("--music3_twinflow_weight", type=float, default=1.0)
    parser.add_argument("--music3_twinflow_estimate_order", type=int, default=2)
    parser.add_argument("--music3_twinflow_target_clamp", type=float, default=1.0)
    parser.add_argument("--music3_self_flow", action="store_true")
    parser.add_argument("--music3_self_flow_weight", type=float, default=0.5)
    parser.add_argument("--music3_self_flow_mask_ratio", type=float, default=0.5)
    parser.add_argument("--music3_self_flow_student_layer", type=int, default=22)
    parser.add_argument("--music3_self_flow_teacher_layer", type=int, default=22)
    parser.add_argument("--music3_gradient_checkpointing_interval", type=int, default=1)
    parser.add_argument("--music3_gradient_checkpointing_segment_stride", type=int, default=1)
    parser.add_argument(
        "--music3_timestep_sampling",
        choices=("sigmoid", "uniform", "cubic"),
        default="sigmoid",
        help="flow-matching data-time distribution; sigmoid is the default",
    )
    return parser


def main():
    parser = setup_music3_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    if args.resume == "latest":
        args.resume = resolve_latest_music3_state(args.output_dir, args.output_name)
    if args.music3_input_perturbation < 0:
        parser.error("--music3_input_perturbation must be non-negative")
    if not 0 <= args.music3_conditioning_dropout < 1:
        parser.error("--music3_conditioning_dropout must be in [0,1)")
    if not 0 <= args.music3_mixflow_gamma <= 1:
        parser.error("--music3_mixflow_gamma must be in [0,1]")
    if not 0 <= args.music3_flowmap_gate <= 1:
        parser.error("--music3_flowmap_gate must be in [0,1]")
    if not 0 <= args.music3_ema_decay < 1:
        parser.error("--music3_ema_decay must be in [0,1)")
    if args.music3_twinflow and not args.music3_flowmap:
        parser.error("--music3_twinflow requires --music3_flowmap")
    if args.music3_twinflow and args.music3_ema_decay <= 0:
        parser.error("--music3_twinflow requires a positive --music3_ema_decay")
    if args.music3_twinflow_estimate_order < 1:
        parser.error("--music3_twinflow_estimate_order must be positive")
    if args.music3_twinflow_target_clamp <= 0:
        parser.error("--music3_twinflow_target_clamp must be positive")
    if args.music3_self_flow and args.music3_twinflow:
        parser.error("--music3_self_flow and --music3_twinflow are mutually exclusive")
    if args.music3_self_flow and args.music3_ema_decay <= 0:
        parser.error("--music3_self_flow requires a positive --music3_ema_decay")
    if args.music3_self_flow_weight <= 0:
        parser.error("--music3_self_flow_weight must be positive")
    if not 0 < args.music3_self_flow_mask_ratio <= 1:
        parser.error("--music3_self_flow_mask_ratio must be in (0,1]")
    if args.music3_self_flow_student_layer < 0 or args.music3_self_flow_teacher_layer < 0:
        parser.error("self-flow layers must be non-negative")
    if args.music3_gradient_checkpointing_interval < 1:
        parser.error("--music3_gradient_checkpointing_interval must be positive")
    if args.music3_gradient_checkpointing_segment_stride < args.music3_gradient_checkpointing_interval:
        parser.error("checkpointing segment stride must be >= checkpointing interval")
    args.dit_dtype = "float16"
    args.vae_dtype = "float32"
    trainer = MiniMaxMusic3NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
