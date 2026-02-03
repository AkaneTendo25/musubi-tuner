"""
ACE-Step 1.5 LoRA training with musubi-tuner infrastructure.

This module provides LoRA fine-tuning for ACE-Step 1.5 audio generation model,
following the NetworkTrainer pattern from musubi-tuner.
"""

import argparse
import os
from typing import Optional

import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_ACESTEP, ARCHITECTURE_ACESTEP_FULL
from musubi_tuner.acestep import acestep_utils
from musubi_tuner.acestep.acestep_config import (
    ACESTEP_LATENT_CHANNELS,
    ACESTEP_SAMPLE_RATE,
    ACESTEP_LATENT_HZ,
    TURBO_SHIFT3_TIMESTEPS,
)
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AceStepNetworkTrainer(NetworkTrainer):
    """LoRA trainer for ACE-Step 1.5 audio generation model."""

    def __init__(self):
        super().__init__()
        self._model_path: str = None
        self._vae_path: str = None
        self._text_encoder_path: str = None
        self._is_turbo: bool = True

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_ACESTEP

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_ACESTEP_FULL

    @property
    def i2v_training(self) -> bool:
        return False  # No image-to-video for audio

    @property
    def control_training(self) -> bool:
        return False  # Control training not supported initially

    def handle_model_specific_args(self, args):
        """Process ACE-Step specific arguments."""
        # ACE-Step only supports bfloat16
        self.dit_dtype = torch.bfloat16
        args.dit_dtype = "bfloat16"

        self._model_path = args.dit
        self._vae_path = args.vae
        self._text_encoder_path = getattr(args, "text_encoder", None)
        self._is_turbo = getattr(args, "is_turbo", True)

        # ACE-Step uses discrete timesteps for turbo model
        self.default_discrete_flow_shift = 3.0
        self.default_guidance_scale = 0.0  # No CFG for turbo

        self._i2v_training = False
        self._control_training = False

        logger.info(f"ACE-Step training mode: {'turbo' if self._is_turbo else 'base'}")

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """Load and process sample prompts for audio generation."""
        device = accelerator.device

        logger.info(f"Loading sample prompts from {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load text encoder for encoding sample prompts
        if args.text_encoder is None:
            logger.warning("No text_encoder specified, cannot encode sample prompts")
            return prompts

        logger.info(f"Loading text encoder from {args.text_encoder}")
        tokenizer, text_encoder = acestep_utils.load_text_encoder(
            args.text_encoder,
            device=str(device),
            dtype=torch.bfloat16,
        )
        text_encoder.eval()

        # Encode prompts
        sample_prompts_te_outputs = {}

        for prompt_dict in prompts:
            prompt = prompt_dict.get("prompt", "")
            lyrics = prompt_dict.get("lyrics", "[Instrumental]")
            negative_prompt = prompt_dict.get("negative_prompt", "")

            # Process prompt
            if prompt and prompt not in sample_prompts_te_outputs:
                logger.info(f"Encoding prompt: {prompt[:50]}...")
                hidden_states, attention_mask = acestep_utils.encode_text_for_acestep(
                    tokenizer,
                    text_encoder,
                    prompt,
                    lyrics,
                    device=str(device),
                )
                sample_prompts_te_outputs[prompt] = (
                    hidden_states.cpu(),
                    attention_mask.cpu(),
                )

            # Process negative prompt if different
            if negative_prompt and negative_prompt not in sample_prompts_te_outputs:
                hidden_states, attention_mask = acestep_utils.encode_text_for_acestep(
                    tokenizer,
                    text_encoder,
                    negative_prompt,
                    "",
                    device=str(device),
                )
                sample_prompts_te_outputs[negative_prompt] = (
                    hidden_states.cpu(),
                    attention_mask.cpu(),
                )

        del tokenizer, text_encoder
        clean_memory_on_device(device)

        # Build sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            prompt = prompt_dict.get("prompt", "")
            if prompt in sample_prompts_te_outputs:
                embed, mask = sample_prompts_te_outputs[prompt]
                prompt_dict_copy["encoder_hidden_states"] = embed
                prompt_dict_copy["encoder_attention_mask"] = mask

            negative_prompt = prompt_dict.get("negative_prompt", "")
            if negative_prompt in sample_prompts_te_outputs:
                neg_embed, neg_mask = sample_prompts_te_outputs[negative_prompt]
                prompt_dict_copy["negative_encoder_hidden_states"] = neg_embed
                prompt_dict_copy["negative_encoder_attention_mask"] = neg_mask

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def do_inference(
        self,
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
        image_path=None,
        control_video_path=None,
    ):
        """Generate audio sample during training.

        For ACE-Step turbo model:
        - Uses 8-step discrete timesteps
        - No CFG (classifier-free guidance disabled)
        - Returns audio as [B, C, F, H, W] format for compatibility (F=1, audio as 2D)
        """
        device = accelerator.device
        model = accelerator.unwrap_model(transformer)

        # Get encoder hidden states from sample parameter
        encoder_hidden_states = sample_parameter.get("encoder_hidden_states")
        encoder_attention_mask = sample_parameter.get("encoder_attention_mask")

        if encoder_hidden_states is None:
            logger.warning("No encoder_hidden_states in sample_parameter, skipping audio generation")
            return None

        encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=torch.bfloat16)
        encoder_attention_mask = encoder_attention_mask.to(device=device)

        # Determine audio duration from sample parameter or use default
        duration = sample_parameter.get("audio_duration", 30.0)  # Default 30 seconds
        latent_length = int(duration * ACESTEP_LATENT_HZ)  # 25 Hz latent rate

        # Process through model.encoder (like original ACE-Step trainer)
        if hasattr(model, "encoder"):
            # Get encoder dtype
            encoder_dtype = next(model.encoder.parameters()).dtype

            # Create dummy lyric embeddings
            lyric_hidden_states = torch.zeros(
                1, 1, encoder_hidden_states.shape[-1],
                device=device, dtype=encoder_dtype
            )
            lyric_attention_mask = torch.zeros(1, 1, device=device, dtype=encoder_attention_mask.dtype)

            # Create dummy refer_audio
            refer_audio_hidden = torch.zeros(1, 1, 64, device=device, dtype=encoder_dtype)
            refer_audio_order_mask = torch.zeros(1, device=device, dtype=torch.long)

            with torch.no_grad():
                encoder_hidden_states, encoder_attention_mask = model.encoder(
                    text_hidden_states=encoder_hidden_states.to(encoder_dtype),
                    text_attention_mask=encoder_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                    refer_audio_order_mask=refer_audio_order_mask,
                )
            # Cast back to bfloat16 for inference
            encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)

        # Initialize noise
        latents = torch.randn(
            1, latent_length, ACESTEP_LATENT_CHANNELS,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        # Attention mask for latents (all ones since we generate full sequence)
        attention_mask = torch.ones(1, latent_length, device=device)

        # Get timesteps for turbo model (8 discrete steps)
        timesteps = torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=torch.bfloat16)
        num_steps = min(sample_steps, len(timesteps)) if sample_steps else len(timesteps)
        timesteps = timesteps[:num_steps]

        # Create context_latents (required by decoder)
        # context_latents = [silence_latent, chunk_masks] -> [1, T, 128]
        if hasattr(model, "silence_latent"):
            silence = model.silence_latent[:, :latent_length, :].to(device=device, dtype=torch.bfloat16)
            if silence.shape[1] < latent_length:
                pad_len = latent_length - silence.shape[1]
                silence = torch.cat([silence, torch.zeros(1, pad_len, 64, device=device, dtype=torch.bfloat16)], dim=1)
        else:
            silence = torch.zeros(1, latent_length, 64, device=device, dtype=torch.bfloat16)
        chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=torch.bfloat16)
        context_latents = torch.cat([silence, chunk_masks], dim=-1)

        # Denoising loop
        logger.info(f"Running {num_steps}-step denoising loop for audio generation")
        for i, t in enumerate(tqdm(timesteps, desc="Audio sampling")):
            t_batch = t.unsqueeze(0).expand(1)
            r_batch = t_batch  # r = t for turbo model

            latent_input = latents.to(dtype=torch.bfloat16)

            with accelerator.autocast(), torch.no_grad():
                if hasattr(model, "decoder"):
                    # Use decoder forward
                    decoder_outputs = model.decoder(
                        hidden_states=latent_input,
                        timestep=t_batch,
                        timestep_r=r_batch,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        context_latents=context_latents,
                    )
                else:
                    # Fallback for models without separate decoder
                    decoder_outputs = model(
                        hidden_states=latent_input,
                        timestep=t_batch,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    )

            # Get velocity prediction
            if isinstance(decoder_outputs, tuple):
                velocity = decoder_outputs[0]
            else:
                velocity = decoder_outputs

            # Euler step: x_{t-1} = x_t - dt * v
            # For discrete timesteps, we compute dt from consecutive timesteps
            if i < num_steps - 1:
                t_next = timesteps[i + 1]
                dt = t - t_next
            else:
                dt = t  # Final step goes to t=0

            latents = latents - dt * velocity

        # Decode latents to audio with VAE
        if vae is not None and not isinstance(vae, type(self)):  # Check it's not VaeStub
            logger.info("Decoding audio latents with VAE")
            try:
                # Move VAE to device
                vae.to(device)
                vae.eval()

                # Transpose latents for VAE: [B, T, C] -> [B, C, T]
                latents_for_vae = latents.permute(0, 2, 1).to(vae.dtype if hasattr(vae, 'dtype') else torch.float32)

                with torch.no_grad():
                    audio = vae.decode(latents_for_vae)
                    if hasattr(audio, "sample"):
                        audio = audio.sample

                # audio shape: [B, 2, samples] (stereo)
                audio = audio.to(torch.float32).cpu()

                # Normalize audio
                audio = audio / audio.abs().max().clamp(min=1e-5)

                vae.to("cpu")
                clean_memory_on_device(device)

                # Return in compatible format: [B, C, F, H, W] where we use F=1, H=2 (channels), W=samples
                # For audio, we reshape to [B, 1, 1, 2, samples] for compatibility
                audio = audio.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 2, samples]
                return audio

            except Exception as e:
                logger.error(f"VAE decode failed: {e}")
                vae.to("cpu")
                clean_memory_on_device(device)
                return None
        else:
            logger.warning("VAE not available for audio decode, returning latents only")
            # Return latents in compatible format
            latents_cpu = latents.to(torch.float32).cpu()
            return latents_cpu.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T, C]

    def save_audio_sample(self, audio_tensor: torch.Tensor, save_path: str, sample_rate: int = ACESTEP_SAMPLE_RATE):
        """Save audio tensor to file.

        Args:
            audio_tensor: Audio tensor [B, 2, samples] or [B, C, F, H, W] format
            save_path: Output file path (without extension)
            sample_rate: Audio sample rate
        """
        try:
            import torchaudio
        except ImportError:
            logger.warning("torchaudio not available, cannot save audio sample")
            return None

        # Handle different tensor formats
        if audio_tensor.dim() == 5:
            # [B, C, F, H, W] format from do_inference
            # Extract audio: [B, 1, 1, 2, samples] -> [2, samples]
            audio = audio_tensor[0, 0, 0]  # [2, samples]
        elif audio_tensor.dim() == 3:
            # [B, 2, samples] format
            audio = audio_tensor[0]  # [2, samples]
        elif audio_tensor.dim() == 2:
            # [2, samples] or [samples, 2]
            audio = audio_tensor
        else:
            logger.warning(f"Unexpected audio tensor shape: {audio_tensor.shape}")
            return None

        # Ensure [2, samples] shape (stereo)
        if audio.shape[0] != 2 and audio.shape[-1] == 2:
            audio = audio.T

        # Normalize to [-1, 1]
        audio = audio.float()
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        # Save as WAV
        audio_path = save_path + ".wav"
        try:
            torchaudio.save(audio_path, audio.cpu(), sample_rate)
            logger.info(f"Saved audio sample to {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load AutoencoderOobleck VAE."""
        if vae_path is None:
            logger.warning("No VAE path specified, using stub for training (assumes cached latents)")

            class _VaeStub:
                def requires_grad_(self, *_, **__):
                    return self

                def eval(self):
                    return self

                def to(self, *_, **__):
                    return self

            return _VaeStub()

        logger.info(f"Loading ACE-Step VAE from {vae_path}")
        vae = acestep_utils.load_acestep_vae(vae_path, device="cpu", dtype=vae_dtype)
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        """Load AceStepConditionGenerationModel."""
        logger.info(f"Loading ACE-Step model from {dit_path}")

        model, config = acestep_utils.load_acestep_model(
            dit_path,
            device=loading_device,
            dtype=dit_weight_dtype or torch.bfloat16,
            attn_mode=attn_mode,
        )

        self._is_turbo = config.get("is_turbo", True)

        # Add gradient checkpointing method compatible with base trainer
        def _enable_gradient_checkpointing(cpu_offload: bool = False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "_set_gradient_checkpointing"):
                model._set_gradient_checkpointing(True)
            logger.info("Enabled gradient checkpointing for ACE-Step model")

        model.enable_gradient_checkpointing = _enable_gradient_checkpointing

        return model

    def compile_transformer(self, args, transformer):
        """Compile transformer blocks for optimization."""
        # ACE-Step decoder structure - compile decoder blocks if available
        if hasattr(transformer, "decoder"):
            return model_utils.compile_transformer(
                args,
                transformer,
                [transformer.decoder],
                disable_linear=self.blocks_to_swap > 0,
            )
        return transformer

    def enable_gradient_checkpointing(self, transformer, cpu_offload: bool = False):
        """Enable gradient checkpointing for ACE-Step model."""
        if hasattr(transformer, "gradient_checkpointing_enable"):
            transformer.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing via gradient_checkpointing_enable()")
        elif hasattr(transformer, "_set_gradient_checkpointing"):
            transformer._set_gradient_checkpointing(True)
            logger.info("Enabled gradient checkpointing via _set_gradient_checkpointing()")
        else:
            logger.warning("Model does not support gradient checkpointing")

    def scale_shift_latents(self, latents):
        """Apply latent scaling (already done in cache for ACE-Step)."""
        return latents  # No additional scaling needed

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
    ):
        """Forward pass through ACE-Step decoder with flow matching loss.

        ACE-Step audio latents: [B, T, 64] (batch, time, channels)
        Note: For audio, we treat the latents differently than image/video.
        """
        bsz = latents.shape[0]

        # Get conditioning from batch
        # These are loaded from the cached text encoder outputs
        encoder_hidden_states = batch.get("encoder_hidden_states")
        encoder_attention_mask = batch.get("encoder_attention_mask")
        attention_mask = batch.get("attention_mask")
        context_latents = batch.get("latents_context")

        if encoder_hidden_states is None:
            # Fallback: load from batch keys used by the base trainer
            # This handles the case where different key names are used
            encoder_hidden_states = batch.get("llm")
            encoder_attention_mask = batch.get("llm_mask")

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(accelerator.device, dtype=network_dtype)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(accelerator.device)
        if context_latents is not None:
            context_latents = context_latents.to(accelerator.device, dtype=network_dtype)

        # Process text embeddings through model.encoder (like original ACE-Step trainer)
        # This projects text from 1024->2048 and combines with dummy lyrics/timbre
        if hasattr(transformer, "encoder") and encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]

            # Get encoder dtype (may be different from network_dtype)
            encoder_dtype = next(transformer.encoder.parameters()).dtype

            # Create dummy lyric embeddings (use embed_tokens if available, else zeros)
            # Original trainer uses: lyric_hidden_states = text_encoder.embed_tokens(lyric_input_ids)
            # For simplicity, use zeros with proper shape
            lyric_hidden_states = torch.zeros(
                bsz, 1, encoder_hidden_states.shape[-1],  # [B, 1, 1024]
                device=accelerator.device, dtype=encoder_dtype
            )
            lyric_attention_mask = torch.zeros(bsz, 1, device=accelerator.device, dtype=encoder_attention_mask.dtype)

            # Create dummy refer_audio (zeros for text2music, like original trainer)
            refer_audio_hidden = torch.zeros(1, 1, 64, device=accelerator.device, dtype=encoder_dtype)
            refer_audio_order_mask = torch.zeros(1, device=accelerator.device, dtype=torch.long)

            # Run through model.encoder to get properly formatted encoder_hidden_states
            # Cast input to encoder dtype to avoid dtype mismatch
            with torch.no_grad():
                encoder_hidden_states, encoder_attention_mask = transformer.encoder(
                    text_hidden_states=encoder_hidden_states.to(encoder_dtype),
                    text_attention_mask=encoder_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                    refer_audio_order_mask=refer_audio_order_mask,
                )
            # Cast back to network_dtype and re-enable gradients for backprop through LoRA
            encoder_hidden_states = encoder_hidden_states.to(network_dtype).detach().requires_grad_(True)

        # Create context_latents if not provided (required by decoder)
        # Original trainer: context_latents = cat([silence_latent, chunk_masks], dim=-1) -> [B, T, 128]
        # For text2music: silence for src_latents, ones for chunk_masks (generate everything)
        if context_latents is None and latents is not None:
            latent_length = latents.shape[1]
            # Get silence latent from model or use zeros
            if hasattr(transformer, "silence_latent"):
                silence = transformer.silence_latent[:, :latent_length, :].to(accelerator.device, dtype=network_dtype)
                if silence.shape[0] < bsz:
                    silence = silence.expand(bsz, -1, -1)
                # Pad if needed
                if silence.shape[1] < latent_length:
                    pad_len = latent_length - silence.shape[1]
                    silence = torch.cat([silence, torch.zeros(bsz, pad_len, 64, device=accelerator.device, dtype=network_dtype)], dim=1)
            else:
                silence = torch.zeros(bsz, latent_length, 64, device=accelerator.device, dtype=network_dtype)

            # chunk_masks = 1 means "generate this region" (text2music generates everything)
            chunk_masks = torch.ones(bsz, latent_length, 64, device=accelerator.device, dtype=network_dtype)
            # context_latents = [src_latents, chunk_masks] -> [B, T, 128]
            context_latents = torch.cat([silence, chunk_masks], dim=-1)

        # Sample discrete timesteps for turbo model
        if self._is_turbo:
            t, r = acestep_utils.sample_discrete_timestep(bsz, accelerator.device, network_dtype)
        else:
            # For non-turbo, use provided timesteps
            t = timesteps.to(accelerator.device, dtype=network_dtype)
            r = t

        # Prepare noisy input
        noisy_model_input = noisy_model_input.to(accelerator.device, dtype=network_dtype)
        latents = latents.to(accelerator.device, dtype=network_dtype)

        # Ensure gradients for gradient checkpointing
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        # Forward through decoder
        # The ACE-Step model has a decoder attribute that processes the latents
        with accelerator.autocast():
            if hasattr(transformer, "decoder"):
                decoder_outputs = transformer.decoder(
                    hidden_states=noisy_model_input,
                    timestep=t,
                    timestep_r=r,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )
            else:
                # Fallback for models without separate decoder attribute
                decoder_outputs = transformer(
                    hidden_states=noisy_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

        # Get prediction (first element of tuple output)
        model_pred = decoder_outputs[0] if isinstance(decoder_outputs, tuple) else decoder_outputs

        # Flow matching target: v = x1 - x0 (noise - latents)
        # This is the velocity field we're predicting
        target = noise - latents

        return model_pred, target

    # endregion model specific


def acestep_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ACE-Step specific arguments."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Path to Qwen3 text encoder (for sample generation)",
    )
    parser.add_argument(
        "--is_turbo",
        action="store_true",
        default=True,
        help="Use turbo model discrete timesteps (default: True)",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=240.0,
        help="Maximum audio duration in seconds (default: 240)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = acestep_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # Force bfloat16 for ACE-Step
    if args.mixed_precision != "bf16":
        logger.warning("ACE-Step requires bf16 precision, overriding mixed_precision")
        args.mixed_precision = "bf16"

    # Set defaults for base trainer arguments not used by ACE-Step
    if not hasattr(args, "fp8_scaled"):
        args.fp8_scaled = False
    if not hasattr(args, "fp8_base"):
        args.fp8_base = False
    if not hasattr(args, "fp8_llm"):
        args.fp8_llm = False
    if not hasattr(args, "disable_numpy_memmap"):
        args.disable_numpy_memmap = False

    trainer = AceStepNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
