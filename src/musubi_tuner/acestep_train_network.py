"""
ACE-Step 1.5 LoRA training with musubi-tuner infrastructure.

This module provides LoRA fine-tuning for ACE-Step 1.5 audio generation model,
following the NetworkTrainer pattern from musubi-tuner.
"""

import argparse
import os
from typing import Optional

import torch
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


def _get_embed_tokens_layer(text_encoder):
    if hasattr(text_encoder, "embed_tokens"):
        return text_encoder.embed_tokens
    if hasattr(text_encoder, "model") and hasattr(text_encoder.model, "embed_tokens"):
        return text_encoder.model.embed_tokens
    if hasattr(text_encoder, "get_input_embeddings"):
        return text_encoder.get_input_embeddings()
    return None


class AceStepNetworkTrainer(NetworkTrainer):
    """LoRA trainer for ACE-Step 1.5 audio generation model."""

    def __init__(self):
        super().__init__()
        self._model_path: str = None
        self._vae_path: str = None
        self._text_encoder_path: str = None
        self._is_turbo: bool = True
        self._silence_latent: Optional[torch.Tensor] = None  # Loaded from silence_latent.pt

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
        sample_prompts_lyric_outputs = {}
        embed_tokens = _get_embed_tokens_layer(text_encoder)

        for prompt_dict in prompts:
            prompt = prompt_dict.get("prompt", "")
            lyrics = prompt_dict.get("lyrics", "[Instrumental]")
            negative_prompt = prompt_dict.get("negative_prompt", "")

            # Process prompt
            if prompt and prompt not in sample_prompts_te_outputs:
                logger.info(f"Encoding prompt: {prompt[:50]}...")
                bpm = prompt_dict.get("bpm")
                keyscale = prompt_dict.get("keyscale", prompt_dict.get("key"))
                timesig = prompt_dict.get("timesignature")
                if isinstance(timesig, str):
                    timesig = int(timesig) if timesig.isdigit() else None
                duration_meta = prompt_dict.get("duration")

                # Match original flow: text prompt and lyrics are separate encoder branches.
                formatted_prompt = acestep_utils.format_text_for_acestep(
                    prompt,
                    "",
                    bpm=bpm,
                    key=keyscale,
                    time_signature=timesig,
                    duration=duration_meta,
                )
                if "\n\n# Lyrics\n" in formatted_prompt:
                    formatted_prompt = formatted_prompt.split("\n\n# Lyrics\n", 1)[0]

                text_inputs = tokenizer(
                    formatted_prompt,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs["input_ids"].to(device)
                attention_mask = text_inputs["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = text_encoder(input_ids)
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                sample_prompts_te_outputs[prompt] = (
                    hidden_states.cpu(),
                    attention_mask.cpu(),
                )

                # Format lyrics with language header like official handler
                lyric_text = acestep_utils.format_lyrics_for_acestep(lyrics, language="unknown")
                lyric_inputs = tokenizer(
                    lyric_text,
                    max_length=2048,  # Official uses 2048 for lyrics
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                lyric_input_ids = lyric_inputs["input_ids"].to(device)
                lyric_attention_mask = lyric_inputs["attention_mask"].to(device)
                if embed_tokens is not None:
                    with torch.no_grad():
                        lyric_hidden_states = embed_tokens(lyric_input_ids)
                else:
                    lyric_hidden_states = torch.zeros(
                        lyric_input_ids.shape[0], lyric_input_ids.shape[1], hidden_states.shape[-1], device=device, dtype=hidden_states.dtype
                    )
                sample_prompts_lyric_outputs[prompt] = (
                    lyric_hidden_states.cpu(),
                    lyric_attention_mask.cpu(),
                )

            # Process negative prompt if different
            if negative_prompt and negative_prompt not in sample_prompts_te_outputs:
                formatted_negative = acestep_utils.format_text_for_acestep(negative_prompt, "")
                if "\n\n# Lyrics\n" in formatted_negative:
                    formatted_negative = formatted_negative.split("\n\n# Lyrics\n", 1)[0]
                neg_inputs = tokenizer(
                    formatted_negative,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                neg_input_ids = neg_inputs["input_ids"].to(device)
                attention_mask = neg_inputs["attention_mask"].to(device)
                with torch.no_grad():
                    neg_outputs = text_encoder(neg_input_ids)
                    hidden_states = neg_outputs.last_hidden_state if hasattr(neg_outputs, "last_hidden_state") else neg_outputs[0]
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
                prompt_dict_copy["text_hidden_states"] = embed
                prompt_dict_copy["text_attention_mask"] = mask
            if prompt in sample_prompts_lyric_outputs:
                lyr_embed, lyr_mask = sample_prompts_lyric_outputs[prompt]
                prompt_dict_copy["lyric_hidden_states"] = lyr_embed
                prompt_dict_copy["lyric_attention_mask"] = lyr_mask

            negative_prompt = prompt_dict.get("negative_prompt", "")
            if negative_prompt in sample_prompts_te_outputs:
                neg_embed, neg_mask = sample_prompts_te_outputs[negative_prompt]
                prompt_dict_copy["negative_text_hidden_states"] = neg_embed
                prompt_dict_copy["negative_text_attention_mask"] = neg_mask

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
        - Uses 8-step discrete timesteps with ODE solver
        - No CFG (classifier-free guidance disabled)
        - Returns audio as [B, C, F, H, W] format for compatibility (F=1, audio as 2D)

        This implementation follows the official ACE-Step generate_audio method.
        """
        device = accelerator.device
        model = accelerator.unwrap_model(transformer)
        dtype = torch.bfloat16

        # Ensure model is in eval mode for inference
        was_training = model.training
        model.eval()

        # Get text encoder hidden states from sample parameter
        text_hidden_states = sample_parameter.get("text_hidden_states")
        text_attention_mask = sample_parameter.get("text_attention_mask")
        if text_hidden_states is None:
            text_hidden_states = sample_parameter.get("encoder_hidden_states")
            text_attention_mask = sample_parameter.get("encoder_attention_mask")
        lyric_hidden_states = sample_parameter.get("lyric_hidden_states")
        lyric_attention_mask = sample_parameter.get("lyric_attention_mask")

        if text_hidden_states is None:
            logger.warning("No encoder_hidden_states in sample_parameter, skipping audio generation")
            if was_training:
                model.train()
            return None

        text_hidden_states = text_hidden_states.to(device=device, dtype=dtype)
        text_attention_mask = text_attention_mask.to(device=device)

        # Log input shape for debugging
        logger.info(f"text_hidden_states shape: {text_hidden_states.shape}, unique values: {text_hidden_states.unique().numel()}")

        # Determine audio duration from sample parameter or use default
        duration = sample_parameter.get("audio_duration", 30.0)  # Default 30 seconds
        latent_length = int(duration * ACESTEP_LATENT_HZ)  # 25 Hz latent rate

        # Get silence_latent (loaded from silence_latent.pt during model loading)
        silence_latent = acestep_utils.get_silence_latent(
            self._silence_latent, device=device, dtype=dtype, latent_length=latent_length
        )

        # For text2music, src_latents = silence_latent
        src_latents = silence_latent.expand(1, -1, -1)

        # Lyric branch: use cached lyrics when available.
        if lyric_hidden_states is not None and lyric_attention_mask is not None:
            lyric_hidden_states = lyric_hidden_states.to(device=device, dtype=dtype)
            lyric_attention_mask = lyric_attention_mask.to(device=device)
        else:
            lyric_hidden_states = torch.zeros(1, 1, text_hidden_states.shape[-1], device=device, dtype=dtype)
            lyric_attention_mask = torch.zeros(1, 1, device=device, dtype=text_attention_mask.dtype)

        # Create dummy refer_audio (no reference audio for text2music)
        refer_audio_hidden = torch.zeros(1, 1, 64, device=device, dtype=dtype)
        refer_audio_order_mask = torch.zeros(1, device=device, dtype=torch.long)

        # Chunk masks: all ones = generate everything (text2music mode)
        chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)

        # is_covers = 0 for text2music (not a cover song)
        is_covers = torch.zeros(1, device=device, dtype=torch.long)

        # Attention mask for latents (all ones since we generate full sequence)
        attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

        # Use model.generate_audio to match original ACE-Step inference path exactly.
        # This avoids subtle drift from hand-rolled denoising loops.
        seed = generator.initial_seed() if generator is not None else None
        t_schedule = torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=dtype)
        num_steps = min(sample_steps, len(t_schedule)) if sample_steps else len(t_schedule)
        t_schedule = t_schedule[:num_steps]
        logger.info(f"Running model.generate_audio with {num_steps} steps (seed={seed})")
        with torch.no_grad():
            gen_outputs = model.generate_audio(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
                refer_audio_order_mask=refer_audio_order_mask,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                silence_latent=silence_latent,
                attention_mask=attention_mask,
                seed=seed,
                fix_nfe=num_steps,
                infer_method="ode",
                shift=3.0,
                timesteps=t_schedule,
            )
        latents = gen_outputs["target_latents"]

        # Restore model training state
        if was_training:
            model.train()

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
                # Cast to float32 like official handler (no value normalization needed,
                # VAE output is already in reasonable range)
                audio = audio.to(torch.float32).cpu()

                # Clamp to valid audio range [-1, 1] to prevent clipping artifacts
                audio = torch.clamp(audio, -1.0, 1.0)

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

        # Clamp to valid audio range [-1, 1] (VAE output should already be in range)
        audio = audio.float()
        audio = torch.clamp(audio, -1.0, 1.0)

        # Save as WAV
        audio_path = save_path + ".wav"
        try:
            torchaudio.save(audio_path, audio.cpu(), sample_rate)
            logger.info(f"Saved audio sample to {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        """Override to save audio samples instead of images/videos."""
        import time
        import os

        sample_steps = sample_parameter.get("sample_steps", 8)  # Default to 8 steps for turbo
        seed = sample_parameter.get("seed")
        prompt = sample_parameter.get("prompt", "")

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            actual_seed = seed
        else:
            actual_seed = torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(actual_seed)

        logger.info(f"prompt: {prompt}")
        logger.info(f"actual seed being used: {actual_seed}")
        logger.info(f"sample steps: {sample_steps}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        # Check for self-referencing _orig_mod (compiled model)
        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_train = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()

        # Generate audio
        audio = self.do_inference(
            accelerator,
            args,
            sample_parameter,
            vae,
            dit_dtype,
            transformer,
            discrete_flow_shift=3.0,  # Turbo shift
            sample_steps=sample_steps,
            width=0,  # Not used for audio
            height=0,  # Not used for audio
            frame_count=0,  # Not used for audio
            generator=generator,
            do_classifier_free_guidance=False,  # Turbo doesn't use CFG
            guidance_scale=1.0,
            cfg_scale=None,
        )

        if not has_self_ref_orig_mod:
            transformer.train(was_train)

        # Save audio
        if audio is None:
            logger.error("No audio generated")
            return

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = os.path.join(
            save_dir,
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        self.save_audio_sample(audio, save_path)

        # Cleanup
        vae.to("cpu")
        clean_memory_on_device(device)

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
        # Store silence_latent from checkpoint (loaded from silence_latent.pt)
        self._silence_latent = config.get("silence_latent", None)

        # Add gradient checkpointing method compatible with base trainer
        def _enable_gradient_checkpointing(cpu_offload: bool = False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "_set_gradient_checkpointing"):
                model._set_gradient_checkpointing(True)
            logger.info("Enabled gradient checkpointing for ACE-Step model")

        model.enable_gradient_checkpointing = _enable_gradient_checkpointing

        # Add block swap methods (no-op for ACE-Step, doesn't support block swapping)
        def _switch_block_swap_for_inference():
            pass  # ACE-Step doesn't use block swapping

        def _switch_block_swap_for_training():
            pass  # ACE-Step doesn't use block swapping

        model.switch_block_swap_for_inference = _switch_block_swap_for_inference
        model.switch_block_swap_for_training = _switch_block_swap_for_training

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
        elif encoder_hidden_states is not None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1],
                device=accelerator.device,
                dtype=torch.long,
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(accelerator.device)
        if context_latents is not None:
            context_latents = context_latents.to(accelerator.device, dtype=network_dtype)

        needs_condition_encode = False
        if hasattr(transformer, "encoder") and encoder_hidden_states is not None:
            # If cache was built with --dit, encoder_hidden_states are already conditioned.
            # If cache was built from raw text encoder outputs, we need to run condition encoder here.
            expected_hidden_size = getattr(getattr(transformer, "config", None), "hidden_size", None)
            if expected_hidden_size is None:
                needs_condition_encode = True
            else:
                needs_condition_encode = encoder_hidden_states.shape[-1] != expected_hidden_size

        # Process text embeddings through model.encoder when needed.
        if needs_condition_encode:
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
            refer_audio_hidden = torch.zeros(bsz, 1, 64, device=accelerator.device, dtype=encoder_dtype)
            refer_audio_order_mask = torch.zeros(bsz, device=accelerator.device, dtype=torch.long)

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

        # Ignore malformed cached context and rebuild from model silence/chunk masks.
        if context_latents is not None:
            if context_latents.dim() != 3 or context_latents.shape[-1] != 128:
                logger.warning("Invalid cached context_latents detected; rebuilding context_latents from model")
                context_latents = None

        # Create context_latents if not provided (required by decoder)
        # Original trainer: context_latents = cat([silence_latent, chunk_masks], dim=-1) -> [B, T, 128]
        # For text2music: silence for src_latents, ones for chunk_masks (generate everything)
        if context_latents is None and latents is not None:
            latent_length = latents.shape[1]
            # Get silence latent (loaded from silence_latent.pt during model loading)
            silence = acestep_utils.get_silence_latent(
                self._silence_latent, device=accelerator.device, dtype=network_dtype, latent_length=latent_length
            )
            if silence.shape[0] < bsz:
                silence = silence.expand(bsz, -1, -1)

            # chunk_masks = 1 means "generate this region" (text2music generates everything)
            chunk_masks = torch.ones(bsz, latent_length, 64, device=accelerator.device, dtype=network_dtype)
            # context_latents = [src_latents, chunk_masks] -> [B, T, 128]
            context_latents = torch.cat([silence, chunk_masks], dim=-1)

        # Prepare inputs
        latents = latents.to(accelerator.device, dtype=network_dtype)  # x0 (data)
        noise = noise.to(accelerator.device, dtype=network_dtype)  # x1 (noise)

        # Sample discrete timesteps for turbo model (must recompute noisy input with these!)
        # Original ACE-Step trainer: t, r = sample_discrete_timestep(bsz, device, dtype)
        if self._is_turbo:
            t, r = acestep_utils.sample_discrete_timestep(bsz, accelerator.device, network_dtype)
        else:
            # For non-turbo, use provided timesteps (normalized to 0-1)
            t = (timesteps.to(accelerator.device, dtype=network_dtype) / 1000.0).clamp(0, 1)
            r = t

        # Recompute noisy model input with correct timesteps (flow matching interpolation)
        # Original ACE-Step: xt = t * x1 + (1 - t) * x0 = t * noise + (1 - t) * latents
        t_expanded = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting with [B, T, C]
        noisy_model_input = t_expanded * noise + (1.0 - t_expanded) * latents

        # Ensure gradients for gradient checkpointing
        if args.gradient_checkpointing:
            noisy_model_input = noisy_model_input.detach().requires_grad_(True)

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
        # Original ACE-Step: flow = x1 - x0; loss = MSE(pred, flow)
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
