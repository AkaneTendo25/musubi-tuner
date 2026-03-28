import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_COSMOS, ARCHITECTURE_COSMOS_FULL
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils.device_utils import synchronize_device
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.cosmos.cosmos_dit import CosmosDiT, load_cosmos_dit, COSMOS_CONFIGS
from musubi_tuner.cosmos.rectified_flow import RectifiedFlow

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.utils import model_utils

# Cosmos uses WAN 2.1 VAE and Qwen2.5-VL text encoder
from musubi_tuner.wan.modules.vae import WanVAE


class CosmosNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.rectified_flow = None

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_COSMOS

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_COSMOS_FULL

    def handle_model_specific_args(self, args):
        self.cosmos_config_name = args.cosmos_model_size  # "2B" or "14B"
        self.cosmos_config = COSMOS_CONFIGS[self.cosmos_config_name]

        # Detect dit dtype from args
        self.dit_dtype = torch.bfloat16
        if args.mixed_precision == "fp16":
            self.dit_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            self.dit_dtype = torch.bfloat16

        if args.fp8_scaled and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights are already in fp8 format, cannot scale to fp8. "
                "Please use fp16/bf16 weights."
            )

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        # Rectified flow parameters
        self.rf_shift = args.cosmos_rf_shift
        self.rf_time_distribution = args.cosmos_time_distribution

        # Qwen2.5-VL text encoder config
        self.text_encoder_name = args.text_encoder
        self.text_len = args.cosmos_text_len

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None
        vae = WanVAE(vae_path=vae_path, device="cpu", dtype=vae_dtype, cache_device=cache_device)
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
        fp8_scaled = args.fp8_scaled

        logger.info(f"Loading Cosmos DiT model ({self.cosmos_config_name}) from {dit_path}")
        model = load_cosmos_dit(
            config_name=self.cosmos_config_name,
            checkpoint_path=dit_path,
            device=loading_device,
            dtype=None if fp8_scaled else dit_weight_dtype,
            fp8_scaled=fp8_scaled,
        )

        # Initialize rectified flow scheduler
        device = accelerator.device if torch.cuda.is_available() else "cpu"
        self.rectified_flow = RectifiedFlow(
            train_time_distribution=self.rf_time_distribution,
            train_time_weight_method="uniform",
            shift=self.rf_shift,
            device=device,
            dtype=torch.float32,
        )

        return model

    def compile_transformer(self, args, transformer):
        transformer: CosmosDiT = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # Cosmos uses raw latents from WAN VAE without additional scaling
        return latents

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Use Cosmos rectified flow formulation instead of musubi's default FlowMatchDiscreteScheduler.

        Cosmos rectified flow:
        - Sample t ~ logit-normal (or other distribution)
        - Map to discrete timesteps via shift: ts = shift * t / (1 + (shift-1)*t)
        - Get sigma = ts / 1000
        - x_t = noise * sigma + clean * (1 - sigma)
        """
        batch_size = noise.shape[0]

        if timesteps is not None:
            # Use provided timesteps (e.g., from bucketed sampling)
            sample_timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
        else:
            # Sample time using rectified flow
            t_B = self.rectified_flow.sample_train_time(batch_size).to(device=device, dtype=torch.float32)
            sample_timesteps = self.rectified_flow.get_discrete_timestamp(
                t_B, {"device": device, "dtype": torch.float32}
            )

        # Get sigmas from timesteps
        sigmas = self.rectified_flow.get_sigmas(
            sample_timesteps, {"device": device, "dtype": torch.float32}
        )

        # Interpolation: x_t = noise * sigma + clean * (1 - sigma)
        # Note: In Cosmos convention, x_0 is noise, x_1 is clean data
        sigmas_expanded = sigmas.view(batch_size, *([1] * (len(latents.shape) - 1)))
        noisy_model_input = noise * sigmas_expanded + latents * (1 - sigmas_expanded)

        return noisy_model_input, sample_timesteps

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
        model: CosmosDiT = transformer

        # Get Qwen text embeddings from cached batch
        # Cached as varlen_qwen_{dtype} → batch key "qwen", list of [seq_len, 100352] tensors
        context = [t.to(device=accelerator.device, dtype=network_dtype) for t in batch["qwen"]]
        # Stack variable-length tensors into a single batch tensor [B, N, D]
        crossattn_emb = torch.stack(context)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            crossattn_emb.requires_grad_(True)

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        crossattn_emb = crossattn_emb.to(device=accelerator.device, dtype=network_dtype)

        with accelerator.autocast():
            model_pred = model(
                noisy_model_input,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
            )

        # Rectified flow velocity target: v_t = noise - clean_data
        # This matches musubi's convention: target = noise - latents
        target = noise - latents

        return model_pred, target

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """Cache Text Encoder outputs for sample prompts using Qwen2.5-VL."""
        from musubi_tuner.cosmos.text_encoder import CosmosTextEncoder

        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Qwen2.5-VL text encoder
        logger.info(f"loading Qwen text encoder: {self.text_encoder_name}")
        text_encoder = CosmosTextEncoder(
            model_name_or_path=self.text_encoder_name,
            device=str(device),
            dtype=self.dit_dtype,
            max_length=self.text_len,
        )

        sample_prompts_te_outputs = {}
        with torch.no_grad():
            for prompt_dict in prompts:
                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = ""
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                    if p is None:
                        continue
                    if p not in sample_prompts_te_outputs:
                        logger.info(f"cache Text Encoder outputs for prompt: {p}")
                        embeddings = text_encoder.encode([p])
                        sample_prompts_te_outputs[p] = embeddings[0]  # [seq_len, 100352]

        del text_encoder
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["qwen_embeds"] = sample_prompts_te_outputs[p]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_qwen_embeds"] = sample_prompts_te_outputs[p]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    # endregion model specific


def cosmos_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Cosmos Predict 2.5 specific parser setup."""
    parser.add_argument(
        "--cosmos_model_size",
        type=str,
        default="14B",
        choices=["2B", "14B"],
        help="Cosmos model size variant",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Cosmos text encoder model name or path. Uses Qwen2.5-VL architecture. "
        "(default: nvidia/Cosmos-Reason1-7B)",
    )
    parser.add_argument(
        "--vae_cache_cpu",
        action="store_true",
        help="cache features in VAE on CPU",
    )
    parser.add_argument(
        "--cosmos_rf_shift",
        type=int,
        default=5,
        help="Rectified flow shift parameter (default: 5)",
    )
    parser.add_argument(
        "--cosmos_time_distribution",
        type=str,
        default="logitnormal",
        choices=["uniform", "logitnormal"],
        help="Time distribution for rectified flow training (default: logitnormal)",
    )
    parser.add_argument(
        "--cosmos_text_len",
        type=int,
        default=512,
        help="Max text token length for Qwen encoder (default: 512)",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use scaled fp8 for DiT weights (reduces VRAM, requires bf16/fp16 weights)",
    )
    return parser


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = setup_parser_common()
        cosmos_setup_parser(parser)
        args = parser.parse_args()
        read_config_from_file(args, parser)

    cosmos_trainer = CosmosNetworkTrainer()
    cosmos_trainer.train(args)


if __name__ == "__main__":
    main()
