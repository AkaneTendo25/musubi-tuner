#!/usr/bin/env python3

"""
Decode precomputed video latents back into videos using the VAE.
This script loads latent files saved during preprocessing and decodes them
back into video clips using the same VAE model.
Basic usage:
    python scripts/decode_latents.py /path/to/latents/dir /path/to/output \
        --model-source /path/to/ltx2.safetensors
"""

from pathlib import Path

import torch
import torchaudio
import torchvision.utils
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar

from ltx_trainer import logger
from ltx_trainer.model_loader import load_audio_vae_decoder, load_video_vae_decoder, load_vocoder
from ltx_trainer.video_utils import save_video

disable_progress_bar()
console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Decode precomputed video latents back into videos using the VAE.",
)


class LatentsDecoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        vae_tiling: bool = False,
        with_audio: bool = False,
    ):
        """Initialize the decoder with model configuration.
        Args:
            model_path: Path to LTX-2 checkpoint (.safetensors)
            device: Device to use for computation
            vae_tiling: Whether to enable VAE tiling for larger video resolutions
            with_audio: Whether to load audio VAE for audio decoding
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.vae = None
        self.audio_vae = None
        self.vocoder = None
        self._load_model(model_path, vae_tiling, with_audio)

    def _load_model(self, model_path: str, vae_tiling: bool, with_audio: bool = False) -> None:
        """Initialize and load the VAE model(s)."""
        with console.status(f"[bold]Loading video VAE decoder from {model_path}...", spinner="dots"):
            self.vae = load_video_vae_decoder(model_path, device=self.device, dtype=torch.bfloat16)

            if vae_tiling:
                self.vae.enable_tiling()

        if with_audio:
            with console.status(f"[bold]Loading audio VAE decoder from {model_path}...", spinner="dots"):
                self.audio_vae = load_audio_vae_decoder(model_path, device=self.device, dtype=torch.bfloat16)

            with console.status(f"[bold]Loading vocoder from {model_path}...", spinner="dots"):
                self.vocoder = load_vocoder(model_path, device=self.device)

    @torch.inference_mode()
    def decode(self, latents_dir: Path, output_dir: Path, seed: int | None = None) -> None:
        """Decode all latent files in the directory recursively.
        Args:
            latents_dir: Directory containing latent files (.pt)
            output_dir: Directory to save decoded videos
            seed: Optional random seed for noise generation
        """
        # Find all .pt files recursively
        latent_files = list(latents_dir.rglob("*.pt"))

        if not latent_files:
            logger.warning(f"No .pt files found in {latents_dir}")
            return

        logger.info(f"Found {len(latent_files):,} latent files to decode")

        # Process files with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Decoding latents", total=len(latent_files))

            for latent_file in latent_files:
                # Calculate relative path to maintain directory structure
                rel_path = latent_file.relative_to(latents_dir)
                output_subdir = output_dir / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    self._process_file(latent_file, output_subdir, seed)
                except Exception as e:
                    logger.error(f"Error processing {latent_file}: {e}")
                    continue

                progress.advance(task)

        logger.info(f"Decoding complete! Videos saved to {output_dir}")

    def _process_file(self, latent_file: Path, output_dir: Path, seed: int | None) -> None:
        """Process a single latent file."""
        # Load the latent data
        data = torch.load(latent_file, map_location=self.device, weights_only=False)

        # Get latents - handle both old patchified [seq_len, C] and new [C, F, H, W] formats
        latents = data["latents"]
        num_frames = data["num_frames"]
        height = data["height"]
        width = data["width"]

        # Check if latents need reshaping (old patchified format)
        if latents.dim() == 2:
            # Old format: [seq_len, C] -> reshape to [C, F, H, W]
            _seq_len, channels = latents.shape
            latents = latents.reshape(num_frames, height, width, channels)
            latents = latents.permute(3, 0, 1, 2)  # [F, H, W, C] -> [C, F, H, W]

        # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
        latents = latents.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)

        # Create generator only if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Decode the video (VAE decoder uses forward/call, not decode method)
        video = self.vae(latents)  # [B, C, F, H, W]

        # Convert to [F, C, H, W] format and normalize to [0, 1]
        video = video[0]  # Remove batch dimension -> [C, F, H, W]
        video = video.permute(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]
        video = (video + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        video = video.clamp(0, 1)

        # Determine output format and save
        is_image = video.shape[0] == 1
        if is_image:
            # Save as PNG for single frame
            output_path = output_dir / f"{latent_file.stem}.png"
            torchvision.utils.save_image(
                video[0],  # [C, H, W] in [0, 1]
                str(output_path),
            )
        else:
            # Save as MP4 for video using PyAV-based save_video
            output_path = output_dir / f"{latent_file.stem}.mp4"
            fps = data.get("fps", 24)  # Use stored FPS or default to 24
            save_video(
                video_tensor=video,  # [F, C, H, W] in [0, 1]
                output_path=output_path,
                fps=fps,
            )

    @torch.inference_mode()
    def decode_audio(self, latents_dir: Path, output_dir: Path) -> None:
        """Decode all audio latent files in the directory recursively.
        Args:
            latents_dir: Directory containing audio latent files (.pt)
            output_dir: Directory to save decoded audio files
        """
        # Check if audio VAE is loaded
        if self.audio_vae is None or self.vocoder is None:
            logger.warning("Audio VAE or vocoder not loaded. Skipping audio decoding.")
            return

        # Find all .pt files recursively
        latent_files = list(latents_dir.rglob("*.pt"))

        if not latent_files:
            logger.warning(f"No .pt files found in {latents_dir}")
            return

        logger.info(f"Found {len(latent_files):,} audio latent files to decode")

        # Process files with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Decoding audio latents", total=len(latent_files))

            for latent_file in latent_files:
                # Calculate relative path to maintain directory structure
                rel_path = latent_file.relative_to(latents_dir)
                output_subdir = output_dir / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    self._process_audio_file(latent_file, output_subdir)
                except Exception as e:
                    logger.error(f"Error processing audio {latent_file}: {e}")
                    continue

                progress.advance(task)

        logger.info(f"Audio decoding complete! Audio files saved to {output_dir}")

    def _process_audio_file(self, latent_file: Path, output_dir: Path) -> None:
        """Process a single audio latent file."""
        # Load the latent data
        data = torch.load(latent_file, map_location=self.device, weights_only=False)

        latents = data["latents"].to(device=self.device, dtype=torch.float32)
        num_time_steps = data["num_time_steps"]
        freq_bins = data["frequency_bins"]

        # Handle both old patchified [seq_len, C] and new [C, T, F] formats
        if latents.dim() == 2:
            # Old format: [seq_len, channels] where seq_len = time * freq
            # Reshape to [C, T, F]
            latents = latents.reshape(num_time_steps, freq_bins, -1)  # [T, F, C]
            latents = latents.permute(2, 0, 1)  # [T, F, C] -> [C, T, F]

        # Add batch dimension: [C, T, F] -> [1, C, T, F]
        latents = latents.unsqueeze(0)

        # Set correct dtype for audio VAE
        latents = latents.to(dtype=torch.bfloat16)

        # Decode audio using audio VAE decoder (produces mel spectrogram)
        mel_spectrogram = self.audio_vae(latents)

        # Convert mel spectrogram to waveform using vocoder
        waveform = self.vocoder(mel_spectrogram)

        # Save as WAV
        output_path = output_dir / f"{latent_file.stem}.wav"
        sample_rate = self.vocoder.output_sample_rate
        torchaudio.save(str(output_path), waveform[0].cpu(), sample_rate)


@app.command()
def main(
    latents_dir: str = typer.Argument(
        ...,
        help="Directory containing the precomputed latent files (searched recursively)",
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directory to save the decoded videos (maintains same folder hierarchy as input)",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    seed: int | None = typer.Option(
        default=None,
        help="Random seed for noise generation during decoding",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Also decode audio latents (requires audio_latents directory)",
    ),
    audio_latents_dir: str | None = typer.Option(
        default=None,
        help="Directory containing audio latent files (defaults to 'audio_latents' sibling of latents_dir)",
    ),
) -> None:
    """Decode precomputed video latents back into videos using the VAE.
    This script recursively searches for .pt latent files in the input directory
    and decodes them to videos, maintaining the same folder hierarchy in the output.
    Examples:
        # Basic usage
        python scripts/decode_latents.py /path/to/latents /path/to/videos \\
            --model-path /path/to/ltx2.safetensors
        # With VAE tiling for large videos
        python scripts/decode_latents.py /path/to/latents /path/to/videos \\
            --model-path /path/to/ltx2.safetensors --vae-tiling
        # With audio decoding
        python scripts/decode_latents.py /path/to/latents /path/to/videos \\
            --model-path /path/to/ltx2.safetensors --with-audio
    """
    latents_path = Path(latents_dir)
    output_path = Path(output_dir)

    if not latents_path.exists() or not latents_path.is_dir():
        raise typer.BadParameter(f"Latents directory does not exist: {latents_path}")

    decoder = LatentsDecoder(
        model_path=model_path,
        device=device,
        vae_tiling=vae_tiling,
        with_audio=with_audio,
    )
    decoder.decode(latents_path, output_path, seed=seed)

    # Decode audio if requested
    if with_audio:
        audio_path = Path(audio_latents_dir) if audio_latents_dir else latents_path.parent / "audio_latents"

        if audio_path.exists():
            audio_output_path = output_path.parent / "decoded_audio"
            decoder.decode_audio(audio_path, audio_output_path)
        else:
            logger.warning(f"Audio latents directory not found: {audio_path}")


if __name__ == "__main__":
    app()
