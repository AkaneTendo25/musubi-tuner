"""Shared rollout generation for LTX-2 RL — used by both Phase A (offline cache) and
Phase B online mode, so the two paths generate identically.

``build_generate_fn`` returns a ``generate_fn(prompt, seeds) -> list[sample dict]`` that drives
``do_inference(return_latents=True)`` for each seed. The transformer forward is wrapped in
``accelerator.autocast()`` so it is dtype-correct whether the LoRA adapter is bf16 (offline, cast
for generation) or fp32 (online, kept for training) — autocast casts the fp32 weight to bf16 for the
matmul, giving identical results either way.

AV mode (``--ltx2_mode av``): the text embed produced by ``_encode_prompt_text`` is the
concatenated ``[video | audio]`` context (2x caption_channels for LTX-2.0, or
video_dim+audio_dim for LTX-2.3); ``do_inference`` already runs the AV denoise loop, so this
module additionally captures the clean audio latent (``audio_x0``), the audio half of the context
(``a_ctx``), and the (shared) audio attention mask so Phase B can train the audio NFT branch.
The plain video path is unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import torch


def prepare_sampling_args(args) -> None:
    """Fill guidance/scale args that the --sample_prompts parsing path normally provides."""
    for attr, default in (("av_bimodal_scale", 1.0), ("rescale_scale", 0.0)):
        if getattr(args, attr, None) is None:
            setattr(args, attr, default)


def make_sigma_schedule(num_steps: int) -> torch.Tensor:
    """FALLBACK sigma schedule only. The real schedule each rollout was denoised with is returned by
    ``do_inference(return_latents=True)`` (5th element) and cached per sample; this linear linspace is
    used only if that value is absent (older do_inference signature)."""
    return torch.linspace(0.999, 1e-3, int(num_steps), dtype=torch.float32)


def _write_video_mp4(video, path: str, fps: int = 16) -> None:
    """Decoded video ``[1,C,T,H,W]``/``[C,T,H,W]`` in [0,1] -> mp4 at ``path`` (``[T,H,W,C]`` uint8)."""
    from torchvision.io import write_video

    v = video
    if v.dim() == 5:
        v = v[0]
    v = (v.detach().cpu().float().clamp(0, 1) * 255.0).to(torch.uint8)  # [C,T,H,W]
    write_video(path, v.permute(1, 2, 3, 0).contiguous(), fps=fps)  # [T,H,W,C]


def _decode_audio_latent_to_wav(audio_latent, checkpoint_path: str, path: str) -> tuple:
    """Decode an audio latent ``[1,C,T,F]`` to a wav via the ``ltx2_audio_preview`` subprocess.

    The LTX-2 audio decoder + vocoder run in a SEPARATE process, so the audio VAE never co-resides
    with the DiT on the GPU (the rollout's VRAM footprint is unchanged). Returns ``(ok, detail)``
    where ``detail`` carries the child's stderr/stdout tail on failure.
    """
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix="_ltx2_rl_audio.pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save({"latents": audio_latent.detach().cpu()}, tmp_path)
        cmd = [
            sys.executable,
            "-m",
            "musubi_tuner.ltx2_audio_preview",
            "--checkpoint",
            checkpoint_path,
            "--input",
            tmp_path,
            "--output",
            path,
            "--device",
            "auto",
            "--dtype",
            "fp32",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return True, ""
        detail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
        return False, detail[-2000:]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _read_wav_tensor(path: str):
    """Read a wav into ``([channels, samples] float32 in [-1,1], sample_rate)``."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    a = a.reshape(-1, ch).T if ch > 1 else a.reshape(1, -1)
    return torch.from_numpy(a.copy()), sr


def build_generate_fn(
    net_trainer,
    args,
    accelerator,
    transformer,
    vae,
    dit_dtype: torch.dtype,
    device,
    *,
    num_steps: int,
    needs_media: bool,
    sigma_schedule: torch.Tensor,
    te_dtype,
    prompt_embeds: Dict[str, Any] = None,
    media_needs: frozenset = frozenset(),
) -> Callable[[str, List[int]], List[Dict[str, Any]]]:
    """Build the rollout generator.

    If ``prompt_embeds`` is given ({prompt: (embed, mask)} precomputed on CPU), it is used instead
    of encoding inline — so the text encoder (Gemma, ~24 GB) can be freed before this runs and never
    co-resides with the DiT (the offline 24 GB path). Otherwise prompts are encoded inline (online).
    """

    # AV mode is gated entirely behind --ltx2_mode av (mirrors handle_model_specific_args, which
    # sets _ltx_mode / _audio_video). The narrow video path below is 100% unchanged when not AV.
    is_av = getattr(net_trainer, "_ltx_mode", getattr(args, "ltx2_mode", "video")) == "av"
    # Authentic-PPO: when on, do_inference samples with the SDE sampler and fills a per-rollout
    # trajectory sink, which we stack and cache so Phase B can score the exact per-step log-prob ratio.
    _rl_sde = bool(getattr(args, "rl_sde_sampler", False))
    # The model that knows how to split the concatenated AV context (LTX2Wrapper.model). Resolved
    # once here so the per-sample loop does not unwrap repeatedly.
    _av_split_model = None
    if is_av:
        from musubi_tuner.networks.lora_ltx2 import _split_av_context

        _unwrapped = accelerator.unwrap_model(transformer) if hasattr(accelerator, "unwrap_model") else transformer
        _av_split_model = getattr(_unwrapped, "model", _unwrapped)

    # Which decoded media must be MATERIALIZED ON DISK for file-based rewards (av_align/av_desync
    # need video_file/audio_file; clap/audiobox need an audio_waveform tensor). Pixel-tensor video
    # rewards (HPSv3, sharpness, ...) read sample["video"] directly and need none of this.
    want_video_file = "video_file" in media_needs
    want_audio_file = is_av and "audio_file" in media_needs
    want_audio_waveform = is_av and "audio_waveform" in media_needs
    _media_dir = None
    if want_video_file or want_audio_file or want_audio_waveform:
        import atexit
        import shutil
        import tempfile

        _media_dir = tempfile.mkdtemp(prefix="ltx2_rl_media_")
        atexit.register(lambda d=_media_dir: shutil.rmtree(d, ignore_errors=True))

    def generate_fn(prompt: str, seeds: List[int]) -> List[Dict[str, Any]]:
        if prompt_embeds is not None and prompt in prompt_embeds:
            embed, mask = prompt_embeds[prompt]
        else:
            embed, mask = net_trainer._encode_prompt_text(accelerator, prompt, te_dtype)
        samples: List[Dict[str, Any]] = []
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            sample_parameter = {"prompt": prompt, "prompt_embeds": embed, "prompt_attention_mask": mask}
            traj_sink: List[Dict[str, Any]] = [] if _rl_sde else None
            with torch.no_grad(), accelerator.autocast():
                out = net_trainer.do_inference(
                    accelerator,
                    args,
                    sample_parameter,
                    vae,
                    dit_dtype,
                    transformer,
                    discrete_flow_shift=getattr(args, "discrete_flow_shift", 1.0),
                    sample_steps=num_steps,
                    width=args.sample_width,
                    height=args.sample_height,
                    frame_count=args.sample_frames,
                    generator=generator,
                    do_classifier_free_guidance=getattr(args, "sample_cfg", 1.0) > 1.0,
                    guidance_scale=getattr(args, "sample_cfg", 1.0),
                    cfg_scale=None,
                    decode_video=needs_media,
                    return_latents=True,
                    # AV mode: trigger audio-latent creation so do_inference runs the AV denoise
                    # loop (the wrapper splits the [video|audio] context and returns audio_x0).
                    # Without this, do_inference defaults to the video-only path and the 6144 AV
                    # concat reaches the transformer unsplit -> "Context 6144 vs 4096" crash.
                    # audio_decoder/vocoder stay None: audio_x0 (the NFT target) is derived from the
                    # velocity prediction directly; only the decoded waveform needs the vocoder.
                    enable_audio_preview=is_av,
                    rl_trajectory_sink=traj_sink,
                )
            # do_inference(return_latents=True) returns the REAL sigma schedule as a 5th element; fall
            # back to the passed schedule only if an older signature returns the 4-tuple.
            if len(out) == 5:
                _video, _audio, video_x0, audio_x0, real_sigmas = out
            else:
                _video, _audio, video_x0, audio_x0 = out
                real_sigmas = None
            sample = {
                "seed": int(seed),
                "prompt": prompt,
                "video_x0": video_x0.squeeze(0).to("cpu", torch.float32),
                # In AV mode ``embed`` is the concatenated [video|audio] context (2x caption_channels,
                # or video_dim+audio_dim for LTX-2.3). The wrapper splits it internally via
                # ``_split_av_context``; Phase B passes this SAME concat as ``context`` for the AV
                # forward, so v_ctx carries the full AV context (the video path stays unchanged).
                "v_ctx": embed.to("cpu", torch.float32),
                "v_mask": (mask.to("cpu") if mask is not None else torch.ones(embed.shape[0], dtype=torch.bool)),
                "sigmas": (real_sigmas.to("cpu", torch.float32) if real_sigmas is not None else sigma_schedule.clone()),
            }
            if traj_sink:
                # Stack the per-step SDE trajectory for authentic PPO. x_t/x_next are [1,C,F,H,W] from
                # generation -> squeeze to [C,F,H,W] then stack over the S steps. Phase B subsamples a
                # step per rollout and recomputes the exact per-step ratio at the cached action x_next.
                sample["traj_x_t"] = torch.stack([e["x_t"].squeeze(0) for e in traj_sink], dim=0)  # [S,C,F,H,W]
                sample["traj_x_next"] = torch.stack([e["x_next"].squeeze(0) for e in traj_sink], dim=0)  # [S,C,F,H,W]
                sample["traj_x0_gen"] = torch.stack([e["x0_gen"].squeeze(0) for e in traj_sink], dim=0)  # [S,C,F,H,W]
                sample["traj_sigma"] = torch.tensor([e["sigma"] for e in traj_sink], dtype=torch.float32)  # [S]
                sample["traj_sigma_next"] = torch.tensor([e["sigma_next"] for e in traj_sink], dtype=torch.float32)  # [S]
                if "audio_x_t" in traj_sink[0]:
                    # AV authentic PPO: the audio actions share the (video) sigma schedule -> reuse
                    # traj_sigma/traj_sigma_next; only the audio latents differ.
                    sample["traj_audio_x_t"] = torch.stack([e["audio_x_t"].squeeze(0) for e in traj_sink], dim=0)
                    sample["traj_audio_x_next"] = torch.stack([e["audio_x_next"].squeeze(0) for e in traj_sink], dim=0)
                    sample["traj_audio_x0_gen"] = torch.stack([e["audio_x0_gen"].squeeze(0) for e in traj_sink], dim=0)
            if is_av:
                # Every AV rollout MUST carry an audio latent. A missing one would be silently dropped
                # by _stack_sample_tensors (now raises) and corrupt the group's modality set.
                if audio_x0 is None:
                    raise RuntimeError(
                        f"AV rollout (seed {seed}) produced audio_x0=None: do_inference returned no audio "
                        "latent in --ltx2_mode av (enable_audio_preview is set). Cannot build the AV cache."
                    )
                # audio_x0 = clean generated audio latent [1, C, T, F]; squeeze batch -> [C, T, F].
                # a_ctx = the audio half of the AV context (explicit; the full concat lives in v_ctx).
                # a_mask mirrors the (shared) text mask — AV uses one text attention mask for both
                # modalities. Stored so Phase B can re-noise + forward the audio NFT branch.
                sample["audio_x0"] = audio_x0.squeeze(0).to("cpu", torch.float32)
                _, audio_ctx = _split_av_context(_av_split_model, embed)
                sample["a_ctx"] = audio_ctx.to("cpu", torch.float32)
                sample["a_mask"] = sample["v_mask"].clone()
            if needs_media:
                # decoded pixel video [1,C,T,H,W] in [0,1]; pixel-tensor rewards (HPSv3, sharpness,
                # motion_physics, ...) consume frames directly from this tensor.
                sample["video"] = _video.to("cpu", torch.float32) if _video is not None else None
                # File-based / waveform rewards: materialize decoded media on disk. Video is already
                # decoded above; audio is decoded from its latent in a SUBPROCESS so the audio VAE
                # never co-resides with the DiT. These scoring-only keys are excluded from the cache.
                if want_video_file and sample["video"] is not None:
                    vpath = os.path.join(_media_dir, f"rollout_{seed}.mp4")
                    _write_video_mp4(sample["video"], vpath, fps=int(getattr(args, "sample_fps", 16) or 16))
                    sample["video_file"] = vpath
                if (want_audio_file or want_audio_waveform) and is_av and audio_x0 is not None:
                    apath = os.path.join(_media_dir, f"rollout_{seed}.wav")
                    _ok, _detail = _decode_audio_latent_to_wav(audio_x0, args.ltx2_checkpoint, apath)
                    if not _ok:
                        raise RuntimeError(
                            f"RL rollout (seed {seed}): audio decode subprocess failed; an audio/sync reward "
                            f"cannot score. Verify the LTX-2 checkpoint provides an audio decoder + vocoder.\n{_detail}"
                        )
                    if want_audio_file:
                        sample["audio_file"] = apath
                    if want_audio_waveform:
                        sample["audio_waveform"], _sr = _read_wav_tensor(apath)
            samples.append(sample)
        return samples

    return generate_fn
