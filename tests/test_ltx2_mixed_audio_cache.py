import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_LTX2
from musubi_tuner.dataset.audio_quota_sampler import split_concat_indices_by_audio
from musubi_tuner.dataset.image_video_dataset import AudioDataset, DatasetGroup
from musubi_tuner.ltx2_cache_latents import _should_cache_audio_dataset_video_latents


def test_mixed_av_audio_dataset_gets_virtual_video_cache():
    audio_dataset = object()

    assert _should_cache_audio_dataset_video_latents(
        audio_only=False,
        audio_video=True,
        audio_datasets=[audio_dataset],
    )


def test_video_only_cache_does_not_create_audio_virtual_video_cache():
    assert not _should_cache_audio_dataset_video_latents(
        audio_only=False,
        audio_video=False,
        audio_datasets=[object()],
    )
    assert not _should_cache_audio_dataset_video_latents(
        audio_only=False,
        audio_video=True,
        audio_datasets=[],
    )


def test_virtual_video_and_audio_cache_reach_audio_sampler(tmp_path):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    save_file(
        {'latents_1x2x2_float32': torch.zeros(4, 1, 2, 2)},
        cache_dir / 'voice_1024x1024_ltx2.safetensors',
    )
    save_file(
        {'audio_latents_10x4x2_float32': torch.zeros(4, 10, 2)},
        cache_dir / 'voice_1024x1024_ltx2_audio.safetensors',
    )
    save_file(
        {'gemma_embeds_float32': torch.zeros(1, 2)},
        cache_dir / 'voice_ltx2_te.safetensors',
    )

    dataset = AudioDataset(
        resolution=(1024, 1024),
        caption_extension='.txt',
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        audio_directory=str(tmp_path),
        cache_directory=str(cache_dir),
        cache_only=True,
        architecture=ARCHITECTURE_LTX2,
    )
    dataset.prepare_for_training()

    audio_indices, non_audio_indices = split_concat_indices_by_audio(DatasetGroup([dataset]))
    assert audio_indices == [0]
    assert non_audio_indices == []
