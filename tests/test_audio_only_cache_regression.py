from argparse import Namespace

from musubi_tuner import cache_latents
from musubi_tuner.dataset.image_video_dataset import ItemInfo


def test_common_latent_cache_pipeline_accepts_item_without_visual_content(tmp_path):
    item = ItemInfo(
        "tone.wav",
        "a clean tone",
        (832, 480),
        (832, 480, 124),
        frame_count=124,
        latent_cache_path=str(tmp_path / "tone.safetensors"),
    )
    assert item.content is None

    class AudioOnlyDataset:
        def retrieve_latent_cache_batches(self, num_workers):
            assert num_workers == 1
            yield (832, 480, 124), [item]

        def get_all_latent_cache_files(self):
            return []

    encoded = []
    cache_latents.encode_datasets(
        [AudioOnlyDataset()],
        lambda batch: encoded.extend(batch),
        Namespace(num_workers=1, skip_existing=False, batch_size=1, keep_cache=True),
    )

    assert encoded == [item]
    assert item.content is None
