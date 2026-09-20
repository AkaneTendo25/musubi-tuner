import torch

from musubi_tuner.modules.fp8_optimization_utils import _dequant_cache_lookup, _dequant_cache_store


def test_dequant_cache_does_not_reuse_trainable_sources():
    source = torch.tensor([2.0])
    device = source.device
    _dequant_cache_store((source,), source.dtype, device, source * 2)
    assert _dequant_cache_lookup((source,), source.dtype, device) is not None

    source.requires_grad_(True)
    assert _dequant_cache_lookup((source,), source.dtype, device) is None
    result = source * 3
    _dequant_cache_store((source,), source.dtype, device, result)
    assert _dequant_cache_lookup((source,), source.dtype, device) is None

    source.requires_grad_(False)
    _dequant_cache_store((source,), source.dtype, device, source * 4)
    assert _dequant_cache_lookup((source,), source.dtype, device).item() == 8.0
