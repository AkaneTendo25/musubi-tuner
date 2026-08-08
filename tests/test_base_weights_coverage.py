"""Guard against a `--base_weights` file merging into nothing.

`create_network_from_weights` builds its module list from key names alone. A file
written in a convention the loader does not recognise therefore produces a network
of zero modules; `merge_to` iterates that empty list and still logs "weights are
merged", so a full training run proceeds against an unmodified base while every
message says the adapter was applied. This happened for real with an adapter
published in peft's `diffusion_model.<dotted>.lora_A` naming: 0 of 200 modules
merged, no error, and the run was an unlabelled duplicate of the no-adapter run.

These tests pin the three cases: nothing matched (error), everything matched
(silent), part matched (warning, merge proceeds).
"""

import pytest
import torch
from torch import nn

from musubi_tuner.networks.lora import create_network_from_weights
from musubi_tuner.training.trainer_base import check_base_weights_coverage


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


TARGET = ["Block"]


def native_sd(*module_names, rank=2):
    sd = {}
    for name in module_names:
        sd[f"{name}.lora_down.weight"] = torch.zeros(rank, 8)
        sd[f"{name}.lora_up.weight"] = torch.zeros(8, rank)
        sd[f"{name}.alpha"] = torch.tensor(float(rank))
    return sd


def network_for(sd):
    return create_network_from_weights(TARGET, 1.0, sd, unet=Model(), for_inference=True)


def test_unrecognised_convention_raises_instead_of_merging_nothing():
    # peft naming: the loader splits on the first dot, sees "diffusion_model", and
    # matches no module -- the exact shape of the real failure.
    sd = {
        "diffusion_model.blocks.0.proj.lora_A.weight": torch.zeros(2, 8),
        "diffusion_model.blocks.0.proj.lora_B.weight": torch.zeros(8, 2),
    }
    network = network_for(sd)
    assert not network.unet_loras, "precondition: this convention yields no modules"

    with pytest.raises(ValueError, match="merged into nothing"):
        check_base_weights_coverage("adapter.safetensors", sd, network)


def test_full_match_passes_without_warning(caplog):
    sd = native_sd("lora_unet_blocks_0_proj", "lora_unet_blocks_1_proj")
    network = network_for(sd)
    assert len(network.unet_loras) == 2

    with caplog.at_level("WARNING"):
        check_base_weights_coverage("adapter.safetensors", sd, network)
    assert not caplog.records


def test_partial_match_warns_but_still_merges(caplog):
    # An adapter may legitimately target modules this project does not wrap; those
    # stay unmerged and the run continues, but the count is worth naming because a
    # convention mismatch affecting part of a file looks exactly like this.
    sd = native_sd("lora_unet_blocks_0_proj")
    sd.update(native_sd("lora_unet_token_refiner_blocks_0_proj"))
    network = network_for(sd)
    assert len(network.unet_loras) == 1

    with caplog.at_level("WARNING"):
        check_base_weights_coverage("adapter.safetensors", sd, network)
    assert len(caplog.records) == 1
    assert "3 of 6 tensors have no matching module" in caplog.records[0].message
