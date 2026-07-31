"""Conformance oracle for the hash-pinned upstream XM selector.

Set ``XM_REFERENCE_MODEL_UTILS`` to ``model/model_utils.py`` from the pinned
upstream commit to enable this otherwise optional test.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.ltx2_train_network as ltx2_train_network
from musubi_tuner.ltx2_explorative import per_sample_reconstruction_loss
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer


UPSTREAM_COMMIT = "9d06ced61e2d2775a34782eb5830584ae4ef6094"
UPSTREAM_FUNCTION_SHA256 = "7e32b7044e207fbebc31c47822cc8941a2afa3f96c55a768e481de36ca82627c"


def _load_pinned_upstream_xm():
    source_path = os.environ.get("XM_REFERENCE_MODEL_UTILS")
    if not source_path:
        pytest.skip("set XM_REFERENCE_MODEL_UTILS to the pinned upstream XM model/model_utils.py")

    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "xm_chunked_best_of_k")
    function_source = ast.get_source_segment(source, function_node)
    assert function_source is not None
    digest = hashlib.sha256(function_source.encode()).hexdigest()
    assert digest == UPSTREAM_FUNCTION_SHA256, (
        f"upstream xm_chunked_best_of_k does not match pinned commit {UPSTREAM_COMMIT}: {digest}"
    )

    namespace = {"math": math, "torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[function_node], type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["xm_chunked_best_of_k"]


class _FlowHarness:
    training = True
    call_dit = LTX2NetworkTrainer.call_dit
    _call_dit_forward_xm = LTX2NetworkTrainer._call_dit_forward_xm

    def __init__(self, scale: torch.nn.Parameter) -> None:
        self.scale = scale

    def _call_dit_once(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, dtype, **kwargs):
        condition = batch["condition"].view(-1, 1, 1, 1, 1)
        prediction = self.scale * noisy.float() + condition
        target = noise.float() - latents.float()
        return {
            "video_pred": prediction,
            "video_target": target,
            "video_loss_weight": 1.0,
        }, torch.tensor(0.0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_ltx_xm_matches_pinned_upstream_winners_losses_predictions_and_gradients(monkeypatch, device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU conformance case")
    device = torch.device(device_name)
    upstream_xm = _load_pinned_upstream_xm()
    ground_truth = torch.tensor([0.25, -0.75], device=device).view(2, 1, 1, 1, 1)
    condition = torch.tensor([0.1, -0.2], device=device)
    sigma = torch.tensor([0.2, 0.8], device=device).view(2, 1, 1, 1, 1)
    timesteps = sigma.flatten() * 1000.0
    candidates = [
        torch.tensor([2.0, -2.0], device=device).view(2, 1, 1, 1, 1),
        torch.tensor([0.5, 3.0], device=device).view(2, 1, 1, 1, 1),
        torch.tensor([-1.0, -0.5], device=device).view(2, 1, 1, 1, 1),
    ]
    expected_selected_noise = torch.stack((candidates[1][0], candidates[0][1]))

    reference_scale = torch.nn.Parameter(torch.tensor(0.7, device=device))
    reference_replay_noise: list[torch.Tensor] = []

    def reference_forward(noisy, cond):
        return reference_scale * noisy.float() + cond.view(-1, 1, 1, 1, 1)

    def reference_loss_wrapper(model_forward, conditions, gt_samples, learning, rand_inputs, rand_seeds, **kwargs):
        del rand_seeds, kwargs
        noisy = (1.0 - sigma) * gt_samples + sigma * rand_inputs
        prediction = model_forward(noisy, conditions)
        target = rand_inputs - gt_samples
        losses = (prediction - target).square().flatten(1).mean(dim=1)
        if learning:
            reference_replay_noise.append(rand_inputs.detach().clone())
        return losses, prediction

    upstream_candidates = iter(candidates)
    original_randn = torch.randn

    def fixed_upstream_randn(shape, *args, **kwargs):
        if tuple(shape) == tuple(candidates[0].shape):
            return next(upstream_candidates).to(device=kwargs.get("device", "cpu"))
        return original_randn(shape, *args, **kwargs)

    with monkeypatch.context() as upstream_patch:
        upstream_patch.setattr(torch, "randn", fixed_upstream_randn)
        reference_losses, reference_predictions = upstream_xm(
            reference_forward,
            reference_loss_wrapper,
            condition,
            ground_truth,
            best_of_k=3,
            max_chunk_bs_mult=1,
            save_mem_mode=True,
        )
    reference_losses.sum().backward()
    reference_gradient = reference_scale.grad.detach().clone()

    implementation_scale = torch.nn.Parameter(torch.tensor(0.7, device=device))
    harness = _FlowHarness(implementation_scale)
    implementation_candidates = iter(candidates[1:])
    monkeypatch.setattr(
        ltx2_train_network,
        "seeded_randn_like",
        lambda _tensor, _seed: next(implementation_candidates).clone(),
    )
    initial_noisy = (1.0 - sigma) * ground_truth + sigma * candidates[0]
    output, _ = harness.call_dit(
        SimpleNamespace(ltx2_xm_k=3, loss_type="mse", huber_delta=1.0),
        SimpleNamespace(device=device),
        None,
        ground_truth,
        {"condition": condition},
        candidates[0],
        initial_noisy,
        timesteps,
        torch.float32,
    )
    implementation_losses = per_sample_reconstruction_loss(
        output["video_pred"],
        output["video_target"],
        None,
        loss_type="mse",
        huber_delta=1.0,
    )
    implementation_losses.sum().backward()

    assert len(reference_replay_noise) == 1
    torch.testing.assert_close(reference_replay_noise[0], expected_selected_noise)
    torch.testing.assert_close(output["video_target"] + ground_truth, expected_selected_noise)
    torch.testing.assert_close(implementation_losses, reference_losses)
    torch.testing.assert_close(output["video_pred"], reference_predictions)
    torch.testing.assert_close(implementation_scale.grad, reference_gradient)
