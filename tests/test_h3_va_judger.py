from __future__ import annotations

import itertools

import pytest
import torch

from musubi_tuner.minimax_h3.va_judger import (
    SCORE_DIMENSIONS,
    VAJudgerClient,
    VAJudgerError,
    dimension_normalized_advantages,
    nft_mixture_loss,
    reward_components,
)


def _result(first, second):
    return {
        "dimension_scores": {dimension: {"1": first[index], "2": second[index]} for index, dimension in enumerate(SCORE_DIMENSIONS)}
    }


def test_score_group_scores_both_orientations_and_cancels_position_bias():
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        results = []
        for pair in payload["pairs"]:
            # The fake judge adds one point to video 1 and subtracts one from
            # video 2. Scoring both orientations must remove that pure bias.
            first = [base + pair["sample_index_1"] + 1 for base in range(3, 8)]
            second = [base + pair["sample_index_2"] - 1 for base in range(3, 8)]
            results.append({"id": pair["id"], **_result(first, second)})
        return {"status": "success", "results": results}

    scores = VAJudgerClient("http://reward:8100/predict", pair_batch_size=2, transport=transport).score_group(
        "a prompt", ["a.mp4", "b.mp4", "c.mp4"]
    )

    assert [len(call[1]["pairs"]) for call in calls] == [2, 2, 2]
    assert all(call[0] == "http://reward:8100/predict" for call in calls)
    sent_pairs = list(itertools.chain.from_iterable(call[1]["pairs"] for call in calls))
    assert [(pair["sample_index_1"], pair["sample_index_2"]) for pair in sent_pairs] == [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ]
    assert [pair["pair_index_in_group"] for pair in sent_pairs] == list(range(6))
    assert [pair["id"] for pair in sent_pairs] == [
        f"group0000_pair{index:04d}_{first}_{second}"
        for index, (first, second) in enumerate([(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)])
    ]
    assert all(pair["group_pair_count"] == 6 and pair["group_size"] == 3 for pair in sent_pairs)
    torch.testing.assert_close(
        scores,
        torch.tensor(
            [
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "response,match",
    [
        ({"status": "error", "results": []}, "did not report success"),
        ({"status": "success", "results": []}, "0 results for 1 pairs"),
        ({"status": "success", "results": [{}]}, "missing dimension_scores"),
        (
            {
                "status": "success",
                "results": [_result([11, 5, 5, 5, 5], [5, 5, 5, 5, 5])],
            },
            "outside",
        ),
        (
            {
                "status": "success",
                "results": [_result([0, 5, 5, 5, 5], [5, 5, 5, 5, 5])],
            },
            "outside",
        ),
        (
            {
                "status": "success",
                "results": [_result([float("nan"), 5, 5, 5, 5], [5, 5, 5, 5, 5])],
            },
            "outside",
        ),
    ],
)
def test_score_group_rejects_failed_or_malformed_rewards(response, match):
    def transport(_url, payload, _timeout):
        if isinstance(response.get("results"), list):
            for pair, result in zip(payload["pairs"], response["results"]):
                if isinstance(result, dict):
                    result["id"] = pair["id"]
        return response

    client = VAJudgerClient("http://reward", pair_batch_size=1, transport=transport)
    with pytest.raises(VAJudgerError, match=match):
        client.score_group("prompt", ["a.mp4", "b.mp4"])


def test_dimension_normalization_prevents_scale_dominance_and_returns_shared_scalar():
    scores = torch.tensor(
        [
            [1.0, 100.0, 4.0],
            [2.0, 200.0, 4.0],
            [3.0, 300.0, 4.0],
        ]
    )
    advantages = dimension_normalized_advantages(scores, eps=1e-8)
    one_dimension = dimension_normalized_advantages(scores[:, :1], eps=1e-8)

    # The first two dimensions encode the same ranking at different scales;
    # the constant third dimension contributes zero before the dimension mean.
    torch.testing.assert_close(advantages, one_dimension * (2.0 / 3.0))
    assert advantages.shape == (3,)
    assert float(advantages.sum()) == pytest.approx(0.0, abs=1e-6)


def test_reward_components_match_released_overall_audio_video_mapping():
    raw = torch.tensor([[3.0, 6.0, 8.0, 4.0, 9.0], [9.0, 9.0, 2.0, 7.0, 6.0]])
    torch.testing.assert_close(
        reward_components(raw),
        torch.tensor([[0.6, 0.8, 0.4], [0.8, 0.2, 0.7]]),
    )


def test_score_group_rejects_reordered_results_by_id():
    def transport(_url, payload, _timeout):
        first, second = payload["pairs"]
        return {
            "status": "success",
            "results": [
                {"id": second["id"], **_result([5] * 5, [5] * 5)},
                {"id": first["id"], **_result([5] * 5, [5] * 5)},
            ],
        }

    client = VAJudgerClient("http://reward", pair_batch_size=2, transport=transport)
    with pytest.raises(VAJudgerError, match="order/identity mismatch"):
        client.score_group("prompt", ["a.mp4", "b.mp4", "c.mp4"])


def _nft_gradient(advantage: float) -> float:
    pred = torch.tensor([[1.0]], requires_grad=True)
    loss = nft_mixture_loss(
        pred,
        old_x0=torch.tensor([[1.0]], requires_grad=True),
        reference_x0=torch.tensor([[1.0]], requires_grad=True),
        target_x0=torch.tensor([[0.0]], requires_grad=True),
        advantage=torch.tensor([advantage], requires_grad=True),
        beta_mix=1.0,
        kl_beta=0.0,
        advantage_clip=5.0,
    )
    loss.backward()
    return float(pred.grad), loss


def test_nft_positive_and_negative_mixture_have_opposite_gradient_directions():
    positive_gradient, positive_loss = _nft_gradient(5.0)
    negative_gradient, negative_loss = _nft_gradient(-5.0)

    # Gradient descent follows the preferred positive extrapolation toward x0=0,
    # while a negative advantage pushes the current policy away from that target.
    assert positive_gradient > 0
    assert negative_gradient < 0
    torch.testing.assert_close(positive_loss, negative_loss)


def test_nft_h3_x0_reconstruction_uses_plus_sigma_velocity_sign():
    velocity = torch.tensor([[1.0]], requires_grad=True)
    xt = torch.tensor([[0.5]])
    sigma = torch.tensor([[0.5]])
    pred_x0 = xt + sigma * velocity
    loss = nft_mixture_loss(
        pred_x0,
        old_x0=torch.ones_like(pred_x0),
        reference_x0=torch.ones_like(pred_x0),
        target_x0=torch.zeros_like(pred_x0),
        advantage=torch.tensor([5.0]),
        beta_mix=1.0,
        kl_beta=0.0,
    )
    loss.backward()

    assert float(velocity.grad) > 0  # a descent step lowers v and therefore lowers x0


def test_nft_detaches_all_targets_and_supports_a_mask():
    pred = torch.tensor([[1.0, 100.0]], requires_grad=True)
    old = torch.tensor([[1.0, 100.0]], requires_grad=True)
    reference = torch.tensor([[1.0, 100.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0]], requires_grad=True)
    advantage = torch.tensor([5.0], requires_grad=True)
    loss = nft_mixture_loss(
        pred,
        old,
        reference,
        target,
        advantage,
        beta_mix=1.0,
        kl_beta=0.0,
        element_mask=torch.tensor([[1.0, 0.0]]),
    )
    loss.backward()

    assert pred.grad is not None and float(pred.grad[0, 0]) > 0
    assert float(pred.grad[0, 1]) == 0.0
    assert old.grad is None and reference.grad is None and target.grad is None and advantage.grad is None


def test_nft_matches_source_formula_including_l1_normalizer_gradient():
    pred = torch.tensor([[0.5, 2.0]], requires_grad=True)
    old = torch.tensor([[1.0, 1.5]])
    target = torch.tensor([[0.0, 0.25]])
    advantage = torch.tensor([1.5])
    actual = nft_mixture_loss(
        pred,
        old,
        reference_x0=torch.zeros_like(pred),
        target_x0=target,
        advantage=advantage,
        beta_mix=0.7,
        kl_beta=0.0,
        advantage_clip=5.0,
    )
    actual_gradient = torch.autograd.grad(actual, pred, retain_graph=True)[0]

    positive = 0.7 * pred + 0.3 * old
    negative = 1.7 * old - 0.7 * pred
    wp = (positive.double() - target.double()).abs().mean(dim=1).clamp_min(1e-5)
    wn = (negative.double() - target.double()).abs().mean(dim=1).clamp_min(1e-5)
    reward_mix = ((advantage / 5.0) / 2.0 + 0.5).clamp(0.0, 1.0)
    expected = (
        (
            reward_mix * (positive - target).square().mean(dim=1) / wp
            + (1.0 - reward_mix) * (negative - target).square().mean(dim=1) / wn
        ).mean()
        / 0.7
        * 5.0
    )
    expected_gradient = torch.autograd.grad(expected, pred)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
