from __future__ import annotations

import json
import torch

from scripts.train_state8_profile_ownership_repair import (
    PROFILE_PROJECTION_DIM,
    build_profile_ownership_head,
    freeze_profile_ownership_head,
    masked_profile_loss,
    masked_profile_panel_loss,
    profile_optimizer_groups,
    profile_state_features,
    profile_kc_support,
    profile_target_logit_bias,
    row_targets,
    row_target_panel,
    stable_projection,
)


def test_profile_ownership_head_matches_float32_feature_contract() -> None:
    head = build_profile_ownership_head(64, 30)
    output = head(torch.zeros(2, 64 * PROFILE_PROJECTION_DIM, dtype=torch.float32))
    assert output.shape == (2, 30)
    assert output.dtype == torch.float32


def test_profile_ownership_head_is_identical_across_rank_local_rngs() -> None:
    torch.manual_seed(11)
    left = build_profile_ownership_head(3, 5, projection_dim=7)
    torch.manual_seed(29)
    right = build_profile_ownership_head(3, 5, projection_dim=7)
    torch.testing.assert_close(left.weight, right.weight, rtol=0, atol=0)
    torch.testing.assert_close(left.bias, right.bias, rtol=0, atol=0)


def test_profile_state_features_preserve_layer_axis() -> None:
    state = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    projection = stable_projection(5, 2, 17)
    observed = profile_state_features(state, projection)
    expected = torch.einsum("bld,dp->blp", state.mean(dim=2), projection).flatten(1)
    torch.testing.assert_close(observed, expected)
    assert observed.shape == (2, 6)


def test_profile_state_features_can_use_stable_layernorm() -> None:
    state = torch.randn(3, 2, 4, 5)
    observed = profile_state_features(
        state, stable_projection(5, 3, 19), normalization="layernorm"
    )
    torch.testing.assert_close(observed.mean(dim=1), torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        observed.square().mean(dim=1), torch.ones(3), atol=2e-3, rtol=0
    )


def test_profile_target_bias_uses_only_eligible_training_targets(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = []
    for value in (0.2, 0.8):
        rows.append({
            "task": "read_student_profile",
            "future_smoothed_rates": [value] * 30,
            "future_eligible_mask": [True] * 30,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    bias = profile_target_logit_bias(path)
    torch.testing.assert_close(bias, torch.zeros(30), atol=1e-6, rtol=0)


def test_profile_target_bias_shrinks_sparse_and_unsupported_kcs_to_train_global(
    tmp_path,
) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({
        "task": "read_student_profile",
        "future_smoothed_rates": [0.8, None, 0.2],
        "future_eligible_mask": [True, False, True],
    }) + "\n")
    bias = profile_target_logit_bias(path, output_dim=3, prior_strength=8)
    expected_left = (0.8 + 8 * 0.5) / 9
    expected_right = (0.2 + 8 * 0.5) / 9
    torch.testing.assert_close(
        bias,
        torch.tensor([
            torch.logit(torch.tensor(expected_left)),
            0.0,
            torch.logit(torch.tensor(expected_right)),
        ]),
        atol=1e-6,
        rtol=0,
    )


def test_profile_kc_support_uses_only_profile_training_rows(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"task": "read_student_profile", "future_eligible_mask": [True, False, True]},
        {"task": "read_student_profile", "future_eligible_mask": [False, False, True]},
        {"task": "predict_correctness", "future_eligible_mask": [True, True, True]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    torch.testing.assert_close(profile_kc_support(path, 3), torch.tensor([1.0, 0.0, 2.0]))


def test_profile_kc_support_accepts_raw_profile_manifest_rows(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"future_eligible_mask": [True, False]}) + "\n")
    torch.testing.assert_close(profile_kc_support(path, 2), torch.tensor([1.0, 0.0]))


def test_profile_target_bias_rejects_manifest_without_any_training_support(
    tmp_path,
) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({
        "task": "read_student_profile",
        "future_smoothed_rates": [None, None],
        "future_eligible_mask": [False, False],
    }) + "\n")
    try:
        profile_target_logit_bias(path, output_dim=2)
    except ValueError as error:
        assert "no eligible KC targets" in str(error)
    else:
        raise AssertionError("A wholly unsupported training manifest must fail")


def test_profile_optimizer_groups_raise_only_head_lr() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.representation = torch.nn.Linear(2, 2)
            self.profile_ownership_head = torch.nn.Linear(2, 1)

    wrapper = Wrapper()
    groups = profile_optimizer_groups(wrapper, 1e-6, 3e-4)
    assert [group["group_name"] for group in groups] == [
        "representation", "profile_ownership_head"
    ]
    assert [group["lr"] for group in groups] == [1e-6, 3e-4]


def test_freeze_profile_ownership_head_leaves_representation_trainable() -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.representation = torch.nn.Linear(2, 2)
            self.profile_ownership_head = torch.nn.Linear(2, 1)

    wrapper = Wrapper()
    freeze_profile_ownership_head(wrapper)
    assert all(parameter.requires_grad for parameter in wrapper.representation.parameters())
    assert not any(
        parameter.requires_grad for parameter in wrapper.profile_ownership_head.parameters()
    )


def test_training_projection_matches_frozen_evaluator() -> None:
    from scripts.evaluate_state8_reusable_interface import canonical_features

    state = torch.randn(2, 3, 4, 5)
    projection = stable_projection(5, 7, 23)
    observed = profile_state_features(state, projection).detach().numpy()
    expected = canonical_features(state, 7, 23)
    torch.testing.assert_close(torch.from_numpy(observed), torch.from_numpy(expected))


def test_masked_profile_loss_rewards_correct_student_ownership() -> None:
    prediction = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)
    target = torch.tensor([[0.1, 0.9, 0.4]])
    negative = torch.tensor([[0.9, 0.1, 0.4]])
    mask = torch.tensor([[True, True, True]])
    overlap = torch.tensor([[True, True, False]])
    loss, metrics = masked_profile_loss(
        prediction, target, mask, negative, overlap, ownership_margin=0.1
    )
    assert float(metrics["profile_ownership_preference"].detach()) > 0.1
    assert float(metrics["profile_ownership_margin_loss"].detach()) == 0.0
    loss.backward()
    assert prediction.grad is not None


def test_masked_profile_loss_excludes_sparse_kcs_from_value_and_gradient() -> None:
    prediction = torch.tensor([[0.2, 0.9]], requires_grad=True)
    target = torch.tensor([[0.1, 0.1]])
    negative = torch.tensor([[0.9, 0.9]])
    mask = torch.tensor([[True, True]])
    overlap = torch.tensor([[True, True]])
    loss, _ = masked_profile_loss(
        prediction,
        target,
        mask,
        negative,
        overlap,
        ownership_margin=0.0,
        kc_weight=torch.tensor([1.0, 0.0]),
    )
    loss.backward()
    assert prediction.grad is not None
    assert float(prediction.grad[0, 0]) != 0.0
    assert float(prediction.grad[0, 1]) == 0.0


def test_row_targets_never_reads_student_identity() -> None:
    row = {
        "student_id": "secret",
        "future_smoothed_rates": [0.2, None, 0.8],
        "future_eligible_mask": [True, False, True],
        "ownership_negative_future_smoothed_rates": [0.8, None, 0.2],
        "ownership_overlap_mask": [True, False, True],
    }
    target, mask, negative, overlap = row_targets(row, torch.device("cpu"))
    torch.testing.assert_close(target, torch.tensor([[0.2, 0.5, 0.8]]))
    torch.testing.assert_close(negative, torch.tensor([[0.8, 0.5, 0.2]]))
    assert mask.tolist() == [[True, False, True]]
    assert overlap.tolist() == [[True, False, True]]


def test_masked_profile_panel_loss_rewards_all_correct_ownerships() -> None:
    prediction = torch.tensor([[0.2, 0.8]], requires_grad=True)
    target = torch.tensor([[0.1, 0.9]])
    negatives = torch.tensor([[[0.9, 0.1], [0.7, 0.3]]])
    mask = torch.tensor([[True, True]])
    overlaps = torch.tensor([[[True, True], [True, True]]])
    loss, metrics = masked_profile_panel_loss(
        prediction,
        target,
        mask,
        negatives,
        overlaps,
        ownership_margin=0.1,
    )
    assert float(metrics["profile_ownership_preference"]) > 0.1
    assert float(metrics["profile_ownership_margin_loss"]) == 0.0
    assert float(metrics["profile_ownership_panel_size"]) == 2.0
    loss.backward()
    assert prediction.grad is not None


def test_row_target_panel_falls_back_and_reads_panel() -> None:
    row = {
        "future_smoothed_rates": [0.2, 0.8],
        "future_eligible_mask": [True, True],
        "ownership_negative_future_smoothed_rates": [0.8, 0.2],
        "ownership_overlap_mask": [True, True],
    }
    _, _, legacy_targets, legacy_masks = row_target_panel(row, torch.device("cpu"))
    assert legacy_targets.shape == (1, 1, 2)
    assert legacy_masks.shape == (1, 1, 2)
    row["ownership_negative_future_smoothed_rates_panel"] = [
        [0.8, 0.2], [0.7, 0.3]
    ]
    row["ownership_overlap_mask_panel"] = [[True, True], [True, False]]
    _, _, targets, masks = row_target_panel(row, torch.device("cpu"))
    assert targets.shape == (1, 2, 2)
    assert masks.tolist() == [[[True, True], [True, False]]]


def test_row_target_panel_uses_seen_scope_for_native_and_overlap() -> None:
    row = {
        "future_smoothed_rates": [0.2, 0.8],
        "future_eligible_mask": [True, True],
        "ownership_training_mask": [True, False],
        "ownership_negative_future_smoothed_rates": [0.8, 0.2],
        "ownership_overlap_mask": [True, False],
    }
    _, native_mask, _, overlap = row_target_panel(row, torch.device("cpu"))
    assert native_mask.tolist() == [[True, False]]
    assert overlap.tolist() == [[[True, False]]]
