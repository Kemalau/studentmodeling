from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from scripts.build_state8_student_profile_readout import PROFILE_KIND, PROFILE_TASK
import scripts.train_state8_profile_ownership_repair as ownership
from scripts.train_state8_profile_ownership_v5 import (
    apply_profile_loss_budget,
    is_profile_pair,
    masked_profile_causal_row,
    validate_v5_weights,
)


def test_v5_profile_generation_weight_is_bounded() -> None:
    validate_v5_weights(0.0)
    validate_v5_weights(0.1)
    validate_v5_weights(1.0)
    with pytest.raises(ValueError):
        validate_v5_weights(-0.1)
    with pytest.raises(ValueError):
        validate_v5_weights(1.1)


def test_v5_suppresses_only_profile_causal_pairs() -> None:
    profile = {"causal_target_kind": PROFILE_KIND}
    retention = {"causal_target_kind": "diagnosis_natural_schema"}
    assert is_profile_pair(profile)
    assert not is_profile_pair(retention)
    assert masked_profile_causal_row(profile) == (None, True)
    assert masked_profile_causal_row(retention) == (retention, False)
    assert masked_profile_causal_row(None) == (None, False)


def test_v5_loss_budget_changes_only_profile_generation() -> None:
    namespace = Namespace(generation_weight=1.0)
    previous = apply_profile_loss_budget(
        namespace, {"task": PROFILE_TASK}, 0.1
    )
    assert previous == 1.0
    assert namespace.generation_weight == 0.1
    namespace.generation_weight = previous
    previous = apply_profile_loss_budget(
        namespace, {"task": "predict_correctness"}, 0.1
    )
    assert previous == 1.0
    assert namespace.generation_weight == 1.0


def test_v5_runtime_patch_chain_is_nonrecursive() -> None:
    row = {
        "future_smoothed_rates": [0.2, 0.8],
        "future_eligible_mask": [True, True],
        "ownership_negative_future_smoothed_rates": [0.8, 0.2],
        "ownership_overlap_mask": [True, True],
        "ownership_negative_future_smoothed_rates_panel": [
            [0.8, 0.2], [0.7, 0.3]
        ],
        "ownership_overlap_mask_panel": [[True, True], [True, True]],
    }

    def patched_targets(value, device):
        return ownership.row_target_panel(value, device)

    def patched_loss(prediction, target, mask, negatives, overlaps, **kwargs):
        return ownership.masked_profile_panel_loss(
            prediction, target, mask, negatives, overlaps, **kwargs
        )

    original_targets = ownership.row_targets
    original_loss = ownership.masked_profile_loss
    ownership.row_targets = patched_targets
    ownership.masked_profile_loss = patched_loss
    try:
        target, mask, negatives, overlaps = ownership.row_targets(
            row, torch.device("cpu")
        )
        loss, metrics = ownership.masked_profile_loss(
            torch.tensor([[0.2, 0.8]], requires_grad=True),
            target,
            mask,
            negatives,
            overlaps,
            ownership_margin=0.02,
        )
        assert torch.isfinite(loss)
        assert float(metrics["profile_ownership_panel_size"]) == 2.0
    finally:
        ownership.row_targets = original_targets
        ownership.masked_profile_loss = original_loss
