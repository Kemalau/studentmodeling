from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_state8_profile_ownership_lrfix_v4_formal_amlt import validate as validate_amlt
from scripts.validate_state8_profile_ownership_lrfix_v4_promotion import validate


ROOT = Path(__file__).resolve().parents[1]


def summary(*, p: float = 0.01, delta: float = 0.02) -> dict:
    return {
        "stage": "state8_profile_ownership_lrfix_v4_sparsekc_smoke_watcher_v1",
        "remote_passed": True,
        "failures": [],
        "completion": {"status": "diagnostic_complete", "micro_steps": 200},
        "metrics": {
            "diagnostic_only": True,
            "strict_state_only": True,
            "students": 88,
            "supported_student_kc_cells": 634,
            "ownership_permutation_replicates": 2000,
            "ownership_delta_permuted_mean_minus_native": delta,
            "ownership_permutation_p_one_sided": p,
            "profile_ownership_feature_normalization": "layernorm",
            "native_rmse": 0.4,
        },
        "repair": {
            "head_learning_rate": 3e-4,
            "head_reset_from_training_targets": True,
            "minimum_kc_train_support": 4,
            "bias_prior_strength": 8.0,
            "sparse_kc_policy": "exclude_below_train_support_then_weight_equally",
        },
    }


def write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload))
    return path


def test_formal_amlt_is_bound_and_confirmatory_closed() -> None:
    validate_amlt(
        ROOT
        / "amlt/feeds_state8_profile_ownership_lrfix_v4_formal_world4_20260821.yaml"
    )


def test_positive_significant_smoke_promotes(tmp_path: Path) -> None:
    result = validate(write(tmp_path, summary()))
    assert result["promotion_allowed"] is True
    assert result["xes_official_test_inspected"] is False


def test_r2_watcher_stage_promotes(tmp_path: Path) -> None:
    payload = summary()
    payload["stage"] = "state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_watcher_v1"
    assert validate(write(tmp_path, payload))["promotion_allowed"] is True


def test_evaluation_only_recovery_stage_promotes(tmp_path: Path) -> None:
    payload = summary()
    payload["stage"] = "state8_profile_ownership_lrfix_v4_smoke_recovery_watcher_v1"
    assert validate(write(tmp_path, payload))["promotion_allowed"] is True


@pytest.mark.parametrize("p,delta", [(0.05, 0.02), (0.01, 0.0), (0.8, -0.1)])
def test_gate_fails_closed_on_weak_ownership(tmp_path: Path, p: float, delta: float) -> None:
    assert validate(write(tmp_path, summary(p=p, delta=delta)))["promotion_allowed"] is False
