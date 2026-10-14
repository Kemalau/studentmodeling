from __future__ import annotations

import numpy as np

from scripts.evaluate_state8_reusable_interface import (
    metrics,
    prediction_reliability,
    prediction_stability_null,
    readable_profiles,
    scientific_gates,
)


def test_disjoint_history_stability_rejects_permuted_students() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(40, 8))
    right = left + rng.normal(scale=0.01, size=left.shape)
    result = prediction_stability_null(left, right, 200, 11)
    assert result["same_student"] < result["permuted_student_mean"]
    assert result["p_one_sided"] < 0.05


def test_reliability_diagnostic_exposes_constant_state_collapse() -> None:
    prediction = np.full((40, 4), 0.5)
    result = prediction_reliability(prediction, prediction, np.ones(4))
    assert result["role"] == "diagnostic_only_not_a_scientific_gate"
    assert result["mean_noise_adjusted_signal_over_fit_variance"] == 0.0


def test_reliability_diagnostic_excludes_zero_variance_kcs() -> None:
    left = np.asarray([[0.2, 0.5], [0.8, 0.5]])
    result = prediction_reliability(left, left, np.asarray([0.1, 0.0]))
    assert result["supported_kcs"] == 1


def test_new_c4_gates_do_not_require_cross_student_swap() -> None:
    comparison = {
        "rmse_delta_model_minus_reference": {
            "bootstrap_probability_model_better": 0.99
        },
        "weak_f1_delta_model_minus_reference": {
            "bootstrap_probability_model_better": 0.98
        },
    }
    result = {
        "students": {"test": 88},
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "uncertainty": {
            "state8_vs_population": comparison,
            "state8_vs_same_window_kc_outcomes": comparison,
            "state_ownership_permutation_null": {
                "rmse": {"p_one_sided": 0.01}
            },
            "disjoint_history_prediction_stability": {"p_one_sided": 0.02},
        },
    }
    gates = scientific_gates(result)
    assert all(gates.values())
    assert all("swap" not in name for name in gates)


def test_same_window_baseline_is_a_scientific_gate() -> None:
    strong = {
        "rmse_delta_model_minus_reference": {
            "bootstrap_probability_model_better": 0.99
        },
        "weak_f1_delta_model_minus_reference": {
            "bootstrap_probability_model_better": 0.99
        },
    }
    weak = {
        "rmse_delta_model_minus_reference": {
            "bootstrap_probability_model_better": 0.50
        },
        "weak_f1_delta_model_minus_reference": {
            "bootstrap_probability_model_better": 0.50
        },
    }
    result = {
        "students": {"test": 88},
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "uncertainty": {
            "state8_vs_population": strong,
            "state8_vs_same_window_kc_outcomes": weak,
            "state_ownership_permutation_null": {"rmse": {"p_one_sided": 0.01}},
            "disjoint_history_prediction_stability": {"p_one_sided": 0.01},
        },
    }
    gates = scientific_gates(result)
    assert gates["predictive_utility_rmse"]
    assert not gates["predictive_utility_beyond_same_window_rmse"]
    assert not gates["readable_weakness_beyond_same_window"]


def test_weakness_predictions_do_not_use_test_future_support() -> None:
    prediction = np.asarray([[0.1, 0.2, 0.3, 0.4]])
    target = np.asarray([[0.1, 0.2, 0.3, 0.4]])
    future_mask = np.asarray([[False, False, True, True]])
    rows = [
        {
            "example_id": "student-1",
            "weak_kc_indices": [0, 1],
            "weak_kcs": ["a", "b"],
            "future_eligible_mask": [False, False, True, True],
        }
    ]
    fixed = np.asarray([True, True, True, True])
    result = metrics(
        prediction,
        target,
        future_mask,
        rows,
        weakness_candidate_mask=fixed,
    )
    assert result["weakness"]["weak_top2_macro_f1"] == 1.0
    profiles = readable_profiles(prediction, rows, ["a", "b", "c", "d"], fixed)
    assert profiles[0]["predicted_weaknesses"] == ["a", "b"]
    assert profiles[0]["future_support_visible_to_readout"] is False


def test_bootstrap_weakness_uses_the_fixed_candidate_set() -> None:
    # Import through the already-loaded evaluator module; its script-directory
    # setup mirrors the production entrypoint's import environment.
    from scripts.evaluate_state8_reusable_interface import profile_v3

    model = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
    reference = np.asarray([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])
    target = model.copy()
    # A leaked per-student future mask would force the first row to candidates
    # c/d and reverse the intended comparison.
    future_mask = np.asarray([[False, False, True, True], [True, True, False, False]])
    rows = [
        {"weak_kc_indices": [0, 1]},
        {"weak_kc_indices": [0, 1]},
    ]
    fixed = np.asarray([True, True, True, True])
    result = profile_v3.paired_student_bootstrap(
        model,
        reference,
        target,
        future_mask,
        rows,
        50,
        7,
        fixed,
    )
    assert result["weak_f1_delta_model_minus_reference"]["value"] == 1.0
