from __future__ import annotations

from scripts.audit_state8_future_profile_v2 import interface_gates, set_f1


def test_set_f1_is_order_independent() -> None:
    assert set_f1(["a", "b"], ["b", "a"]) == 1.0
    assert set_f1(["a"], ["a", "b"]) == 2 / 3


def test_interface_gate_recomputation() -> None:
    comparison = {
        "rmse_delta_model_minus_reference": {"bootstrap_probability_model_better": 0.99},
        "weak_f1_delta_model_minus_reference": {"bootstrap_probability_model_better": 0.98},
    }
    metrics = {
        "students": {"test": 88},
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "uncertainty": {
            "state8_vs_population": comparison,
            "state8_vs_same_window_kc_outcomes": comparison,
            "state_ownership_permutation_null": {"rmse": {"p_one_sided": 0.01}},
            "disjoint_history_prediction_stability": {"p_one_sided": 0.02},
        },
    }
    assert all(interface_gates(metrics).values())
