import numpy as np

from scripts.evaluate_dbe_student_profile_transport_v3 import (
    comparison_gates,
    expanded_feature_views,
    history_statistics,
    paired_student_bootstrap,
    per_kc_validation_ensemble,
    permutation_null,
    validate_rows,
)


def test_expanded_feature_views_preserve_slot_and_depth_structure():
    state = np.arange(2 * 8 * 3 * 5, dtype=np.float32).reshape(2, 8, 3, 5)
    views = expanded_feature_views(state, projection_dim=4, projection_seed=7)
    assert views["final_layer_slots_rp"].shape == (2, 12)
    assert views["mean_layers_slots_rp"].shape == (2, 12)
    assert views["depth8_slots_rp"].shape == (2, 96)
    assert views["layerwise_slots_rp"].shape == (2, 96)
    assert views["layer_slot_norms"].shape == (2, 24)
    assert np.allclose(views["layer_slot_norms"], np.linalg.norm(state, axis=-1).reshape(2, -1))


def test_visible_history_statistics_use_only_state_history():
    rows = [
        {
            "state_history": [
                {"kc_names": ["a"], "correctness": 1},
                {"kc_names": ["a", "b"], "correctness": 0},
            ],
            "prefix_counts": [99, 99],
            "prefix_correct": [99, 0],
        }
    ]
    value = history_statistics(rows, ["a", "b"], "state_history")
    assert value.shape == (1, 4)
    assert np.allclose(value[0, :2], [0.5, 1 / 3])
    assert np.allclose(value[0, 2:], np.log1p([2, 1]))


def test_per_kc_selection_uses_validation_not_test():
    train_x = np.arange(8, dtype=np.float32)[:, None]
    valid_x = np.arange(4, dtype=np.float32)[:, None]
    test_x = np.arange(3, dtype=np.float32)[:, None]
    train_y = np.concatenate((train_x / 7, 1 - train_x / 7), axis=1)
    valid_y = np.concatenate((valid_x / 3, 1 - valid_x / 3), axis=1)
    test_y = np.zeros((3, 2), dtype=np.float32)
    targets = {
        "train": (train_y, np.ones_like(train_y, dtype=bool), np.ones_like(train_y)),
        "valid": (valid_y, np.ones_like(valid_y, dtype=bool), np.ones_like(valid_y)),
        "test": (test_y, np.ones_like(test_y, dtype=bool), np.ones_like(test_y)),
    }
    result = per_kc_validation_ensemble(
        {"signal": train_x, "noise": train_x[::-1]},
        {"signal": valid_x, "noise": valid_x[::-1]},
        {"signal": test_x, "noise": test_x[::-1]},
        targets,
        [0.1, 1.0],
        min_valid_targets=3,
    )
    assert result["prediction"].shape == test_y.shape
    assert all(row["selected_by"] == "per_kc_validation_rmse" for row in result["selection"])


def test_bootstrap_and_permutation_report_expected_direction():
    rows = [
        {"weak_kc_indices": [0, 1], "future_eligible_mask": [True] * 4},
        {"weak_kc_indices": [2, 3], "future_eligible_mask": [True] * 4},
        {"weak_kc_indices": [0, 1], "future_eligible_mask": [True] * 4},
        {"weak_kc_indices": [2, 3], "future_eligible_mask": [True] * 4},
    ]
    target = np.asarray(
        [[0.1, 0.2, 0.8, 0.9], [0.8, 0.9, 0.1, 0.2]] * 2
    )
    mask = np.ones_like(target, dtype=bool)
    model = target.copy()
    reference = np.full_like(target, 0.5)
    bootstrap = paired_student_bootstrap(
        model, reference, target, mask, rows, replicates=100, seed=3
    )
    assert bootstrap["rmse_delta_model_minus_reference"]["value"] < 0
    null = permutation_null(model, target, mask, rows, replicates=100, seed=4)
    assert null["rmse"]["observed"] == 0.0


def test_validate_rows_rejects_student_overlap():
    def row(split, student):
        return {
            "split": split,
            "student_id": student,
            "kc_vocabulary": ["a", "b", "c", "d"],
            "state_history": [{"correctness": 1, "kc_names": ["a"]}],
            "state_history_length": 1,
            "prefix_counts": [1] * 4,
            "prefix_correct": [1] * 4,
            "future_counts": [5] * 4,
            "future_correct": [3] * 4,
            "future_eligible_mask": [True] * 4,
        }

    rows = {
        "train": [row("train", "same")],
        "valid": [row("valid", "same")],
        "test": [row("test", "third")],
    }
    with np.testing.assert_raises_regex(ValueError, "overlap"):
        validate_rows(rows)


def test_comparison_gates_require_effect_and_uncertainty():
    def metric(rmse, f1):
        return {
            "future_rate": {"rmse": rmse},
            "weakness": {"weak_top2_macro_f1": f1},
        }

    metrics = {
        "state8": metric(0.1, 0.7),
        "same_window_kc_outcomes": metric(0.2, 0.6),
        "state8_plus_same_window_kc_outcomes": metric(0.09, 0.71),
    }
    uncertainty = {
        "state8_vs_same_window_kc_outcomes": {
            "rmse_delta_model_minus_reference": {
                "bootstrap_probability_model_better": 0.99
            },
            "weak_f1_delta_model_minus_reference": {
                "bootstrap_probability_model_better": 0.98
            },
        },
        "state_ownership_permutation_null": {"rmse": {"p_one_sided": 0.01}},
    }
    transport = {
        "clean": {"directions": 40},
        "D_profile_specific": 0.11,
    }
    assert all(comparison_gates(metrics, uncertainty, transport).values())
