import numpy as np

from scripts.evaluate_dbe_student_profile_probe import (
    feature_views,
    fit_weighted_ridge,
    matched_pairs,
    predict_ridge,
    swap_metrics,
    weakness_metrics,
)


def row(weak, eligible=None):
    eligible = eligible or [True] * 4
    return {
        "weak_kc_indices": weak,
        "future_eligible_mask": eligible,
    }


def test_feature_views_shapes_are_fixed():
    state = np.arange(3 * 4 * 2 * 5, dtype=np.float32).reshape(3, 4, 2, 5)
    views = feature_views(state, projection_dim=3, projection_seed=7)
    assert views["final_layer_mean_slots"].shape == (3, 5)
    assert views["mean_layers_mean_slots"].shape == (3, 5)
    assert views["layerwise_mean_slots_rp"].shape == (3, 12)


def test_weighted_ridge_ignores_unsupported_cells():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([[0.0], [0.3], [0.7], [1.0]], dtype=np.float32)
    mask = np.ones_like(y, dtype=bool)
    weight, bias = fit_weighted_ridge(x, y, mask, alpha=0.01)
    prediction = predict_ridge(x, weight, bias)
    assert prediction[0, 0] < prediction[-1, 0]


def test_weighted_ridge_uses_train_only_constants_for_sparse_kcs():
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([[0.2, 0.9], [0.4, 0.0], [0.6, 0.0]], dtype=np.float32)
    mask = np.asarray([[True, True], [True, False], [True, False]])
    weight, bias = fit_weighted_ridge(x, y, mask, alpha=1.0)
    prediction = predict_ridge(x, weight, bias)
    assert np.allclose(prediction[:, 1], 0.9)


def test_weakness_and_swap_follow_the_supplied_vector():
    rows = [row([0, 1]), row([2, 3])]
    prediction = np.asarray([[0.0, 0.1, 0.8, 0.9], [0.8, 0.9, 0.0, 0.1]])
    weak = weakness_metrics(prediction, rows, np.ones((2, 4), dtype=bool))
    swap = swap_metrics(prediction, rows, seed=1)
    assert weak["weak_top2_exact_set_accuracy"] == 1.0
    assert swap["swap_to_donor_weak_top2_f1"] == 1.0
    assert swap["swap_to_native_weak_top2_f1"] == 0.0
    assert swap["D_profile_specific"] == 1.0
    assert matched_pairs(rows, seed=1) == [(0, 1)]
