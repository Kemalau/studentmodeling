import json

import numpy as np

from scripts.audit_state8_profile_seen_unseen_features import (
    paired_bootstrap,
    permutation_null,
    visible_support,
)


def test_visible_support_uses_exact_history_kcs() -> None:
    rows = [
        {
            "kc_vocabulary": ["a", "b", "c"],
            "state_history": [{"kc_names": ["a", "a"]}, {"kc_names": ["c", "x"]}],
        }
    ]
    np.testing.assert_array_equal(
        visible_support(rows), np.asarray([[True, False, True]])
    )


def test_permutation_and_bootstrap_detect_student_alignment() -> None:
    target = np.asarray([[0.0], [0.2], [0.8], [1.0]])
    model = target.copy()
    reference = np.full_like(target, 0.5)
    mask = np.ones_like(target, dtype=bool)
    ownership = permutation_null(model, target, mask, replicates=999, seed=3)
    comparison = paired_bootstrap(
        model, reference, target, mask, replicates=999, seed=4
    )
    assert ownership["p_one_sided"] < 0.05
    assert ownership["delta_permuted_mean_minus_observed"] > 0
    assert comparison["value"] < 0
    assert comparison["bootstrap_probability_model_better"] > 0.95
