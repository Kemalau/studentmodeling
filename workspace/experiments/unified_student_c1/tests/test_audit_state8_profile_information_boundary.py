import json

import numpy as np

from scripts.audit_state8_profile_information_boundary import (
    audit_split,
    ownership_null,
)


def event(kc: str, correct: int) -> dict:
    return {"kc_names": [kc], "correctness": correct}


def test_ownership_null_detects_aligned_visible_rate() -> None:
    prediction = np.asarray([[0.1], [0.3], [0.7], [0.9]])
    target = prediction.copy()
    mask = np.ones_like(prediction, dtype=bool)
    value = ownership_null(prediction, target, mask, replicates=999, seed=7)
    assert value["native_rmse"] == 0.0
    assert value["delta_permuted_mean_minus_native"] > 0.2
    assert value["p_one_sided"] <= 0.01


def test_split_audit_separates_seen_and_unseen_future_kcs(tmp_path) -> None:
    rows = []
    for index, outcome in enumerate((0, 0, 1, 1)):
        rows.append(
            {
                "student_id": str(index),
                "cutoff_position": 10,
                "state_history_length": 2,
                "state_history": [event("a", outcome), event("a", outcome)],
                "kc_vocabulary": ["a", "b"],
                "future_smoothed_rates": [0.25 if outcome == 0 else 0.75, 0.5],
                "future_eligible_mask": [True, True],
                "weak_kcs": ["a", "b"],
            }
        )
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = audit_split(path, replicates=199, seed=3)
    assert value["future_supported_cells"] == 8
    assert value["seen_future_supported_cells"] == 4
    assert value["unseen_future_supported_cells"] == 4
    assert value["seen_fraction_of_future_supported_cells"] == 0.5
    assert value["visible_history_length"]["mean_fraction_of_prefix"] == 0.2
    sanity = value["visible_history_rate_ownership_sanity"]
    assert sanity["native_rmse"] < sanity["permuted_rmse_mean"]
