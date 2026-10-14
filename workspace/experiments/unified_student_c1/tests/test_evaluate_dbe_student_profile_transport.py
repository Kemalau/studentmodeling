import numpy as np

from scripts.evaluate_dbe_student_profile_transport import (
    build_transport_pairs,
    readable_profile,
    summarize_transport,
    transport_records,
)


def test_readable_profile_maps_low_and_high_scores_to_named_kcs():
    value = readable_profile(
        ["a", "b", "c", "d"],
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        np.asarray([True, True, True, True]),
    )
    assert value["weak_indices"] == [0, 1]
    assert value["strong_indices"] == [3, 2]
    assert "<weakness>a; b</weakness>" in value["text"]
    assert "raw interaction history" in value["text"]


def test_transport_pair_builder_requires_opposite_future_profiles():
    target = np.asarray([[0.1], [0.2], [0.8], [0.9]])
    mask = np.ones_like(target, dtype=bool)
    assert build_transport_pairs(
        target, mask, min_gap=0.5, max_pairs_per_kc=2
    ) == [(0, 0, 3), (0, 1, 2)]


def test_fixed_query_transport_follows_donor_state_not_recipient():
    target = np.asarray(
        [
            [0.1, 0.8],
            [0.2, 0.7],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.12, 0.82],
            [0.88, 0.12],
        ]
    )
    prediction = target.copy()
    mask = np.ones_like(target, dtype=bool)
    records, summary = transport_records(
        prediction,
        target,
        mask,
        ["kc0", "kc1"],
        seed=7,
        min_gap=0.5,
        max_pairs_per_kc=1,
    )
    assert len(records) == 4
    assert summary["clean"]["swap"]["flip_to_donor_rate"] == 1.0
    assert summary["clean"]["noop"]["flip_to_donor_rate"] == 0.0
    assert summary["clean"]["same_profile"]["flip_to_donor_rate"] == 0.0


def test_empty_clean_transport_is_reported_without_division_error():
    records = [
        {
            "native_closer_to_recipient": False,
            "conditions": {
                name: {"closer_to_donor": False, "directional_progress": 0.0}
                for name in ("noop", "same_profile", "random", "swap")
            },
        }
    ]
    summary = summarize_transport(records)
    assert summary["clean"]["directions"] == 0
    assert summary["D_profile_specific"] is None
