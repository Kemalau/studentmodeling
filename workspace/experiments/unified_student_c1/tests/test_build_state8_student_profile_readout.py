import json

from scripts.build_state8_student_profile_readout import (
    NEUTRAL_TARGET,
    attach_ownership_negatives,
    build_pairs,
    future_profile,
    history_profile,
    profile_label,
    profile_prompt,
)


def event(index, kc, correct):
    return {
        "interaction_id": str(index),
        "question_id": str(index),
        "question_text": f"q{index}",
        "kc_names": [kc],
        "correctness": correct,
    }


def test_profile_uses_only_visible_history():
    row = {
        "kc_vocabulary": ["a", "b", "c", "d"],
        "state_history": [
            event(1, "a", 0), event(2, "a", 0), event(3, "b", 1),
            event(4, "c", 1), event(5, "d", 0), event(6, "d", 1),
        ],
        "future_counts": [99] * 4,
        "future_correct": [99, 0, 0, 0],
    }
    profile = history_profile(row)
    assert profile["weaknesses"][0] == "a"
    assert not set(profile["weaknesses"]) & set(profile["strengths"])
    assert "future" not in json.dumps(profile).lower()
    assert "<student_history>" not in profile_prompt()


def test_future_profile_uses_post_cutoff_labels_as_target_not_prompt():
    row = {
        "kc_vocabulary": ["a", "b", "c", "d"],
        "weak_kcs": ["a", "b"],
        "strong_kcs": ["d", "c"],
    }
    profile = future_profile(row)
    prompt = profile_prompt(row["kc_vocabulary"])
    assert profile["weaknesses"] == ["a", "b"]
    assert profile["strengths"] == ["d", "c"]
    assert "<weakness>a; b</weakness>" in profile["target"]
    assert "<weakness>a; b</weakness>" not in prompt
    assert "a; b; c; d" in prompt


def test_pairs_hold_query_fixed_and_change_only_state_identity():
    rows = []
    for index, weak in enumerate((["a", "b"], ["c", "d"], ["a", "b"], ["c", "d"])):
        rows.append(
            {
                "student_id": str(index),
                "update_texts": [f"update-{index}"],
                "target": f"<weakness>{'; '.join(weak)}</weakness><strength>x; y</strength>",
                "profile": {
                    "weaknesses": list(weak),
                    "strengths": ["x", "y"],
                    "visible_kc_attempts": 8,
                },
            }
        )
    pairs = build_pairs(rows, 2, 7)
    assert all(row["query_held_fixed"] and row["only_state_identity_changes"] for row in pairs)
    assert all(row["native_target"] != row["donor_target"] for row in pairs)
    assert all(row["recipient_id"] != row["same_profile_control_id"] for row in pairs)
    assert all(row["native_control_role"] == "recipient_native" for row in pairs)
    assert all(row["recipient_native_update_texts"] == row["native_update_texts"] for row in pairs)
    assert all(row["native_target_label"] != row["donor_target_label"] for row in pairs)
    assert "insufficient_evidence" in NEUTRAL_TARGET


def test_ownership_negative_is_hard_mismatch_and_train_only():
    rows = []
    for index, rates in enumerate(
        ([0.1, 0.2, 0.8, 0.9], [0.15, 0.25, 0.75, 0.85], [0.9, 0.8, 0.2, 0.1])
    ):
        rows.append(
            {
                "student_id": str(index),
                "future_smoothed_rates": list(rates),
                "future_eligible_mask": [True] * 4,
            }
        )
    attach_ownership_negatives(rows, 17)
    assert rows[0]["ownership_negative_student_id"] == "2"
    assert rows[0]["ownership_target_mse"] > 0.4
    assert rows[0]["ownership_supervision_visible_at_inference"] is False


def test_ownership_negative_overlap_excludes_sparse_kcs() -> None:
    rows = [
        {
            "student_id": "a",
            "future_smoothed_rates": [0.1, 0.9],
            "future_eligible_mask": [True, True],
        },
        {
            "student_id": "b",
            "future_smoothed_rates": [0.8, 0.1],
            "future_eligible_mask": [True, True],
        },
    ]
    attach_ownership_negatives(rows, 17, reliable_kc_mask=[True, False])
    assert rows[0]["ownership_overlap_mask"] == [True, False]


def test_ownership_negative_panel_has_distinct_ranked_students() -> None:
    rows = []
    for index, rates in enumerate(
        ([0.1, 0.2], [0.9, 0.8], [0.7, 0.6], [0.4, 0.3])
    ):
        rows.append({
            "student_id": str(index),
            "future_smoothed_rates": list(rates),
            "future_eligible_mask": [True, True],
        })
    attach_ownership_negatives(rows, 17, negative_count=3)
    assert rows[0]["ownership_negative_panel_size"] == 3
    assert rows[0]["ownership_negative_student_ids"][0] == "1"
    assert len(set(rows[0]["ownership_negative_student_ids"])) == 3
    assert rows[0]["ownership_negative_future_smoothed_rates"] == (
        rows[0]["ownership_negative_future_smoothed_rates_panel"][0]
    )
    assert rows[0]["ownership_overlap_mask"] == rows[0]["ownership_overlap_mask_panel"][0]


def test_seen_scope_uses_only_updater_observed_kcs_and_can_skip() -> None:
    rows = [
        {
            "student_id": "a",
            "future_smoothed_rates": [0.1, 0.9],
            "future_eligible_mask": [True, True],
            "observed_kc_mask": [True, False],
        },
        {
            "student_id": "b",
            "future_smoothed_rates": [0.8, 0.1],
            "future_eligible_mask": [True, True],
            "observed_kc_mask": [True, False],
        },
    ]
    attach_ownership_negatives(rows, 17, ownership_scope="seen")
    assert rows[0]["ownership_training_mask"] == [True, False]
    assert rows[0]["ownership_overlap_mask"] == [True, False]
    rows[0]["observed_kc_mask"] = [False, False]
    attach_ownership_negatives(rows, 17, ownership_scope="seen")
    assert rows[0]["ownership_training_eligible"] is False
    assert rows[0]["ownership_negative_panel_size"] == 0


def test_profile_label_is_order_invariant_and_prefix_free():
    left = profile_label({"weaknesses": ["b", "a"]})
    right = profile_label({"weaknesses": ["a", "b"]})
    assert left == right == "a || b"
    assert "<" not in left and ">" not in left and "\n" not in left
