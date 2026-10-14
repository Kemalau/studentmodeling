from scripts.build_dbe_student_profile_pilot import build_profile_row, kc_outcomes


def event(index: int, kc: str, correct: int) -> dict:
    return {
        "interaction_id": str(index),
        "timestamp": f"2026-01-01 00:{index:02d}:00",
        "question_id": str(index),
        "question_text": f"Question {index}",
        "kc_ids": [kc],
        "kc_names": [kc],
        "options": {},
        "student_response": "x",
        "correct_response": "y",
        "correctness": correct,
        "hint_used": False,
    }


def test_kc_outcomes_deduplicates_multi_route_names():
    row = event(1, "a", 1)
    row["kc_names"] = ["a", "a"]
    assert kc_outcomes([row], ["a", "b"]) == ([1, 0], [1, 0])


def test_profile_targets_come_after_cutoff_and_rank_weaknesses():
    vocabulary = ["a", "b", "c", "d"]
    prefix = [event(index, vocabulary[index % 4], index % 2) for index in range(12)]
    future = []
    index = len(prefix)
    for kc, outcomes in {
        "a": [0, 0, 0],
        "b": [0, 0, 1],
        "c": [1, 1, 0],
        "d": [1, 1, 1],
    }.items():
        for outcome in outcomes:
            future.append(event(index, kc, outcome))
            index += 1
    row = build_profile_row(
        split="test",
        student_id="secret-id",
        trace=prefix + future,
        vocabulary=vocabulary,
        max_history=4,
        min_future=3,
        min_eligible=4,
    )
    assert row is not None
    assert row["state_history_length"] == 4
    assert row["weak_kcs"] == ["a", "b"]
    assert row["strong_kcs"] == ["d", "c"]
    assert "secret-id" not in row["prompt"]
    assert row["target_provenance"]["student_id_visible_to_encoder"] is False
