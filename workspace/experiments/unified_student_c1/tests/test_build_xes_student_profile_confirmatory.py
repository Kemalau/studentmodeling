from scripts.build_xes_student_profile_confirmatory import (
    add_interaction_ids,
    build_confirmatory_rows,
)


def event(position: int, kc: str, correct: int) -> dict:
    return {
        "timestamp": position,
        "question_id": str(position),
        "question_text": f"Question {position}",
        "kc_ids": [kc],
        "kc_names": [kc],
        "options": {},
        "student_response": "a",
        "correct_response": "a" if correct else "b",
        "correctness": correct,
        "question_type": "choice",
    }


def trace(offset: int) -> list[dict]:
    values = []
    for position in range(40):
        kc = f"kc{position % 4}"
        values.append(event(offset + position, kc, (offset + position) % 2))
    return values


def test_confirmatory_builder_keeps_official_test_students_disjoint() -> None:
    train = {f"tr-{index}": trace(index) for index in range(20)}
    test = {f"te-{index}": trace(100 + index) for index in range(5)}
    add_interaction_ids(train, namespace="train")
    add_interaction_ids(test, namespace="test")
    rows, vocabulary, eligible = build_confirmatory_rows(
        train_traces=train,
        test_traces=test,
        seed=42,
        top_kcs=4,
        max_history=4,
        min_future=3,
        min_eligible=4,
        max_train_students=10,
        max_valid_students=2,
    )
    assert vocabulary == ["kc0", "kc1", "kc2", "kc3"]
    assert len(rows["train"]) == 10
    assert len(rows["valid"]) == 2
    assert len(rows["test"]) == eligible["test"] == 5
    fitting = {row["student_id"] for split in ("train", "valid") for row in rows[split]}
    heldout = {row["student_id"] for row in rows["test"]}
    assert not fitting.intersection(heldout)
    assert all(row["dataset"] == "xes3g5m" for row in rows["test"])
    assert all(
        row["target_provenance"]["official_test_student"] is True
        for row in rows["test"]
    )


def test_add_interaction_ids_is_stable_and_unique() -> None:
    traces = {"student": trace(7)}
    add_interaction_ids(traces, namespace="official_test")
    first = [row["interaction_id"] for row in traces["student"]]
    add_interaction_ids(traces, namespace="official_test")
    second = [row["interaction_id"] for row in traces["student"]]
    assert first == second
    assert len(first) == len(set(first))
