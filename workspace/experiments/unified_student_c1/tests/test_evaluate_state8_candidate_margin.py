from scripts.evaluate_state8_candidate_margin import (
    full_history_label_prompt,
    structured_teacher_prompt,
    summarize,
)


def row(pair, path, role, preferred, native="a", donor="b", margin=1.0):
    return {
        "pair_id": str(pair),
        "state_path": path,
        "state_role": role,
        "preferred_label": preferred,
        "native_label": native,
        "donor_label": donor,
        "correct_margin_mean_nll": margin,
    }


def test_summary_separates_recurrent_and_direct_paths():
    records = []
    for pair in range(3):
        records.extend(
            [
                row(pair, "student_recurrent", "native", "a", margin=0.2),
                row(pair, "student_recurrent", "donor", "a", margin=-0.1),
                row(pair, "student_direct", "native", "a", margin=0.7),
                row(pair, "student_direct", "donor", "b", margin=0.8),
            ]
        )
    value = summarize(records)
    assert value["student_recurrent"]["state_accuracy"] == 0.5
    assert value["student_recurrent"]["pair_accuracy"] == 0.0
    assert value["student_direct"]["state_accuracy"] == 1.0
    assert value["student_direct"]["pair_accuracy"] == 1.0


def test_summary_rejects_missing_state_role():
    try:
        summarize([row(1, "student_direct", "native", "a")])
    except ValueError as error:
        assert "lacks" in str(error) or "imbalanced" in str(error)
    else:
        raise AssertionError("Incomplete state-role panel should fail closed")


def test_full_history_label_prompt_keeps_history_without_candidates():
    prompt = (
        "instruction <student_history>wrong response</student_history>\n"
        "<STUDENT_STATE>\n<query>same current item</query>\nAnswer:"
    )
    value = full_history_label_prompt(prompt)
    assert "wrong response" in value
    assert "same current item" in value
    assert "<STUDENT_STATE>" not in value
    assert "alternatives" in value


def test_structured_teacher_prompt_removes_state_marker_and_keeps_history():
    prompt = (
        "Return exactly one valid XML-style tag.\n"
        "<student_history>wrong response</student_history>\n"
        "<STUDENT_STATE>\n<query>same current item</query>\nAnswer:"
    )
    value = structured_teacher_prompt(prompt)
    assert "wrong response" in value
    assert "<STUDENT_STATE>" not in value
