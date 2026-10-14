from scripts.evaluate_state8_eedi_same_question_c4 import (
    CONDITIONS,
    select_recipient_twins,
    semantic_panel_rows,
    validate_progress_bundle,
)
from scripts.train_state8_diagnosis_first_causal_repair import configured_field_order


def twin(role="recipient_native"):
    return {
        "pair_id": "p0",
        "source": "eedi_mining",
        "native_control_role": role,
        "same_question_query_identical": True,
        "raw_history_visible_to_decoder": False,
        "student_response_visible_to_decoder": False,
        "native_target_label": "a",
        "donor_target_label": "b",
        "native_target": (
            "<label>a</label><diagnosis>recipient rule</diagnosis>"
            "<evidence>recipient evidence</evidence>"
        ),
        "donor_target": (
            "<label>b</label><diagnosis>donor rule</diagnosis>"
            "<evidence>donor evidence</evidence>"
        ),
    }


def test_selects_only_recipient_native_rows():
    selected = select_recipient_twins([twin(), twin("same_rule_native")])
    assert len(selected) == 1
    assert selected[0]["native_control_role"] == "recipient_native"


def test_semantic_panel_uses_configured_target_order():
    predictions, evaluation = semantic_panel_rows(
        pair_index=1,
        row=twin(),
        condition="swap",
        output="<diagnosis>x</diagnosis><evidence>y</evidence><label>b</label>",
        prompt="prompt",
    )
    assert [row["reference_role"] for row in predictions] == ["donor", "recipient"]
    first_field = configured_field_order()[0]
    assert predictions[0]["target"].startswith(f"<{first_field}>")
    assert [row["pair_idx"] for row in evaluation] == [1, 1]
    assert len(evaluation) == 2


def test_resumable_bundle_is_pair_complete_and_fingerprint_bound():
    fingerprint = {"schema": 1, "progress_key": "frozen"}
    records = [
        {"pair_idx": 1, "condition": condition}
        for condition in CONDITIONS
    ]
    semantic = [
        {"pair_idx": 1, "condition": condition, "reference_role": role}
        for condition in CONDITIONS
        for role in ("donor", "recipient")
    ]
    bundle = {
        "fingerprint": fingerprint,
        "records": records,
        "judge_predictions": semantic,
        "judge_eval": semantic,
    }
    assert validate_progress_bundle(bundle, fingerprint, 150) == {1}


def test_resumable_bundle_rejects_incomplete_pair():
    import pytest

    fingerprint = {"schema": 1}
    bundle = {
        "fingerprint": fingerprint,
        "records": [{"pair_idx": 1, "condition": "native"}],
        "judge_predictions": [],
        "judge_eval": [],
    }
    with pytest.raises(ValueError, match="Incomplete"):
        validate_progress_bundle(bundle, fingerprint, 150)


def test_resumable_bundle_accepts_legacy_judge_eval_example_ids():
    fingerprint = {"schema": 1, "progress_key": "frozen"}
    records = [{"pair_idx": 1, "condition": condition} for condition in CONDITIONS]
    semantic = [
        {
            "pair_idx": 1,
            "condition": condition,
            "reference_role": role,
            "example_id": f"eedi-same-question-c4:0001:{condition}:{role}",
        }
        for condition in CONDITIONS
        for role in ("donor", "recipient")
    ]
    legacy_eval = [
        {"example_id": row["example_id"], "direct_text": "prompt"}
        for row in semantic
    ]
    bundle = {
        "fingerprint": fingerprint,
        "records": records,
        "judge_predictions": semantic,
        "judge_eval": legacy_eval,
    }
    assert validate_progress_bundle(bundle, fingerprint, 1) == {1}
