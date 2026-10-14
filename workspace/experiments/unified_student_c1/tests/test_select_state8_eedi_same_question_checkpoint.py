from scripts.select_state8_eedi_same_question_checkpoint import evaluate, select


def summary(arm="updater", step=20):
    c4 = {"semantic_D_specific":0.2,"random_matches_either_stable_target_rate":0.1,
          "noop_recipient_retention":0.8,"same_rule_recipient_retention":0.8,
          "cells":{"swap_to_donor":{"semantic_equivalence_rate":0.6},
                   "native_to_recipient":{"semantic_equivalence_rate":0.7}}}
    return {"stage":"state8_eedi_same_question_checkpoint_semantic_summary_v1",
            "status":"complete","diagnostic_only":True,"semantic_judge_authoritative":True,
            "exact_match_authoritative":False,"arm":arm,"checkpoint_step":step,
            "T3_eedi_semantic":0.4,"T3_synthetic_cross_semantic":0.6,
            "T3_synthetic_same_semantic":0.6,"malrule_diagnosis_c4":c4,
            "eedi_same_question_diagnosis_c4":c4}


def test_evaluate_requires_eedi_causal_sanity():
    assert evaluate(summary())["hard_sanity"] is True
    broken=summary(); broken["eedi_same_question_diagnosis_c4"]["semantic_D_specific"]=0.0
    assert evaluate(broken)["hard_sanity"] is False


def test_identity_controls_are_required_for_early_viability():
    broken=summary()
    broken["eedi_same_question_diagnosis_c4"]["noop_recipient_retention"]=0.5
    result=evaluate(broken)
    assert result["hard_sanity"] is False
    assert "eedi_noop_retention" in result["failed_sanity"]


def test_selection_requires_complete_four_cell_matrix():
    values=[summary(arm,step) for arm in ("updater","full") for step in (20,100)]
    result=select(values)
    assert result["selected_for_extended_training"] is not None
    try:
        select(values[:-1])
    except ValueError as error:
        assert "matrix drift" in str(error)
    else:
        raise AssertionError("Incomplete matrix was accepted")
