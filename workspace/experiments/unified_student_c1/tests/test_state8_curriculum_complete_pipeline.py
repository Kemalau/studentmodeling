from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_curriculum_pipeline_connects_selection_to_exact_fit_and_final_eval():
    selection = text("watch_state8_curriculum_selection.sh")
    promotion = text("watch_state8_curriculum_promotion.sh")
    repair = text("watch_state8_curriculum_repair_branches.sh")
    finalfit = text("watch_state8_curriculum_finalfit_world2.sh")
    finaleval = text("watch_state8_curriculum_final_eval.sh")
    objective = text("watch_state8_curriculum_complete_objective.sh")
    global_selection = text("watch_state8_global_selection_and_promotion.sh")
    assert "select_state8_curriculum_checkpoint.py" in selection
    assert "state8_curriculum_screens_20260812" in selection
    assert "STATE8_PROMOTION_SELECTION_FILE" in promotion
    assert "state8_curriculum_repair_branches_20260812" in promotion
    assert "STATE8_CURRICULUM_REPAIR=1" in repair
    assert "state8_curriculum_v1/promotion.json" in promotion
    assert "watch_state8_eedi_same_question_finalfit_world2.sh" in finalfit
    assert "SAME_QUESTION_FINAL_EVAL=1" in finaleval
    assert "state8_curriculum_complete_final_eval_20260812/final_gate.json" not in objective
    assert 'OBJECTIVE_FINAL_GATE="$OBJECTIVE_FINAL_EVAL_DIR/final_gate.json"' in objective
    assert "state8_curriculum_cp3000_proactive_repairs_20260812" in global_selection
    assert "proactive_cp3000" in global_selection
    assert "proactive_cp4000" in global_selection
    assert "state8_causal_teacher_checkpoint_selection_20260812" in global_selection
    assert "causal_teacher_binding" in global_selection
    assert "state8_early_label_checkpoint_selection_20260813" in global_selection
    assert "early_label_binding" in global_selection
    assert 'repair_sources=("$causal_teacher" "$early_label" "$interleaved")' in global_selection
    assert "interleaved_retention" in global_selection
    assert "repair_pass == 1" in global_selection
    assert "repair_terminal == ${#repair_sources[@]}" in global_selection
    assert "optional_sources" in global_selection


def test_causal_teacher_checkpoint_selection_is_wired_before_global_promotion():
    selector = text("watch_state8_causal_teacher_checkpoint_selection.sh")
    supervisor = text("supervise_state8_curriculum_complete_pipeline.sh")
    assert "8201:causal_teacher_low:20" in selector
    assert "8201:causal_teacher_low:100" in selector
    assert "8201:causal_teacher_low:240" in selector
    assert "8201:causal_teacher_low:500" in selector
    assert "8202:causal_teacher_high:20" in selector
    assert "8202:causal_teacher_high:100" in selector
    assert "8202:causal_teacher_high:240" in selector
    assert "8202:causal_teacher_high:500" in selector
    assert "select_state8_causal_teacher_checkpoint.py" in selector
    assert "causal_teacher_checkpoint_selection" in supervisor
    assert "early_label_repairs" in supervisor
    assert "early_label_milestones" in supervisor
    assert "early_label_checkpoint_selection" in supervisor
    assert "ensure_stage proactive_repairs_cp3000" not in supervisor
    assert "ensure_stage proactive_repairs_cp4000" not in supervisor
    assert "ensure_stage repair_branches" not in supervisor
    assert "ensure_stage repair_checkpoint_selection" not in supervisor


def test_all_new_watchers_are_fail_closed():
    for name in (
        "watch_state8_curriculum_selection.sh",
        "watch_state8_curriculum_repair_branches.sh",
        "watch_state8_curriculum_promotion.sh",
        "watch_state8_curriculum_finalfit_world2.sh",
        "watch_state8_curriculum_final_eval.sh",
        "watch_state8_curriculum_complete_objective.sh",
    ):
        value = text(name)
        assert "set -" in value
        assert "complete" in value or "exec bash" in value
