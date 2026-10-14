from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_repairs_only_local_orchestration() -> None:
    source = (
        ROOT / "scripts/supervise_state8_curriculum_complete_pipeline.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "main_generic",
        "main_current_query",
        "milestones_generic",
        "milestones_current_query",
        "screen_${arm}_${step}",
        "selection",
        "causal_teacher_repairs",
        "causal_teacher_milestones",
        "causal_teacher_checkpoint_selection",
        "early_label_repairs",
        "early_label_milestones",
        "early_label_checkpoint_selection",
        "global_selection",
        "promotion",
        "finalfit",
        "finaleval",
        "objective",
        "scientific_gates_unchanged",
        "remote_submission_authority",
        'elif [[ -f "$directory/needs_attention" ]]',
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "kill " not in source
    assert "rm -f \"$directory/needs_attention\"" not in source
    assert "watch_state8_global_selection_and_promotion.sh" in source
    assert "watch_state8_curriculum_promotion.sh" not in source
    assert "watch_state8_causal_teacher_binding_milestones.sh" in source
    assert "watch_state8_causal_teacher_checkpoint_selection.sh" in source
    assert "state8_causal_teacher_binding_repairs_20260812" in source
    assert "state8_early_label_binding_repairs_20260813" in source
    assert "watch_state8_early_label_binding_milestones.sh" in source
    assert "watch_state8_early_label_checkpoint_selection.sh" in source
    # The completed cp3000/cp4000 full-tail lineages remain read-only evidence;
    # the supervisor must not relaunch those expensive failed branches.
    assert "ensure_stage proactive_repairs_cp3000" not in source
    assert "ensure_stage proactive_repairs_cp4000" not in source
    assert "ensure_stage repair_milestones" not in source
    assert "ensure_stage repair_branches" not in source


def test_supervisor_covers_both_arms_and_all_selection_checkpoints() -> None:
    source = (
        ROOT / "scripts/supervise_state8_curriculum_complete_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "for arm in generic current_query" in source
    assert "for step in 2000 3000 4000 5000" in source
    assert "watch_state8_curriculum_checkpoint_screen.sh" in source
    assert "state8_curriculum_complete_objective_20260812/complete" in source
