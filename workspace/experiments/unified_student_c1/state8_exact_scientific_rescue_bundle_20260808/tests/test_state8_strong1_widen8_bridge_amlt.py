from pathlib import Path

from scripts.validate_state8_strong1_widen8_bridge_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_strong1_widen8_bridge_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_strong1_widen8_bridge_20260808.yaml")


def test_bridge_uses_backfill_friendly_atomic_resume_slices() -> None:
    config = (ROOT / "amlt/feeds_state8_strong1_widen8_bridge_20260808.yaml").read_text()
    assert config.count("max_run_duration_seconds: 43200") == 2
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "select_latest_valid_distributed_checkpoint.py" in launcher
    assert '--recovery-save-steps 5' in launcher
    assert '--resume-from-checkpoint "$resume_checkpoint"' in launcher


def test_bridge_preserves_global_batch_four_on_two_a100s() -> None:
    source = (ROOT / "amlt/feeds_state8_strong1_widen8_bridge_20260808.yaml").read_text()
    assert source.count("sku: 80G2-A100") == 1
    assert source.count("process_count_per_node: -1") == 1
    assert "mpi: true" in source
    assert "STATE8_DISTRIBUTED_WORLD_SIZE: '2'" in source
    assert "STATE8_SAVE_STEPS: '10'" in source
    wrapper = (ROOT / "scripts/run_state8_qualified_teacher_curriculum_world2.sh").read_text()
    renderer = (ROOT / "scripts/render_state8_world2_launcher.py").read_text()
    assert "render_state8_world2_launcher.py" in wrapper
    assert "GRADIENT_ACCUMULATION_STEPS=2" in renderer
    assert "cfg['distributed_world_size']*cfg['gradient_accumulation_steps']==4" in renderer


def test_bridge_watcher_recovers_evaluation_and_judge_failures() -> None:
    source = (ROOT / "scripts/watch_state8_strong1_widen8_bridge.sh").read_text()
    for fragment in (
        "submit_eval_recovery",
        "eval_preserved_results",
        "eval_active_jobs.txt",
        "max_eval_recovery_waves=16",
        "(( pass_count > 0 )) || return 0",
        "T3 semantic judge exhausted",
        "C4 semantic judge exhausted",
    ):
        assert fragment in source


def test_bridge_watcher_recovers_amlt_killed_jobs() -> None:
    source = (ROOT / "scripts/watch_state8_strong1_widen8_bridge.sh").read_text()
    assert source.count("'killed'") >= 3


def test_bridge_watcher_runs_only_retained_seed59_primary_arm() -> None:
    source = (ROOT / "scripts/watch_state8_strong1_widen8_bridge.sh").read_text()
    assert "jobs=(state8-strong1-widen8-bridge-lr1e6-seed59)" in source
    assert "eval_jobs=(state8-strong1-widen8-eval-lr1e6-seed59)" in source
    assert 'amlt run "$eval_config" "${eval_selectors[@]}"' in source
    assert 'for slug in "${slugs[@]}"' in source
    assert '"${selection_args[@]}"' in source
