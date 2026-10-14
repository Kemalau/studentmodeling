from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_qualified_teacher_curriculum_amlt import load, validate  # noqa: E402


def test_stage_b_config_passes() -> None:
    config = load(ROOT / "amlt/feeds_state8_qualified_teacher_curriculum_20260805.yaml")
    validate(config)
    assert {job["submit_args"]["env"]["STATE8_REPLAY_VARIANT"] for job in config["jobs"]} == {"t3_heavy", "balanced"}
    assert {job["submit_args"]["env"]["STATE8_SAVE_STEPS"] for job in config["jobs"]} == {"125"}
    highkd = next(job for job in config["jobs"] if "highkd" in job["name"])
    assert highkd["submit_args"]["env"]["STATE8_KD_VARIANT"] == "highkd"


def test_stage_b_filtered_highkd_config_passes() -> None:
    config = load(ROOT / "amlt/feeds_state8_qualified_teacher_curriculum_20260805.yaml")
    config["jobs"] = [job for job in config["jobs"] if "highkd" in job["name"]]
    validate(config)


def test_stage_b_pins_the_completed_sixteen_way_teacher_provenance() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "pg['evaluation_shards']==16 and pg['evaluation_rows']==1494" in launcher
    assert "pg['evaluation_shards']==4" not in launcher


def test_stage_b_uses_preemption_resilient_checkpoint_cadence() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert 'STATE8_SAVE_STEPS="${STATE8_SAVE_STEPS:-125}"' in launcher
    assert '--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 25' in launcher
    assert '--save-steps "$STATE8_SAVE_STEPS" --checkpoint-retention 2' in launcher
    assert "--probe-expected-cadence" in launcher


def test_stage_b_offloads_every_direct_teacher_forward_after_observed_oom() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "--direct-offload-min-tokens 1" in launcher
    assert "cfg['direct_offload_min_tokens']==1" in launcher


def test_stage_b_exact_resume_preserves_initialization_provenance() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    resume_branch = launcher.split('if [[ -n "$resume_checkpoint" ]]', 1)[1].split("else", 1)[0]
    assert '--init-recurrent-state "$STUDENT_STATE_PATH"' in resume_branch
    assert '--resume-from-checkpoint "$resume_checkpoint"' in resume_branch


def test_stage_b_can_widen_a_hash_pinned_legacy_state() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    for fragment in (
        'STATE8_STUDENT_INIT_CAPACITY="${STATE8_STUDENT_INIT_CAPACITY:-8}"',
        'STATE8_STUDENT_INIT_ADAPTER_SHA256',
        'STATE8_STUDENT_INIT_STATE_SHA256',
        "int(config.get('state_token_count',1))==expected_capacity",
        '--init-recurrent-state "$STUDENT_STATE_PATH"',
        '--init-adapter "$STUDENT_MODEL_DIR"',
        '--state-token-count 8',
    ):
        assert fragment in launcher


def test_stage_b_checkpoints_are_scoped_to_the_exact_optimizer_recipe() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_v3" in launcher
    assert "STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_kd1_v2" in launcher
    assert "state8_qualified_teacher_curriculum_seed42_${STATE8_STAGE_B_RECIPE_ID}" in launcher
    assert "state8_qualified_teacher_balanced_curriculum_seed42_${STATE8_STAGE_B_RECIPE_ID}" in launcher
    assert "state8_qualified_teacher_curriculum_seed42_v1_20260805" not in launcher
    assert "state8_qualified_teacher_balanced_curriculum_seed42_v1_20260805" not in launcher


def test_stage_b_cache_preserves_exact_resume_logical_paths() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert 'ln -s "$DATA_CACHE_ROOT/common/valid.jsonl" "$VALID_JSONL"' in launcher
    assert 'ln -s "$DATA_CACHE_ROOT/$STATE8_REPLAY_VARIANT/train5000.jsonl" "$SUBSET"' in launcher
    assert 'export VALID_JSONL="$DATA_CACHE_ROOT/common/valid.jsonl"' not in launcher
    assert 'export SUBSET="$DATA_CACHE_ROOT/$STATE8_REPLAY_VARIANT/train5000.jsonl"' not in launcher


def test_stage_b_highkd_arm_is_isolated_and_scientifically_explicit() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert 'STATE8_KD_VARIANT="${STATE8_KD_VARIANT:-base}"' in launcher
    assert "STATE8_ONLINE_KD_WEIGHT=1.0" in launcher
    assert "state8_qualified_teacher_curriculum_highkd_v1_immutable" in launcher
    assert "state8_qualified_teacher_curriculum_highkd_immutable_v1" in launcher
    assert '--online-kd-weight "$STATE8_ONLINE_KD_WEIGHT"' in launcher
    assert "exact5000_rounds1250_kd1_v1" not in launcher
