from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERIC = (ROOT / "scripts/watch_layerwise_remote_checkpoint_local_c4.sh").read_text()
WRAPPER = (
    ROOT
    / "scripts/watch_layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3950_local_c4.sh"
).read_text()


def test_generic_remote_checkpoint_c4_is_atomic_audited_and_resumable() -> None:
    for fragment in (
        "download_atomic",
        "checkpoint_completion.json",
        "observed==receipt['sha256']",
        "local_gpu_readonly_preflight.lock",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "'pairs':value['pairs'][:50]",
        "--progress-dir",
        "--num-pairs 50",
        "scientific_gate_pass",
        "EXPECTED_WORLD_SIZE",
        "exact_resume_audit.json",
        "audit_layerwise_promoted_checkpoint.py",
        "EXPECTED_PLANNED_OPTIMIZER_STEPS",
        "expected_planned_optimizer_steps",
        "LOCAL_CHECKPOINT_STATE_DIR",
        "LOCAL_C4_COMPLETION_MARKER",
    ):
        assert fragment in GENERIC


def test_round3950_wrapper_pins_the_expected_artifact() -> None:
    for fragment in (
        "LOCAL_C4_SUBDIR=local_lr2e6_round3950_c4",
        "CHECKPOINT_ROUND=3950",
        "ARTIFACT_LABEL=open-t3-selective-kd-lr2e6",
        "eedi_full_open_t3/state_kd/lr2e-6_start12000_stop4250_",
        "/checkpoint-round-3950",
        "watch_layerwise_remote_checkpoint_local_c4.sh",
    ):
        assert fragment in WRAPPER


def test_promoted_round1000_wrapper_is_read_only_and_recipe_pinned() -> None:
    source = (ROOT / "scripts/watch_layerwise_promoted_round1000_local_c4.sh").read_text()
    for fragment in (
        "CHECKPOINT_ROUND=1000",
        "EXPECTED_WORLD_SIZE=16",
        "EXPECTED_PROMOTED_MANIFEST=",
        "taskretention_eedi_full_open_t3_state_path_v1_20260802/checkpoint-round-1000",
        "watch_layerwise_remote_checkpoint_local_c4.sh",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source


def test_state8_taxonomy_midcurve_is_read_only_state_closed_and_semantically_judged() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_midcurve_local.sh").read_text()
    for fragment in (
        "checkpoint-round-500", "EXPECTED_PLANNED_OPTIMIZER_STEPS=1250",
        "watch_layerwise_remote_checkpoint_local_c4.sh",
        "evaluate_recurrent_formal_protocols_taxonomy.py",
        "--max-samples-per-group 25", "--t3-natural-schema",
        "judge_formal_t3_predictions.py", "--judge-backend trapi",
        "--max-records 75", "exact_match_authoritative':False",
        "pipeline_controller.active.pid", "pipeline_controller.heartbeat",
        "LOCAL_C4_COMPLETION_MARKER=\"$state/c4_complete\"",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "optimizer.step" not in source
    assert source.rindex('touch "$state/complete"') > source.index("midcurve_summary.json")


def test_promoted_round2500_sentinel_is_read_only_and_recipe_pinned() -> None:
    source = (ROOT / "scripts/watch_promoted_round2500_local_c4.sh").read_text()
    for fragment in (
        "CHECKPOINT_ROUND=2500",
        "EXPECTED_WORLD_SIZE=16",
        "EXPECTED_PROMOTED_MANIFEST=",
        "taskretention_eedi_full_open_t3_state_path_v1_20260802/checkpoint-round-2500",
        "watch_layerwise_remote_checkpoint_local_c4.sh",
    ):
        assert fragment in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source
