from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_controllers_share_dedicated_feeds_auth() -> None:
    helper = (ROOT / "scripts/use_studentmodeling_feeds_auth.sh").read_text()
    assert ".azure-studentmodeling-feeds" in helper
    assert "env -u AZURE_CONFIG_DIR az account get-access-token" in helper
    assert "studentmodeling_subscription" in helper
    controllers = (
        "watch_state8_eedi_same_question_causal_repair.sh",
        "watch_state8_eedi_same_question_checkpoint_screen.sh",
        "watch_state8_eedi_same_question_repair_branches.sh",
        "watch_state8_eedi_same_question_promotion.sh",
        "watch_state8_eedi_same_question_finalfit_world2.sh",
        "watch_state8_final_eval.sh",
    )
    for name in controllers:
        text = (ROOT / "scripts" / name).read_text()
        assert 'source "$project/scripts/use_studentmodeling_feeds_auth.sh"' in text or (
            'source "$project_dir/scripts/use_studentmodeling_feeds_auth.sh"' in text
        )
        assert ".azure-keyflip-feeds" not in text


def test_reauth_restarts_all_idempotent_same_question_controllers() -> None:
    restart = ROOT / "scripts/restart_state8_eedi_same_question_controllers.sh"
    subprocess.run(["bash", "-n", str(restart)], check=True)
    text = restart.read_text()
    for fragment in (
        "watch_state8_eedi_same_question_causal_repair.sh",
        "watch_state8_eedi_same_question_full_mse1.sh",
        "watch_state8_eedi_same_question_checkpoint_screen.sh",
        "watch_state8_eedi_same_question_early_selection.sh",
        "watch_state8_eedi_same_question_repair_branches.sh",
        "watch_state8_eedi_same_question_promotion.sh",
        "watch_state8_eedi_same_question_finalfit_world2.sh",
        "watch_state8_eedi_same_question_final_eval.sh",
        "watch_state8_eedi_same_question_complete_objective.sh",
        "az account get-access-token",
    ):
        assert fragment in text
    assert "amlt run" not in text
    assert "amlt stop" not in text
    assert "amlt cancel" not in text


def test_reauth_calls_pipeline_restart_only_after_fixed_environment_checks() -> None:
    text = (ROOT / "scripts/reauth_layerwise_unified_pipeline.sh").read_text()
    restart = text.index("restart_state8_eedi_same_question_controllers.sh")
    quota = text.index("quota_after_reauth.json")
    storage = text.index("az storage blob exists")
    assert restart > quota > storage


def test_reauth_restarts_current_state8_and_baseline_controllers_without_remote_mutation() -> None:
    restart = ROOT / "scripts/restart_current_state8_and_baseline_controllers_after_auth.sh"
    subprocess.run(["bash", "-n", str(restart)], check=True)
    text = restart.read_text()
    for fragment in (
        "watch_state8_causal_teacher_checkpoint_selection.sh",
        "watch_state8_early_label_checkpoint_selection.sh",
        "watch_state8_interleaved_retention_selection.sh",
        "watch_state8_global_selection_and_promotion.sh",
        "watch_state8_curriculum_finalfit_world2.sh",
        "watch_state8_curriculum_final_eval.sh",
        "supervise_taskonly_baselines.sh",
        "watch_taskonly_qwen_formal_eval.sh",
        "watch_taskonly_qwen_formal_summary.sh",
        "watch_unrestricted_qwen_t3_v2_rejudge.sh",
        "careful_toe_p87sfl2prk",
        "strong_pepper_8ft9yhwng1",
        "lemon_button_fyw8ljlmqx",
        "bubbly_carpet_7wbmd8wxx1",
        "bright_yam_cpt3f61t59",
        "recover_passed_screen_download",
        "remote_passed",
        "screen result download failed",
        "restart_causal_selector",
        "restart_early_selector",
        "state8_causal_teacher_checkpoint_selection_20260812",
    ):
        assert fragment in text
    assert "amlt cancel" not in text
    assert "amlt stop" not in text
    assert "az storage blob delete" not in text
    assert text.index("restart_causal_selector=0") < text.index(
        "restart_pending state8_causal_teacher_checkpoint_selection_20260813"
    )
    assert text.index("restart_early_selector=0") < text.index(
        "restart_pending state8_early_label_checkpoint_selection_20260813"
    )
    reauth = (ROOT / "scripts/reauth_layerwise_unified_pipeline.sh").read_text()
    call = reauth.index("restart_current_state8_and_baseline_controllers_after_auth.sh")
    quota = reauth.index("quota_after_reauth.json")
    assert call > quota


def test_device_login_lock_contention_is_pending_not_success() -> None:
    reauth = (ROOT / "scripts/reauth_layerwise_unified_pipeline.sh").read_text()
    watcher = (ROOT / "scripts/watch_layerwise_unified_reauth.sh").read_text()
    assert "flock -n 7 || exit 75" in reauth
    assert "rc == 75" in watcher
    assert "authentication remains pending" in watcher
