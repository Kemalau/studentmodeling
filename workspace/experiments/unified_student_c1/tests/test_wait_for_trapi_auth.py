from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trapi_probe_is_serialized_and_bounded() -> None:
    source = (ROOT / "scripts/wait_for_trapi_auth.sh").read_text(encoding="utf-8")
    assert "studentmodeling-trapi-auth-probe.lock" in source
    assert "timeout 30 az account get-access-token --resource api://trapi" in source
    assert "sleep 60" in source


def test_judge_watchers_preflight_trapi_auth() -> None:
    for name in (
        "watch_state8_curriculum_repair_checkpoint_screen.sh",
        "watch_state8_eedi_same_question_repair_branches.sh",
        "watch_state8_final_eval.sh",
        "watch_taskonly_qwen_formal_eval.sh",
        "watch_state8_interleaved_retention_selection.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "bash scripts/wait_for_trapi_auth.sh" in source
