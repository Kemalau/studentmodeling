from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_is_monitoring_only_and_covers_full_baseline_chain() -> None:
    source = (ROOT / "scripts/supervise_taskonly_baselines.sh").read_text(encoding="utf-8")
    for value in (
        "taskonly-qwen-t1-t2-t4-oneepoch-20260813",
        "taskonly-qwen-t3-t5-oneepoch-20260813",
        "t3-eedi-taskonly-qwen-formal-recovery1-20260813",
        "watch_taskonly_qwen_formal_eval.sh",
        "watch_taskonly_qwen_formal_summary.sh",
        "watch_t3_eedi_taskonly_qwen_sharded_recovery.sh",
        "summarize_taskonly_baseline_live_status.py",
        "watch_unrestricted_qwen_t3_v2_rejudge.sh",
        "auth_required",
        "auth_ready",
        "timeout --foreground 90 amlt",
        'mktemp "$output.partial.XXXXXX"',
    ):
        assert value in source
    assert "amlt run" not in source
    assert "amlt cancel" not in source
    assert 'mkdir -p "$state"' in source


def test_t3_eedi_monitor_fails_closed_on_full_support_and_hash() -> None:
    source = (ROOT / "scripts/watch_t3_eedi_taskonly_qwen_formal.sh").read_text(encoding="utf-8")
    assert "amlt run" not in source
    assert "int(group['llm_judge_n'])==894" in source
    assert "len(rows)==894" in source
    assert "329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2" in source
    assert "touch \"$state/complete\"" in source


def test_unrestricted_t3_rejudge_uses_current_shared_rubric() -> None:
    source = (ROOT / "scripts/watch_unrestricted_qwen_t3_v2_rejudge.sh").read_text(
        encoding="utf-8"
    )
    assert "adapter_t3_eedi/predictions.jsonl" in source
    assert "base_t3_malrule/predictions.jsonl" in source
    assert "judge_formal_t3_predictions.py" in source
    assert "judge_backend']=='trapi'" in source
    assert "t3-semantic-equivalence-v2-substantive-schema" in source
    assert "judged_rows'])==1494" in source
    assert "amlt run" not in source
