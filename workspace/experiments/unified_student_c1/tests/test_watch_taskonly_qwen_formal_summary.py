from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_watcher_publishes_integrated_formal_matrix() -> None:
    source = (ROOT / "scripts/watch_taskonly_qwen_formal_summary.sh").read_text(encoding="utf-8")
    assert "scripts/build_formal_evaluation_matrix.py" in source
    assert "scripts/audit_taskonly_target_exposure.py" in source
    assert "unified_recurrent_full259845_u4_v1_final_fit.json" in source
    assert "target_exposure_audit.json" in source
    assert '--taskonly-summary "$state/summary/summary.json"' in source
    assert '--output-dir "$state/formal_specialist_v3"' in source
    assert '--t3-root results/formal_specialist_v2/unrestricted_qwen_t3_v2' in source
    assert '--t3-root results/formal_specialist_v1/t3_llm_judge_raw_amlt' not in source
    assert 'unified_t3_v2/complete' in source
    assert "scripts/compare_taskonly_unified_paired.py" in source
    assert "studentmodeling-qwen3-32b-formal-generation-recovery1-4a100-20260720" in source
    assert "adapter_predict_response/predictions_redacted.jsonl" in source
    assert "adapter_predict_prospective_error/predictions_redacted.jsonl" in source
    assert "for task in T1 T2 T3 T4 T5" in source
    assert "taskonly_unified_paired_full_v1" in source
    assert '--paired-summary "$state/paired/complete.json"' in source
    assert "pending download of unified per-example generation predictions" not in source
