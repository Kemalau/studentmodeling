from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_cp3000_watcher_is_resumable_and_semantically_judged() -> None:
    text=(ROOT/"scripts/watch_state8_matched_checkpoint3000_semantic_rescue.sh").read_text()
    for fragment in (
        "classify_experiment_status","amlt resume","recovery_prefix","max_waves=12",
        "judge_formal_t3_predictions.py","summarize_state8_diagnosis_c4.py",
        "delta_vs_checkpoint3000","semantic_judge_authoritative",
        "validate_state8_matched_checkpoint3000_semantic_rescue_amlt.py",
        "STATE8_CP3000_CONFIG","STATE8_CP3000_BASE_EXPERIMENT",
        "STATE8_CP3000_RECOVERY_PREFIX","STATE8_CP3000_VALIDATOR",
        "select_state8_matched_cp3000_semantic_rescue.py",
        "rescue_receipt.json","state8_matched_cp3000_semantic_v1",
        "scientific_gate_pass","scientific_gate_failed",
        "*/eval/formal_t3_300/predictions.jsonl",
        "*/eval/diagnosis_c4_50/judge_predictions.jsonl",
    ):
        assert fragment in text
