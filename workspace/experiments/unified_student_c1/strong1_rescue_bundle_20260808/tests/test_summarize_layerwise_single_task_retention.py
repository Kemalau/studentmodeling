import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_single_task_retention import validate_open_t3_training_view  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def common(stage: str) -> dict:
    return {
        "stage": stage,
        "scope": "train_jsonl_only",
        "changed_malrule_t3_rows": 100,
        "causal_rows_modified": False,
        "validation_rows_modified": False,
    }


def test_state_path_open_t3_receipt_is_accepted(tmp_path: Path) -> None:
    value = {
        **common("open_t3_training_view_v1"),
        "behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
    }
    write(tmp_path / "metrics/open_t3_training_view.json", value)
    assert validate_open_t3_training_view(
        tmp_path,
        {"training_view": "open_t3_no_candidate_enumeration"},
        "state_path",
    )


def test_selective_kd_open_t3_receipt_is_accepted(tmp_path: Path) -> None:
    value = {
        **common("open_t3_selective_kd_training_view_v1"),
        "global_behavior_kd_weight": 0.5,
        "excluded_behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "excluded_history_rows": 100,
        "retained_history_rows": 4900,
    }
    write(tmp_path / "metrics/open_t3_selective_kd_training_view.json", value)
    assert validate_open_t3_training_view(
        tmp_path, {"training_view": "open_t3_selective_kd"}, "state_kd"
    )


def test_selective_kd_cannot_be_misclassified_as_plain_open_t3(tmp_path: Path) -> None:
    value = {
        **common("open_t3_selective_kd_training_view_v1"),
        "global_behavior_kd_weight": 0.5,
        "excluded_behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "excluded_history_rows": 100,
        "retained_history_rows": 4900,
    }
    write(tmp_path / "metrics/open_t3_selective_kd_training_view.json", value)
    assert not validate_open_t3_training_view(
        tmp_path,
        {"training_view": "open_t3_no_candidate_enumeration"},
        "state_path",
    )


def test_selective_kd_requires_both_excluded_and_retained_rows(tmp_path: Path) -> None:
    value = {
        **common("open_t3_selective_kd_training_view_v1"),
        "global_behavior_kd_weight": 0.5,
        "excluded_behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "excluded_history_rows": 100,
        "retained_history_rows": 0,
    }
    write(tmp_path / "metrics/open_t3_selective_kd_training_view.json", value)
    assert not validate_open_t3_training_view(
        tmp_path, {"training_view": "open_t3_selective_kd"}, "state_kd"
    )
