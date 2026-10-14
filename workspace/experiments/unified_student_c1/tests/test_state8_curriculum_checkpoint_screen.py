import json
import copy
from pathlib import Path

import pytest
import yaml

from scripts.materialize_state8_curriculum_checkpoint_screen import materialize
from scripts.validate_state8_curriculum_checkpoint_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_curriculum_checkpoint_screen_20260812.yaml"


def receipt(tmp_path: Path, step: int) -> Path:
    path = tmp_path / f"checkpoint-{step}.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "step": step,
                "sha256": {
                    "adapter_model": "a" * 64,
                    "recurrent_state": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_materializes_all_curriculum_screen_conditions(tmp_path: Path) -> None:
    for arm in ("generic", "current_query"):
        for step in (2000, 3000, 4000, 5000):
            output = tmp_path / f"{arm}-{step}.yaml"
            materialize(TEMPLATE, receipt(tmp_path, step), arm, step, output)
            validate(output, arm, step)


def test_accepts_exact_amlt_11142_dump_code_binding(tmp_path: Path) -> None:
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, receipt(tmp_path, 2000), "generic", 2000, output)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    config["version"] = "11.14.2"
    config["code"].pop("local_dir")
    config["code"]["storage_id"] = "default"
    config["storage"]["default"] = {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/default",
        "is_output": True,
    }
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    validate(output, "generic", 2000)

    for mutation in ("missing_version", "foreign_storage"):
        invalid = copy.deepcopy(config)
        if mutation == "missing_version":
            invalid.pop("version")
        else:
            invalid["storage"]["default"]["container_name"] = "foreign"
        path = tmp_path / f"{mutation}.yaml"
        path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="Code root drift"):
            validate(path, "generic", 2000)


def test_runner_is_read_only_and_covers_unified_protocols() -> None:
    text = (ROOT / "scripts/run_state8_curriculum_checkpoint_screen.sh").read_text()
    for fragment in (
        "formal_all_tasks",
        "answer_c4_50",
        "malrule_diagnosis_c4_20",
        "eedi_same_question_c4_25",
        "audit_state8_curriculum_checkpoint.py",
        "CHECKPOINT_FAMILY",
        "raw_history_visible_to_decoder",
    ):
        assert fragment in text
    assert "optimizer.step" not in text
    assert "--init-adapter" not in text


def test_watcher_has_quota_guard_recovery_and_semantic_judges() -> None:
    text = (ROOT / "scripts/watch_state8_curriculum_checkpoint_screen.sh").read_text()
    for fragment in (
        "count_active_a100_requests.py",
        "available_user_slots",
        "max_recovery=8",
        "amlt resume",
        "--judge-backend trapi",
        "summarize_state8_curriculum_checkpoint_screen.py",
        "checkpoint-${step}.snapshot_complete",
    ):
        assert fragment in text
