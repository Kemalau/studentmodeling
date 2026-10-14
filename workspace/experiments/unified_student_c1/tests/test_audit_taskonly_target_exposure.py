from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_taskonly_target_exposure import TASKS, audit  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    unified_rows = []
    counts = {"rows": len(TASKS)}
    taskonly_tasks = {}
    taskonly_root = tmp_path / "taskonly"
    for index, task in enumerate(TASKS):
        row = {
            "example_id": f"example-{index}",
            "task": task,
            "target": f"target-{index}",
            "target_label": f"label-{index}",
        }
        unified_rows.append(row)
        write_jsonl(taskonly_root / f"{task}.jsonl", [row])
        counts[f"task/{task}"] = 1
        taskonly_tasks[task] = {"train_rows": 1}
    unified = tmp_path / "unified.jsonl"
    write_jsonl(unified, unified_rows)
    unified_manifest = tmp_path / "unified_manifest.json"
    unified_manifest.write_text(json.dumps({"counts": counts}), encoding="utf-8")
    taskonly_manifest = tmp_path / "taskonly_manifest.json"
    taskonly_manifest.write_text(json.dumps({"tasks": taskonly_tasks}), encoding="utf-8")
    return unified, unified_manifest, taskonly_root, taskonly_manifest


def test_exact_target_exposure_parity(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    value = audit(*args)
    assert value["target_task_exposure_exact_match"] is True
    assert value["total_unified_rows"] == len(TASKS)
    assert all(row["target_supervision_equal"] for row in value["tasks"].values())


def test_rejects_same_id_with_different_target(tmp_path: Path) -> None:
    unified, unified_manifest, taskonly_root, taskonly_manifest = fixture(tmp_path)
    path = taskonly_root / "predict_correctness.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["target"] = "changed"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="exposure parity failed"):
        audit(unified, unified_manifest, taskonly_root, taskonly_manifest)
