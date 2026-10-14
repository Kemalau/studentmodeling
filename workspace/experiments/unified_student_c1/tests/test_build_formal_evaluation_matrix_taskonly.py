from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_formal_evaluation_matrix import (  # noqa: E402
    build_rows,
    markdown,
    paired_summary,
    taskonly_row,
)


def summary() -> dict:
    counts = {"T1": 10913, "T2": 3693, "T3": 1494, "T4": 4600, "T5": 976}
    cells = {
        key: {"primary": 0.5, "n": count, "auroc": 0.6}
        for key, count in counts.items()
    }
    cells["T3"]["judge_backend"] = "trapi"
    cells["T3"]["judge_identity"] = "redmond/interactive:gpt-4o_2024-11-20"
    cells["T4"]["correctness_accuracy"] = 0.7
    return {
        "stage": "taskonly_qwen_formal_summary_v1",
        "status": "complete",
        "same_backbone_as_unified": True,
        "test_used_for_training": False,
        "cells": cells,
    }


def test_taskonly_row_uses_full_frozen_denominators(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary()), encoding="utf-8")
    row = taskonly_row(path)
    assert row["model"] == "Task-only Qwen3-32B QLoRA"
    assert row["T1"]["n"] == 10913
    assert row["T3"]["n"] == 1494
    assert row["T3"]["judge_backend"] == "trapi"
    assert row["T3"]["judge_identity"] == "redmond/interactive:gpt-4o_2024-11-20"
    assert row["T4"]["secondary_name"] == "CorrAcc"


def test_taskonly_row_rejects_non_trapi_t3(tmp_path: Path) -> None:
    value = summary(); value["cells"]["T3"]["judge_backend"] = "responses"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="TRAPI"):
        taskonly_row(path)


def test_current_judged_t3_artifacts_rebuild_existing_formal_rows() -> None:
    rows, result_summary = build_rows(
        ROOT / "results/formal_specialist_v1/qwen_amlt",
        ROOT / "results/formal_specialist_v1/qwen_amlt",
        ROOT / "results/formal_specialist_v1",
        ROOT / "results/formal_specialist_v1/t3_llm_judge_full",
    )
    unified = next(row for row in rows if row["model"] == "Unified Qwen3-32B QLoRA")
    assert unified["T3"]["n"] == 1494
    assert unified["T3"]["value"] == pytest.approx(0.4109772423025435)
    assert unified["T3"]["judge_backend"] == "responses"
    assert unified["T3"]["judge_identity"] == "gpt-5.5"
    assert "identity `gpt-5.5`" in markdown(rows, result_summary)


def test_markdown_warns_when_generative_t3_judges_differ(tmp_path: Path) -> None:
    rows, result_summary = build_rows(
        ROOT / "results/formal_specialist_v1/qwen_amlt",
        ROOT / "results/formal_specialist_v1/qwen_amlt",
        ROOT / "results/formal_specialist_v1",
        ROOT / "results/formal_specialist_v1/t3_llm_judge_full",
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary()), encoding="utf-8")
    rows.insert(2, taskonly_row(path))
    text = markdown(rows, result_summary)
    assert "different blinded semantic judges" in text
    assert "responses:gpt-5.5" in text
    assert "trapi:redmond/interactive:gpt-4o_2024-11-20" in text


def test_paired_summary_reports_unified_minus_taskonly_for_all_tasks(tmp_path: Path) -> None:
    tasks = {}
    for task in ("T1", "T2", "T3", "T4", "T5"):
        tasks[task] = {
            "public_task": task,
            "n": 10,
            "taskonly_minus_unified": -0.1,
            "paired_bootstrap_95ci": [-0.2, 0.0],
            "mcnemar_exact_two_sided_p": 0.05,
            "example_id_set_sha256": "a" * 64,
        }
    path = tmp_path / "paired.json"
    path.write_text(json.dumps({
        "stage": "taskonly_unified_paired_full_v1",
        "status": "complete",
        "tasks": tasks,
    }), encoding="utf-8")
    result = paired_summary(path)
    assert set(result) == {"T1", "T2", "T3", "T4", "T5"}
    assert result["T2"]["unified_minus_taskonly"] == pytest.approx(0.1)
    assert result["T2"]["paired_bootstrap_95ci"] == pytest.approx([0.0, 0.2])

    rows, result_summary = build_rows(
        ROOT / "results/formal_specialist_v1/qwen_amlt",
        ROOT / "results/formal_specialist_v1/qwen_amlt",
        ROOT / "results/formal_specialist_v1",
        ROOT / "results/formal_specialist_v1/t3_llm_judge_full",
    )
    result_summary["unified_minus_taskonly_paired"] = result
    text = markdown(rows, result_summary)
    assert "Unified minus same-backbone task-only Qwen (paired)" in text
    assert "| T2 | +0.100 | [+0.000, +0.200] |" in text
