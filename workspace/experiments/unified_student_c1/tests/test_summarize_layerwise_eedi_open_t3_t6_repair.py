from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_eedi_open_t3_t6_repair import summarize  # noqa: E402


def row(variant: str, lr: float, passed: bool) -> dict:
    t3, t6 = ((200, 600) if variant == "balanced" else (100, 700))
    return {
        "branch": "eedi_full_open_t3",
        "loss_profile": "state_path",
        "learning_rate": lr,
        "task_counts": {
            "diagnose_misconception": t3,
            "predict_error_persistence": t6,
        },
        "all_gates_pass": passed,
        "checks": {"C4": passed},
        "C4": {"D_specific": 0.18},
        "T3_semantic": {"retention_ratio": 0.95},
        "T6": {"retention_ratio": 0.90},
        "formal_screen": {"T3_format_valid_rate": 1.0},
    }


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_selects_only_full_gate_pass(tmp_path: Path) -> None:
    paths = [
        write(tmp_path / "a.json", row("balanced", 5e-7, False)),
        write(tmp_path / "b.json", row("balanced", 1e-6, True)),
        write(tmp_path / "c.json", row("strong", 5e-7, False)),
    ]
    value = summarize(paths)
    assert value["any_all_gates_pass"] is True
    assert value["selected_candidate"]["candidate_id"].endswith("balanced__lr1e-06")


def test_rejects_duplicate_arm(tmp_path: Path) -> None:
    paths = [
        write(tmp_path / "a.json", row("balanced", 5e-7, False)),
        write(tmp_path / "b.json", row("balanced", 5e-7, True)),
        write(tmp_path / "c.json", row("strong", 5e-7, False)),
    ]
    with pytest.raises(ValueError):
        summarize(paths)


def test_incomplete_retired_comparison_is_not_selectable(tmp_path: Path) -> None:
    paths = [
        write(tmp_path / "a.json", row("balanced", 5e-7, True)),
        write(tmp_path / "b.json", row("balanced", 1e-6, True)),
    ]
    value = summarize(paths)
    assert value["selection_complete"] is False
    assert value["observed_arm_count"] == 2
    assert value["missing_arms"] == [
        {"curriculum_variant": "strong", "learning_rate": 5e-7}
    ]
    assert value["any_all_gates_pass"] is True
    assert value["selected_candidate"] is None
