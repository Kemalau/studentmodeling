from __future__ import annotations

import json
from pathlib import Path

from scripts.select_state8_complete_candidate import select


def write(path: Path, stage: str, slug: str, score: float, qualified: bool) -> Path:
    value = {
        "stage": stage,
        "pass": qualified,
        "selected": {"slug": slug, "score": score, "qualified": True} if qualified else None,
        "candidates": [{"slug": slug, "score": score, "qualified": qualified}],
        "exact_match_authoritative": False,
        "semantic_judge_authoritative": True,
    }
    path.write_text(json.dumps(value))
    return path


def test_selects_highest_score_and_preserves_downstream_stage(tmp_path: Path) -> None:
    bridge = write(
        tmp_path / "bridge.json",
        "state8_strong1_widen8_joint_promotion_v1",
        "single_lr5e7_seed60",
        0.45,
        True,
    )
    midpoint = write(
        tmp_path / "midpoint.json",
        "state8_midpoint_diagnosis_first_joint_promotion_v1",
        "midpoint_diagnosis_first_seed63",
        0.62,
        True,
    )
    result = select([bridge, midpoint])
    assert result["stage"] == "state8_midpoint_diagnosis_first_joint_promotion_v1"
    assert result["selected"]["slug"] == "midpoint_diagnosis_first_seed63"
    assert result["complete_selection_audit"]["selected_source"] == str(midpoint)
    assert len(result["candidates"]) == 2


def test_returns_terminal_failure_when_none_qualify(tmp_path: Path) -> None:
    failed = write(
        tmp_path / "failed.json",
        "state8_strong1_widen8_joint_promotion_v1",
        "single_lr5e7_seed60",
        0.12,
        False,
    )
    result = select([failed])
    assert result["stage"] == "state8_complete_candidate_selection_v1"
    assert result["pass"] is False
    assert result["selected"] is None


def test_exact_curriculum_can_win_complete_selection(tmp_path: Path) -> None:
    bridge = write(
        tmp_path / "bridge.json",
        "state8_strong1_widen8_joint_promotion_v1",
        "single_lr5e7_seed60",
        0.45,
        True,
    )
    exact = write(
        tmp_path / "exact.json",
        "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "exact_diagnosis_curriculum_seed65",
        0.70,
        True,
    )
    result = select([bridge, exact])
    assert result["stage"] == "state8_exact_diagnosis_curriculum_joint_promotion_v1"
    assert result["selected"]["slug"] == "exact_diagnosis_curriculum_seed65"
