from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.select_state8_strong1_rescue_source import remote_root, select


def candidate(slug: str, *, t3: float, qualified: bool = False) -> dict:
    floors = {
        "T1": .55, "T2": .20, "T3_eedi_semantic": .17,
        "T3_synthetic_semantic": .45, "T3_schema": .95, "T5": .35,
        "T6": .35, "T6_macro_f1": .30,
        "answer_C4_D_specific": .15, "answer_C4_same_rule_max": .10,
        "answer_C4_noop_max": .05, "diagnosis_C4_D_specific": .15,
        "diagnosis_C4_swap_to_donor": .50,
        "diagnosis_C4_same_rule_retention": .65,
        "diagnosis_C4_noop_retention": .75,
        "diagnosis_C4_random_stable_max": .25,
    }
    checks = {
        "answer_C4_pass": False, "diagnosis_C4_pass": False,
        "T1": .60, "T2": .30, "T3_eedi_semantic": t3,
        "T3_synthetic_semantic": .50, "T3_schema": .98, "T5": .40,
        "T6": .40, "T6_macro_f1": .35, "T6_positive_recall": .2,
        "answer_C4_D_specific": .14, "answer_C4_same_rule": .08,
        "answer_C4_noop": 0.0, "diagnosis_C4_D_specific": .14,
        "diagnosis_C4_swap_to_donor": .48,
        "diagnosis_C4_same_rule_retention": .70,
        "diagnosis_C4_noop_retention": .80,
        "diagnosis_C4_random_stable": .20,
    }
    return {
        "slug": slug, "qualified": qualified, "score": t3,
        "checks": checks, "floors": floors,
        "remote_root": remote_root(slug),
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }


def write(path: Path, rows: list[dict], passed: bool = False) -> None:
    path.write_text(json.dumps({
        "stage": "state8_strong1_widen8_joint_promotion_v1",
        "pass": passed, "selected": None, "candidates": rows,
    }))


def test_selects_smallest_normalized_deficit(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    write(path, [candidate("lr5e7_seed58", t3=.10), candidate("lr1e6_seed59", t3=.16)])
    result = select([path])
    assert result["selected"]["slug"] == "lr1e6_seed59"
    penalties = {row["slug"]: row["penalty"] for row in result["candidates"]}
    assert penalties["lr1e6_seed59"] < penalties["lr5e7_seed58"]


def test_rejects_rescue_after_qualified_bridge(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    write(path, [candidate("lr5e7_seed58", t3=.2)], passed=True)
    with pytest.raises(ValueError, match="forbidden"):
        select([path])


def test_rejects_remote_root_drift(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    row = candidate("single_lr5e7_seed60", t3=.2)
    row["remote_root"] = "/tmp/not-remote"
    write(path, [row])
    with pytest.raises(ValueError, match="Remote root drift"):
        select([path])
