from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_state8_exact_scientific_rescue_amlt import materialize
from scripts.validate_state8_exact_scientific_rescue_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def candidate(checks: dict[str, object]) -> dict[str, object]:
    defaults = {
        "T1": .6, "T2": .3, "T3_eedi_semantic": .2,
        "T3_synthetic_semantic": .5, "T3_schema": .98, "T5": .5,
        "T6": .5, "T6_macro_f1": .4, "T6_positive_recall": .2,
        "answer_C4_pass": True, "answer_C4_D_specific": .2,
        "answer_C4_same_rule": .05, "answer_C4_noop": .02,
        "diagnosis_C4_pass": True, "diagnosis_C4_D_specific": .2,
        "diagnosis_C4_swap_to_donor": .6,
        "diagnosis_C4_same_rule_retention": .7,
        "diagnosis_C4_noop_retention": .8,
        "diagnosis_C4_random_stable": .1,
    }
    defaults.update(checks)
    return {
        "slug": "exact_diagnosis_curriculum_seed65",
        "qualified": False,
        "score": 0.0,
        "checks": defaults,
        "floors": {
            "T1": .55, "T2": .2, "T3_eedi_semantic": .17,
            "T3_synthetic_semantic": .45, "T3_schema": .95, "T5": .35,
            "T6": .35, "T6_macro_f1": .3,
            "answer_C4_D_specific": .15, "answer_C4_same_rule_max": .1,
            "answer_C4_noop_max": .05, "diagnosis_C4_D_specific": .15,
            "diagnosis_C4_swap_to_donor": .5,
            "diagnosis_C4_same_rule_retention": .65,
            "diagnosis_C4_noop_retention": .75,
            "diagnosis_C4_random_stable_max": .25,
        },
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }


def selection(row: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "pass": False, "selected": None, "candidates": [row],
    }


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ({"T3_eedi_semantic": .05}, {"state8-exact-rescue-semantic-preserve-seed66"}),
        ({"T6_positive_recall": 0.0}, {"state8-exact-rescue-retention-preserve-seed68"}),
        ({"diagnosis_C4_pass": False}, {"state8-exact-rescue-causal-taskanchor-seed67"}),
        (
            {"T3_schema": .5, "answer_C4_pass": False},
            {
                "state8-exact-rescue-semantic-preserve-seed66",
                "state8-exact-rescue-causal-taskanchor-seed67",
            },
        ),
    ],
)
def test_materializes_only_diagnosed_arms(tmp_path: Path, checks, expected) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection(candidate(checks))))
    output = tmp_path / "rendered.yaml"
    materialize(
        ROOT / "amlt/feeds_state8_exact_scientific_rescue_20260808.yaml",
        selection_path,
        output,
    )
    import yaml
    config = yaml.safe_load(output.read_text())
    assert {job["name"] for job in config["jobs"]} == expected
    validate(
        output,
        selection_path,
        ROOT / "state8_exact_scientific_rescue_bundle_20260808",
    )


def test_refuses_rescue_after_exact_model_passes(tmp_path: Path) -> None:
    value = selection(candidate({"T3_schema": .5}))
    value["pass"] = True
    value["selected"] = value["candidates"][0]
    path = tmp_path / "selection.json"; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="forbidden"):
        materialize(
            ROOT / "amlt/feeds_state8_exact_scientific_rescue_20260808.yaml",
            path,
            tmp_path / "rendered.yaml",
        )
