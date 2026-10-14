from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_final_eval_amlt import load, validate  # noqa: E402
from publish_state8_final_checkpoint import (  # noqa: E402
    DIAGNOSIS_FIRST_VARIANTS,
    SAME_QUESTION_VARIANTS,
)


def config() -> dict:
    return load(ROOT / "amlt/feeds_state8_final_eval_20260805.yaml")


def test_complete_config_passes() -> None:
    value = config()
    validate(value)
    assert len(value["jobs"]) == 19


def test_single_recovery_job_passes() -> None:
    value = config()
    value["jobs"] = [value["jobs"][7]]
    validate(value, {"state8-final-formal-shard07"})


def test_task_anchor_uses_taxonomy_aware_final_evaluators() -> None:
    formal = (ROOT / "scripts/run_state8_final_formal_shard.sh").read_text()
    causal = (ROOT / "scripts/run_state8_final_c4.sh").read_text()
    for source in (formal, causal):
        assert "task_anchor_base" in source
        assert "task_anchor_c4strong" in source
        assert "state8_taxonomy_contract.py" in source


def test_rejects_duplicate_formal_coordinate() -> None:
    value = copy.deepcopy(config())
    value["jobs"][1]["submit_args"]["env"]["SHARD_INDEX"] = "0"
    with pytest.raises(ValueError, match="shard"):
        validate(value)


def test_rejects_non_single_gpu_job() -> None:
    value = config()
    value["jobs"][0]["sku"] = "80G4-A100"
    with pytest.raises(ValueError, match="Topology"):
        validate(value)


def test_taxonomy_selected_checkpoint_keeps_taxonomy_evaluators() -> None:
    formal = (ROOT / "scripts/run_state8_final_formal_shard.sh").read_text()
    causal = (ROOT / "scripts/run_state8_final_c4.sh").read_text()
    assert "evaluate_recurrent_formal_protocols_taxonomy.py" in formal
    assert "evaluate_recurrent_diagnosis_c4_taxonomy.py" in causal
    assert "highcausal_taxonomy" in formal
    assert "highcausal_taxonomy" in causal


def test_retention_rescue_keeps_diagnosis_first_receipt_contract() -> None:
    formal = (ROOT / "scripts/run_state8_final_formal_shard.sh").read_text()
    causal = (ROOT / "scripts/run_state8_final_c4.sh").read_text()
    validator = (ROOT / "scripts/validate_state8_final_eval.py").read_text()
    variant = "strong1_exact_rescue_retention_preserve_seed68"
    assert variant in formal
    assert variant in causal
    assert variant in validator


def test_every_published_diagnosis_first_variant_reaches_final_evaluation() -> None:
    formal = (ROOT / "scripts/run_state8_final_formal_shard.sh").read_text()
    causal = (ROOT / "scripts/run_state8_final_c4.sh").read_text()
    gate = (ROOT / "scripts/validate_state8_final_eval.py").read_text()
    amlt_validator = (ROOT / "scripts/validate_state8_final_eval_amlt.py").read_text()
    same_question = SAME_QUESTION_VARIANTS
    for variant in DIAGNOSIS_FIRST_VARIANTS - same_question:
        assert variant in formal
        assert variant in causal
        assert variant in gate
        assert variant in amlt_validator
    specialized_formal = (
        ROOT / "scripts/render_state8_eedi_same_question_final_formal_shard.py"
    ).read_text()
    specialized_causal = (
        ROOT / "scripts/render_state8_eedi_same_question_final_c4.py"
    ).read_text()
    specialized_gate = (ROOT / "scripts/validate_state8_final_eval.py").read_text()
    specialized_amlt = (
        ROOT / "scripts/validate_state8_eedi_same_question_final_eval_amlt.py"
    ).read_text()
    for variant in same_question:
        assert variant in specialized_gate
        assert variant in specialized_formal
        assert variant in specialized_causal
    for source in (specialized_formal, specialized_causal):
        assert "same-question" in source.lower()
    assert "state8_eedi_same_question" in specialized_amlt
