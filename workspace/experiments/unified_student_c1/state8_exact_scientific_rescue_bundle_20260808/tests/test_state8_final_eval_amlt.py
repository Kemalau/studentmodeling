from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_final_eval_amlt import load, validate  # noqa: E402


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
