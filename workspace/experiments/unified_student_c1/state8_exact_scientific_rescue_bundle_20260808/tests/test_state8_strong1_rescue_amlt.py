from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.materialize_state8_strong1_rescue_amlt import materialize
from scripts.validate_state8_strong1_rescue_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def selection(path: Path, slug: str = "lr5e7_seed58") -> Path:
    suffix = slug + ("_single_ga4" if slug == "single_lr5e7_seed60" else "")
    path.write_text(json.dumps({
        "stage": "state8_strong1_rescue_source_selection_v1",
        "selected": {
            "slug": slug,
            "remote_root": (
                "/mnt/amulet/studentmodeling/diagnostics/"
                "state8_qualified_teacher_curriculum_highkd_v1_immutable_"
                "taxonomy_semantic_v1_strong1_widen8_" + suffix
            ),
            "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
        },
    }))
    return path


def test_materialized_rescue_is_hash_bound_and_single_gpu(tmp_path: Path) -> None:
    source = selection(tmp_path / "selection.json")
    config = tmp_path / "rescue.yaml"
    materialize(source, config)
    validate(config, source, ROOT / "strong1_rescue_bundle_20260808")
    value = yaml.safe_load(config.read_text())
    assert len(value["jobs"]) == 2
    assert all(job["sku"] == "80G1-A100" and job["mpi"] is False for job in value["jobs"])
    assert {job["submit_args"]["env"]["EXPECTED_STUDENT_ADAPTER_SHA256"] for job in value["jobs"]} == {"a" * 64}
    by_name = {job["name"]: job for job in value["jobs"]}
    semantic = by_name["state8-strong1-rescue-taxonomy_joint-seed61"]["submit_args"]["env"]
    assert semantic["TRAIN_ENTRYPOINT"] == "scripts/train_state8_diagnosis_first_causal_repair.py"
    assert semantic["STATE8_REPAIR_LABEL_WEIGHT"] == "1.0"
    assert semantic["STATE8_CAUSAL_LABEL_WEIGHT"] == "1.0"
    causal = by_name["state8-strong1-rescue-taskanchor_c4strong-seed62"]["submit_args"]["env"]
    assert causal["STATE8_REPAIR_LABEL_WEIGHT"] == "4.0"


def test_validator_rejects_source_hash_drift(tmp_path: Path) -> None:
    source = selection(tmp_path / "selection.json", "single_lr5e7_seed60")
    config = tmp_path / "rescue.yaml"; materialize(source, config)
    value = yaml.safe_load(config.read_text())
    value["jobs"][0]["submit_args"]["env"]["EXPECTED_STUDENT_STATE_SHA256"] = "c" * 64
    config.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError, match="Environment drift"):
        validate(config, source, ROOT / "strong1_rescue_bundle_20260808")


def test_rescue_controller_recovers_amlt_killed_jobs() -> None:
    source = (ROOT / "scripts/supervise_state8_strong1_rescue.sh").read_text()
    assert source.count("'killed'") >= 2


def test_rescue_starts_after_first_failed_joint_selection() -> None:
    source = (ROOT / "scripts/supervise_state8_strong1_rescue.sh").read_text()
    assert "failed_selection_available=false" in source
    assert "selection_fails" in source
    assert "x['pass'] is False and x.get('selected') is None" in source
    assert '[[ "$failed_selection_available" == true ]] && break' in source
    assert "wait until both primary controllers are terminal" not in source
    assert "qualified primary bridge exists; rescue not required" in source
