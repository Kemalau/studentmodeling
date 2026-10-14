from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.materialize_state8_taskanchor_confirmation_amlt import materialize
from scripts.validate_state8_taskanchor_confirmation_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def hashes() -> dict[str, str]:
    return {
        "adapter_model": "a" * 64,
        "recurrent_state": "b" * 64,
        "receipt": "c" * 64,
    }


def test_materialized_confirmation_is_hash_bound_and_valid(tmp_path: Path) -> None:
    output = tmp_path / "confirmation.yaml"
    materialize(ROOT / "amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml", output, 400, hashes())
    value = yaml.safe_load(output.read_text())
    validate(value)
    assert len(value["jobs"]) == 19
    assert {job["submit_args"]["env"]["MILESTONE_STEP"] for job in value["jobs"]} == {"400"}
    assert sum("formal-shard" in job["name"] for job in value["jobs"]) == 16
    assert sum("c4-seed" in job["name"] for job in value["jobs"]) == 3


def test_confirmation_rejects_cross_job_artifact_drift(tmp_path: Path) -> None:
    output = tmp_path / "confirmation.yaml"
    materialize(ROOT / "amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml", output, 600, hashes())
    value = yaml.safe_load(output.read_text())
    value["jobs"][1]["submit_args"]["env"]["MILESTONE_STATE_SHA256"] = "d" * 64
    with pytest.raises(ValueError, match="cross-job milestone"):
        validate(value)


def test_confirmation_launchers_keep_full_support_and_semantic_c4() -> None:
    formal = (ROOT / "scripts/run_state8_taskanchor_confirmation_formal.sh").read_text()
    causal = (ROOT / "scripts/run_state8_taskanchor_confirmation_c4.sh").read_text()
    for fragment in (
        "unsharded_filtered_rows']==21676",
        '--shard-index "$SHARD_INDEX" --num-shards 16',
        "evaluate_recurrent_formal_protocols_diagnosis_first.py",
        "raw_history_visible_to_decoder",
    ):
        assert fragment in formal
    for fragment in (
        "--num-pairs 150",
        "evaluate_recurrent_diagnosis_c4_diagnosis_first.py",
        "--progress-dir",
        "--expected-adapter-sha256",
        "semantic_judgment_pending",
    ):
        assert fragment in causal

