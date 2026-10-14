from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_equal_state_kd_open_t3_lr_scan_amlt import (  # noqa: E402
    JOBS,
    MANIFEST,
    MANIFEST_SHA256,
)
from validate_layerwise_eedi_full_lr_scan_amlt import load, validate  # noqa: E402


def test_equal_open_state_kd_lr_scan_is_matched_and_audited() -> None:
    path = ROOT / "amlt/feeds_layerwise_equal_state_kd_open_t3_lr_scan_20260803.yaml"
    value = load(path)
    validate(
        value,
        jobs_by_lr=JOBS,
        loss_profile="state_kd",
        venv_root="/tmp/studentmodeling-layerwise-equal-open-statekd-lr-venv",
        branch="equal_t3_t6_open_t3",
        train_entrypoint="scripts/train_recurrent_student_m4_open_t3_selective_kd.py",
        segment_rows="1000",
        expected_causal_pairs="50",
        diagnostic_stop_after_round="3250",
        manifest=MANIFEST,
        manifest_sha256=MANIFEST_SHA256,
    )
    environments = [job["submit_args"]["env"] for job in value["jobs"]]
    assert {env["LEARNING_RATE"] for env in environments} == {"2e-6", "5e-6"}
    assert {env["MANIFEST_SHA256"] for env in environments} == {MANIFEST_SHA256}
    assert {job["sku"] for job in value["jobs"]} == {"80G4-A100"}
