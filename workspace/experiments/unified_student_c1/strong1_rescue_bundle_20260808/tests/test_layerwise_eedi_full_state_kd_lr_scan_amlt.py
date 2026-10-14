from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_eedi_full_state_kd_lr_scan_amlt import JOBS  # noqa: E402
from validate_layerwise_eedi_full_lr_scan_amlt import validate  # noqa: E402


def test_full_eedi_state_kd_scan_is_matched_and_fail_closed() -> None:
    path = ROOT / "amlt/feeds_layerwise_eedi_full_state_kd_lr_scan_20260802.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate(
        value,
        jobs_by_lr=JOBS,
        loss_profile="state_kd",
        venv_root="/tmp/studentmodeling-layerwise-eedi-full-statekd-lr-venv",
    )
    jobs = value["jobs"]
    assert {job["submit_args"]["env"]["LEARNING_RATE"] for job in jobs} == {
        "2e-6",
        "5e-6",
    }
    assert {job["submit_args"]["env"]["LOSS_PROFILE"] for job in jobs} == {
        "state_kd"
    }
