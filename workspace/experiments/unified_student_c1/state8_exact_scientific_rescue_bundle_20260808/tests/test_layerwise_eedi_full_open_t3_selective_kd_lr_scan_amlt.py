from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_eedi_full_lr_scan_amlt import validate  # noqa: E402
from validate_layerwise_eedi_full_open_t3_selective_kd_lr_scan_amlt import (  # noqa: E402
    JOBS,
)


def test_selective_kd_open_t3_scan_is_matched_and_audited() -> None:
    path = ROOT / "amlt/feeds_layerwise_eedi_full_open_t3_selective_kd_lr_scan_20260802.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate(
        value,
        jobs_by_lr=JOBS,
        loss_profile="state_kd",
        venv_root=(
            "/tmp/studentmodeling-layerwise-eedi-full-open-t3-selective-kd-lr-venv"
        ),
        branch="eedi_full_open_t3",
        train_entrypoint="scripts/train_recurrent_student_m4_open_t3_selective_kd.py",
    )
    environments = [job["submit_args"]["env"] for job in value["jobs"]]
    assert {env["LOSS_PROFILE"] for env in environments} == {"state_kd"}
    assert {env["TRAIN_ENTRYPOINT"] for env in environments} == {
        "scripts/train_recurrent_student_m4_open_t3_selective_kd.py"
    }
