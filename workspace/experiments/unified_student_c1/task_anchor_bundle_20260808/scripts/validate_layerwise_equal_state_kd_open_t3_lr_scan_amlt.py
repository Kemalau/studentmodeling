#!/usr/bin/env python3
"""Fail-closed validator for the equal-curriculum open-T3 state-KD LR scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_layerwise_eedi_full_lr_scan_amlt import load, validate


JOBS = {
    "layerwise-equal-t3-t6-open-t3-statekd-lr2e6-seed42": "2e-6",
    "layerwise-equal-t3-t6-open-t3-statekd-lr5e6-seed42": "5e-6",
}
MANIFEST = "manifests/layerwise_task_retention_equal_t3_t6_seed42_v1.json"
MANIFEST_SHA256 = "32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(
        load(args.config),
        set(args.jobs) if args.jobs else None,
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
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
