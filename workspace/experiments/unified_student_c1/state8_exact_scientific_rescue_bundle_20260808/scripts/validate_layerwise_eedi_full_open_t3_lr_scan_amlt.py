#!/usr/bin/env python3
"""Fail-closed AMLT validator for the open-T3 full-Eedi LR scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_layerwise_eedi_full_lr_scan_amlt import load, validate


JOBS = {
    "layerwise-eedi-full-open-t3-lr2e6-seed42": "2e-6",
    "layerwise-eedi-full-open-t3-lr5e6-seed42": "5e-6",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(
        load(args.config),
        set(args.jobs) if args.jobs else None,
        jobs_by_lr=JOBS,
        loss_profile="state_path",
        venv_root="/tmp/studentmodeling-layerwise-eedi-full-open-t3-lr-venv",
        branch="eedi_full_open_t3",
        train_entrypoint="scripts/train_recurrent_student_m4_open_t3.py",
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
