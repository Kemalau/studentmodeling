#!/usr/bin/env python3
"""Fail-closed AMLT validator for the full-support Eedi state-KD LR scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_layerwise_eedi_full_lr_scan_amlt import load, validate


JOBS = {
    "layerwise-eedi-full-statekd-lr2e6-seed42": "2e-6",
    "layerwise-eedi-full-statekd-lr5e6-seed42": "5e-6",
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
        loss_profile="state_kd",
        venv_root="/tmp/studentmodeling-layerwise-eedi-full-statekd-lr-venv",
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
