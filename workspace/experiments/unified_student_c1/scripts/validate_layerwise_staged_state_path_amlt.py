#!/usr/bin/env python3
"""Fail-closed validator for staged curriculum plus state-path loss."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_layerwise_staged_task_retention_amlt import load, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(
        load(args.config),
        job_name="layerwise-staged-t3-t6-state-path-seed42",
        venv_root="/tmp/studentmodeling-layerwise-staged-statepath-venv",
        loss_profile="state_path",
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
