#!/usr/bin/env python3
"""Derive a two-A100 schedulability fallback from the approved world4 config."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


SOURCE_SHA256 = "fac0563d10ffeecdcef4020a68af6bc2165342da5716a94dde1302027834a368"


def materialize(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise ValueError(f"approved world4 config hash drift: {observed}")
    config = yaml.safe_load(raw)
    jobs = config.get("jobs") or []
    if len(jobs) != 10:
        raise ValueError(f"Expected ten mutually exclusive variants, observed {len(jobs)}")
    for job in jobs:
        if job.get("sku") != "80G4-A100":
            raise ValueError("source topology drift")
        job["sku"] = "80G2-A100"
        command = str(job.get("command") or "")
        old = "bash scripts/run_state8_unified_final_fit_world4.sh"
        if command.count(old) != 1:
            raise ValueError(f"source launcher drift for {job.get('name')}")
        job["command"] = command.replace(
            old, "bash scripts/run_state8_unified_final_fit_world2.sh"
        )
    config["description"] = (
        "Selected strong1-to-State8 exact one-epoch unified final fit; "
        "two-A100 global-batch16 schedulability fallback"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
