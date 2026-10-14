#!/usr/bin/env python3
"""Derive schedulable matched-State8 final-fit configs from the approved template."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


SOURCE_SHA256 = "40c39e638fdc4869e743488782dcd7e4eb8d36a3e1e799f91ba1aa3130a1a558"
JOB = "state8-diagnosis-first-matched-seed67-unified-finalfit-seed42"
TOPOLOGIES = {
    "world4": ("80G4-A100", "run_state8_unified_final_fit_world4.sh"),
    "world2": ("80G2-A100", "run_state8_unified_final_fit_world2.sh"),
}


def materialize(source: Path, output: Path, topology: str) -> None:
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise ValueError(f"approved final-fit config hash drift: {observed}")
    config = yaml.safe_load(raw)
    matches = [job for job in config.get("jobs") or [] if job.get("name") == JOB]
    if len(matches) != 1:
        raise ValueError(f"Expected one matched final-fit job, observed {len(matches)}")
    job = matches[0]
    if job.get("sku") != "2x80G8-A100-IB-NvLink":
        raise ValueError("source matched topology drift")
    if (job.get("submit_args") or {}).get("env") != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-finalfit-venv",
        "FINALFIT_VARIANT": "diagnosis_first_matched_seed67",
    }:
        raise ValueError("source matched environment drift")
    sku, launcher = TOPOLOGIES[topology]
    job["sku"] = sku
    job["command"] = str(job["command"]).replace(
        "bash scripts/run_state8_unified_final_fit.sh",
        f"bash scripts/{launcher}",
    )
    if str(job["command"]).count(f"bash scripts/{launcher}") != 1:
        raise ValueError("matched launcher replacement drift")
    job["submit_args"]["max_run_duration_seconds"] = 43200
    config["jobs"] = [job]
    config["description"] = (
        "Diagnosis-first matched State8 exact 259845-row final fit; "
        f"{topology}; global batch 16; strict state-only"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), required=True)
    args = parser.parse_args()
    materialize(args.source, args.output, args.topology)
    print(args.output)


if __name__ == "__main__":
    main()
