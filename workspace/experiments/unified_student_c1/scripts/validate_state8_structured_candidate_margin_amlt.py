#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "amlt/feeds_state8_structured_candidate_margin_20260813.yaml"
RUNNER = ROOT / "scripts/run_state8_structured_candidate_margin.sh"


def main() -> None:
    raw = CONFIG.read_text(encoding="utf-8")
    value = yaml.safe_load(raw)
    assert value["target"]["name"] == "Feeds"
    assert value["target"]["service"] == "manifold"
    assert value["environment"]["image"] == (
        "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
    )
    jobs = value["jobs"]
    assert len(jobs) == 1
    job = jobs[0]
    assert job["name"] == "state8-structured-candidate-margin-cp3000"
    assert job["sku"] == "80G1-A100" and job["sla_tier"] == "Standard"
    assert job["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] == "rankfun_aml"
    assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() in job["command"]
    assert not any(token in raw for token in ("REQUIRED", "<...>", "latest", "password"))
    print("state8 structured candidate-margin AMLT validation passed")


if __name__ == "__main__":
    main()
