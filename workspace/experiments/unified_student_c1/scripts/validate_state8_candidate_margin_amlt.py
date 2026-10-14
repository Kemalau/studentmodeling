#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "amlt/feeds_state8_candidate_margin_20260813.yaml"
RUNNER = ROOT / "scripts/run_state8_candidate_margin.sh"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(raw)
    assert config["target"] == {
        "service": "manifold",
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity",
    }
    assert config["environment"]["image"] == (
        "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
    )
    jobs = config["jobs"]
    assert {job["name"] for job in jobs} == {
        "state8-candidate-margin-student-cp3000",
        "state8-candidate-margin-teacher",
    }
    for job in jobs:
        assert job["sku"] == "80G1-A100"
        assert job["sla_tier"] == "Standard"
        assert job["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] == "rankfun_aml"
        assert digest(RUNNER) in job["command"]
    assert jobs[0]["submit_args"]["env"]["AUDIT_MODE"] == "student"
    assert jobs[1]["submit_args"]["env"]["AUDIT_MODE"] == "teacher"
    forbidden = ("REQUIRED", "<...>", "latest", "password", "account_key")
    assert not any(value in raw for value in forbidden)
    print("state8 candidate-margin AMLT validation passed")


if __name__ == "__main__":
    main()
