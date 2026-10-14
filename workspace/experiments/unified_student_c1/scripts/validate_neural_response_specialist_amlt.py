#!/usr/bin/env python3
"""Preflight the hash-bound T2/T4 neural specialist AMLT config."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


EXPECTED = {
    "neural-response-specialist-t2-seed42": {
        "BASELINE_PUBLIC_TASK": "T2",
        "BASELINE_TASK": "predict_response",
        "BASELINE_TRAIN_ROWS": "55569",
        "BASELINE_TEST_ROWS": "3693",
        "BASELINE_TRAIN_SHA256": "e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e",
    },
    "neural-response-specialist-t4-seed42": {
        "BASELINE_PUBLIC_TASK": "T4",
        "BASELINE_TASK": "predict_prospective_error",
        "BASELINE_TRAIN_ROWS": "78600",
        "BASELINE_TEST_ROWS": "4600",
        "BASELINE_TRAIN_SHA256": "240dc2f1a02bbb446e4d73b0301ee4da7f6716c088746f7ec17bf82934c2aa1e",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, root: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "<placeholder>" not in text and "<workspace-name>" not in text
    assert "ml-common-py3.12:latest" not in text
    value = yaml.safe_load(text)
    target = value["target"]
    assert target["name"] == "Feeds"
    assert target.get("service", "manifold") == "manifold"
    assert target["workspace_name"] in {
        "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity",
        "CS-NewsAndFeeds-Singularity",
    }
    if target["workspace_name"] == "CS-NewsAndFeeds-Singularity":
        assert target["workspace_resource_group"] == "rg-cs-newsandfeeds-singularity"
    assert value["environment"]["image"] == "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
    assert value["storage"]["student_store"]["mount_dir"] == "/mnt/amulet"
    jobs = {job["name"]: job for job in value["jobs"]}
    assert set(jobs) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        job = jobs[name]
        assert job["sku"] == "80G1-A100"
        assert job["sla_tier"] == "Standard"
        assert job["process_count_per_node"] == 1 and job["mpi"] is False
        assert job["submit_args"]["max_attempts"] == 2
        env = job["submit_args"]["env"]
        assert env["_AZUREML_SINGULARITY_JOB_UAI"] == "rankfun_aml"
        for key, expected_value in expected.items():
            assert str(env[key]) == expected_value, (name, key, env[key], expected_value)
        assert str(env["BASELINE_TRAIN_DATA"]).startswith("/mnt/amulet/studentmodeling/")
        raw_command = job["command"]
        command = "\n".join(raw_command) if isinstance(raw_command, list) else str(raw_command)
        for filename in (
            "scripts/run_neural_response_specialist.sh",
            "scripts/train_neural_response_specialist.py",
            "scripts/formal_metric_utils.py",
        ):
            assert digest(root / filename) in command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve() if args.root is not None else args.config.resolve().parents[1]
    validate(args.config, root)
    print("valid=true")


if __name__ == "__main__":
    main()
