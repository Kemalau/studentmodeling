#!/usr/bin/env python3
"""Fail-closed validator for the lrfix-v4 smoke evaluation-only recovery."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from validate_state8_curriculum_checkpoint_screen_amlt import IMAGE, WORKSPACE
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from scripts.validate_state8_curriculum_checkpoint_screen_amlt import IMAGE, WORKSPACE


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_state8_profile_ownership_lrfix_v4_smoke_recovery_audit.sh"
EVALUATOR = ROOT / "scripts/early_audit_state8_profile_ownership_checkpoint.py"
JOB = "state8-lrfix-v4-smoke-r2-metadatafix-evalonly-seed8801"


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b|PENDING|RUNNER_SHA256", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job cardinality/name drift")
    job = jobs[0]
    if (
        job.get("sku"), job.get("process_count_per_node"),
        job.get("mpi"), job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
        raise ValueError("Retry/duration drift")
    if (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("Identity drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        hashlib.sha256(EVALUATOR.read_bytes()).hexdigest(),
        f"bash scripts/{RUNNER.name}",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Recovery source binding drift")
    runner = RUNNER.read_text(encoding="utf-8")
    if (
        "state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_seed8801" not in runner
        or 'CHECKPOINT_ROOT="$RUN_ROOT/checkpoint-200"' not in runner
    ):
        raise ValueError("Immutable checkpoint binding missing")
    if "manifests/dbe_student_profile_v1/test.jsonl" not in runner:
        raise ValueError("DBE development audit rows missing")
    folded = runner.casefold()
    if any(fragment in folded for fragment in ("manifests/xes", "/xes_", "xes_profile_source")):
        raise ValueError("Recovery runner must not inspect XES")
    if '"official_xes_test_inspected": False' not in runner:
        raise ValueError("Recovery receipt must attest that XES stayed untouched")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
