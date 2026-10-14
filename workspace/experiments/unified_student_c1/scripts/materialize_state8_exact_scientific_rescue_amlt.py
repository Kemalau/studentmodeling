#!/usr/bin/env python3
"""Materialize only the exact State8 rescue arms justified by failed gates."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml


EXPECTED_STAGE = "state8_exact_diagnosis_curriculum_joint_promotion_v1"
JOB_BY_MODE = {
    "semantic_preserve": "state8-exact-rescue-semantic-preserve-seed66",
    "causal_taskanchor": "state8-exact-rescue-causal-taskanchor-seed67",
    "retention_preserve": "state8-exact-rescue-retention-preserve-seed68",
}


def load_failed_selection(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") != EXPECTED_STAGE:
        raise ValueError("Exact selection stage drift")
    if value.get("pass") is not False or value.get("selected") is not None:
        raise ValueError("Scientific rescue is forbidden after a qualified exact model")
    candidates = value.get("candidates") or []
    if len(candidates) != 1 or candidates[0].get("qualified") is not False:
        raise ValueError("Expected one failed exact candidate")
    artifacts = candidates[0].get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
        not isinstance(digest, str) or len(digest) != 64
        for digest in artifacts.values()
    ):
        raise ValueError("Exact artifact identity drift")
    return value


def required_modes(candidate: dict[str, Any]) -> list[str]:
    checks = candidate.get("checks") or {}
    floors = candidate.get("floors") or {}
    semantic_low = any(
        float(checks.get(key, -1.0)) < float(floors[key])
        for key in ("T3_eedi_semantic", "T3_synthetic_semantic", "T3_schema")
    )
    retention_low = any(
        float(checks.get(key, -1.0)) < float(floors[key])
        for key in ("T1", "T2", "T5", "T6", "T6_macro_f1")
    ) or float(checks.get("T6_positive_recall", 0.0)) <= 0.0
    answer_bad = (
        checks.get("answer_C4_pass") is not True
        or float(checks.get("answer_C4_D_specific", -1.0))
        < float(floors["answer_C4_D_specific"])
        or float(checks.get("answer_C4_same_rule", 1.0))
        > float(floors["answer_C4_same_rule_max"])
        or float(checks.get("answer_C4_noop", 1.0))
        > float(floors["answer_C4_noop_max"])
    )
    diagnosis_bad = (
        checks.get("diagnosis_C4_pass") is not True
        or float(checks.get("diagnosis_C4_D_specific", -1.0))
        < float(floors["diagnosis_C4_D_specific"])
        or float(checks.get("diagnosis_C4_swap_to_donor", -1.0))
        < float(floors["diagnosis_C4_swap_to_donor"])
        or float(checks.get("diagnosis_C4_same_rule_retention", -1.0))
        < float(floors["diagnosis_C4_same_rule_retention"])
        or float(checks.get("diagnosis_C4_noop_retention", -1.0))
        < float(floors["diagnosis_C4_noop_retention"])
        or float(checks.get("diagnosis_C4_random_stable", 1.0))
        > float(floors["diagnosis_C4_random_stable_max"])
    )
    modes: list[str] = []
    if semantic_low:
        modes.append("semantic_preserve")
    if retention_low:
        modes.append("retention_preserve")
    if answer_bad or diagnosis_bad:
        modes.append("causal_taskanchor")
    if not modes:
        raise ValueError("Failed exact model has no diagnosed repair dimension")
    return modes


def materialize(source: Path, selection: Path, output: Path) -> list[str]:
    selected = load_failed_selection(selection)["candidates"][0]
    modes = required_modes(selected)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    jobs = []
    for job in config.get("jobs") or []:
        env = (job.get("submit_args") or {}).get("env") or {}
        mode = str(env.get("STATE8_RESCUE_MODE") or "")
        if mode not in modes:
            continue
        expected_name = JOB_BY_MODE[mode]
        if job.get("name") != expected_name:
            raise ValueError(f"Rescue job/mode drift: {job.get('name')}/{mode}")
        rendered = copy.deepcopy(job)
        rendered_env = rendered["submit_args"]["env"]
        rendered_env["EXPECTED_STUDENT_ADAPTER_SHA256"] = selected[
            "artifact_sha256"
        ]["adapter_model"]
        rendered_env["EXPECTED_STUDENT_STATE_SHA256"] = selected[
            "artifact_sha256"
        ]["recurrent_state"]
        jobs.append(rendered)
    if {str(job["name"]) for job in jobs} != {JOB_BY_MODE[mode] for mode in modes}:
        raise ValueError("Conditional rescue job set drift")
    config["jobs"] = jobs
    config["description"] += f"; selected_modes={','.join(modes)}"
    rendered_text = yaml.safe_dump(config, sort_keys=False)
    if "REQUIRED_AT_CONDITIONAL_MATERIALIZATION" in rendered_text:
        raise ValueError("Unresolved exact artifact placeholders")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(rendered_text, encoding="utf-8")
    os.replace(temporary, output)
    return modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = materialize(args.source, args.selection, args.output)
    print(f"output={args.output} modes={','.join(modes)}")


if __name__ == "__main__":
    main()
