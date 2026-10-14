#!/usr/bin/env python3
"""Materialize the frozen answer-C4 config from the semantic selection receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection.get("stage") != "state8_source_anchor_diagnostic_selection_v1":
        raise ValueError("Selection stage drift")
    arm = selection.get("selected_for_answer_c4_confirmation")
    if not selection.get("diagnostic_gate_pass") or arm not in {
        "full_mse1", "updater_mse1", "full_mse2"
    }:
        raise ValueError("No selected source-anchor arm")
    candidate = selection["candidates"][arm]
    artifacts = candidate.get("artifact_sha256") or {}
    runner = args.template.parents[1] / "scripts/run_state8_source_anchor_answer_c4.sh"
    replacements = {
        "RUNNER_SHA256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "SELECTED_ARM": arm,
        "SELECTED_JOB": str(candidate["job"]),
        "SELECTED_ADAPTER_SHA256": str(artifacts["adapter_model"]),
        "SELECTED_STATE_SHA256": str(artifacts["recurrent_state"]),
    }
    text = args.template.read_text()
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Template placeholder drift: {old}")
        text = text.replace(old, new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
