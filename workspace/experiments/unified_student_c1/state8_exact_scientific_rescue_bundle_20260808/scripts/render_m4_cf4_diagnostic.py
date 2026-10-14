#!/usr/bin/env python3
"""Render a balanced multi-query (CF4) recurrent causal diagnostic.

The previous recurrent curriculum paired every donor state with only one
recipient query.  This diagnostic keeps the frozen test gate untouched while
forcing each accepted donor state to supervise two to four distinct training
queries.  Two objective modes run in parallel so data attribution is not
confounded with another blind weight sweep.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


SOURCE = "amlt/feeds_m4_causal_donor_directional_preemptible_20260731.yaml"
TRAIN_MANIFEST = "manifests/cf4_balanced_paired_state_swap_train_seed20260731_v1.json"
VALID_MANIFEST = "manifests/cf_paired_state_swap_valid_seed20260731_v1.json"
TRAIN_MANIFEST_SHA256 = "4e7907cf4d74eced4f4c5c2dddf7213a2649cddca081fd2b4a51f35934b35773"
TRAIN_JSONL_SHA256 = "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766"
VALID_JSONL_SHA256 = "3e6b5fbcab3d232d5b3a65a005fbf9ba5719a8a4f8251d6ea85b861a7c1c3e22"

SPECS = {
    "bootstrap_cf4_w1": {
        "job": "m4-causal-cf4-bootstrap-symmetric-w1-round0to250-seed42",
        "objective": "bootstrap_then_symmetric",
    },
    "donor_directional_cf4_w1": {
        "job": "m4-causal-cf4-donor-directional-w1-round0to250-seed42",
        "objective": "donor_directional",
    },
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one template fragment, found {count}: {old!r}")
    return text.replace(old, new, 1)


def replace_all_checked(text: str, old: str, new: str, expected: int) -> str:
    count = text.count(old)
    if count != expected:
        raise ValueError(
            f"Expected {expected} template fragments, found {count}: {old!r}"
        )
    return text.replace(old, new)


def command_for(source: str, branch: str, objective: str) -> str:
    command = replace_once(
        source,
        "export CAUSAL_DATA=/tmp/recurrent_causal_pairs_v1",
        "export CAUSAL_DATA=/tmp/recurrent_causal_pairs_cf4_balanced_v1",
    )
    command = replace_once(
        command,
        "manifests/cf_paired_state_swap_train_seed20260730_v1.json",
        TRAIN_MANIFEST,
    )
    command = replace_once(
        command,
        "edb1bda6cc6aceb4507ac599377284c533014cef34b80e1063e062a74752267e",
        TRAIN_JSONL_SHA256,
    )
    command = replace_once(
        command,
        'test -f "$$CACHE_DIR/integrity.json"\n',
        'test -f "$$CACHE_DIR/integrity.json"\n'
        f'test "$$(sha256sum {TRAIN_MANIFEST} | cut -d\' \' -f1)" = '
        f'{TRAIN_MANIFEST_SHA256}\n',
    )
    command = replace_once(
        command,
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/'
        'world_model_v1_m4_causal_$${BRANCH}_seed42_round500_20260731"',
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/'
        'world_model_v1_m4_causal_cf4_$${BRANCH}_seed42_round250_20260731"',
    )
    command = replace_once(
        command,
        "assert 0 < int(receipt.get('round', -1)) <= 500, receipt",
        "assert 0 < int(receipt.get('round', -1)) <= 250, receipt",
    )
    command = replace_once(
        command,
        "--diagnostic-stop-after-round 500",
        "--diagnostic-stop-after-round 250",
    )
    for old, new in (
        ("assert completion.get('micro_steps') == 2000", "assert completion.get('micro_steps') == 1000"),
        ("assert completion.get('causal_pairs_seen') == 2000", "assert completion.get('causal_pairs_seen') == 1000"),
        ("assert completion.get('unique_causal_pairs_seen') == 1726", "assert completion.get('unique_causal_pairs_seen') == 1000"),
        ("assert config.get('diagnostic_stop_after_round') == 500", "assert config.get('diagnostic_stop_after_round') == 250"),
    ):
        command = replace_once(command, old, new)

    if objective != "donor_directional":
        command = replace_all_checked(
            command,
            "assert config.get('causal_objective_mode') == 'donor_directional', config",
            f"assert config.get('causal_objective_mode') == '{objective}', config",
            2,
        )
        command = replace_once(
            command,
            "--causal-objective-mode donor_directional",
            f"--causal-objective-mode {objective}",
        )

    # The frozen C4 evaluator must see only one GPU after the four replicated
    # training ranks exit; otherwise Transformers may shard the backbone and
    # place the recurrent state on a different device.
    command = replace_once(
        command,
        "python -u scripts/evaluate_recurrent_causal_gate.py \\\n",
        "export CUDA_VISIBLE_DEVICES=0\npython -u scripts/evaluate_recurrent_causal_gate.py \\\n",
    )
    if branch not in command and "${BRANCH}" not in command:
        raise ValueError("Rendered command lost its branch address")
    return command


def render(base: dict) -> dict:
    source_jobs = base.get("jobs") or []
    if not source_jobs:
        raise ValueError("Source config has no jobs")
    source = source_jobs[0]
    result = copy.deepcopy(base)
    result["description"] = (
        "CF4 balanced multi-query recurrent causal diagnostic: each accepted "
        "donor state supervises 2-4 distinct queries; compare bootstrap-to-"
        "symmetric against persistent donor-directional at matched weight"
    )
    result["jobs"] = []
    for branch, spec in SPECS.items():
        job = copy.deepcopy(source)
        job["name"] = spec["job"]
        job["command"] = command_for(
            str(job["command"]), branch, spec["objective"]
        )
        env = job["submit_args"]["env"]
        env["BRANCH"] = branch
        env["CAUSAL_WEIGHT"] = "1.0"
        result["jobs"].append(job)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(SOURCE))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(json.dumps({"jobs": [job["name"] for job in value["jobs"]], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
