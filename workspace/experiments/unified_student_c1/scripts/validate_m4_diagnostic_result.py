#!/usr/bin/env python3
"""Fail closed on a selected M4 diagnostic before the full epoch is launched."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "completion.json",
    "gradient_audit.json",
    "validation_metrics.json",
    "causal_validation_metrics.json",
    "run_config.json",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_numbers(value: Any, prefix: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f"{prefix}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite metric at {prefix}: {value}")


def validate(root: Path, projection: str, arm: str) -> dict[str, Any]:
    if projection not in {"identity", "trainable"}:
        raise ValueError(f"Unknown projection: {projection}")
    if arm not in {"generation_only", "generation_kd"}:
        raise ValueError(f"Unknown arm: {arm}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing diagnostic files: {missing}")

    completion = read_json(root / "completion.json")
    gradients = read_json(root / "gradient_audit.json")
    validation = read_json(root / "validation_metrics.json")
    causal = read_json(root / "causal_validation_metrics.json")
    config = read_json(root / "run_config.json")
    finite_numbers(validation, "validation")
    finite_numbers(causal, "causal_validation")

    checks = {
        "diagnostic_receipt": completion.get("status") == "diagnostic_complete"
        and completion.get("diagnostic") is True,
        "four_micro_steps": int(completion.get("micro_steps", -1)) == 4,
        "optimizer_step_executed": int(completion.get("optimizer_steps", 0)) >= 1,
        "real_examples_exercised": int(completion.get("real_examples_seen", 0)) == 4,
        "causal_path_exercised": completion.get("contains_causal_pair") is True
        and int(completion.get("causal_pairs_seen", 0)) >= 1,
        "frozen_weights_have_no_grad": not gradients.get("frozen_tensors_with_grad"),
        "all_lora_tensors_receive_grad": int(
            (gradients.get("lora") or {}).get("parameter_tensors", 0)
        )
        == int((gradients.get("lora") or {}).get("tensors_with_nonzero_gradient", -1))
        and float((gradients.get("lora") or {}).get("gradient_abs_sum", 0.0)) > 0,
        "validation_support": int(validation.get("candidate_n", validation.get("n", 0)))
        >= 20,
        "causal_validation_support": int(causal.get("n", 0)) >= 20,
        "causal_clean_support": int(causal.get("clean_n", 0)) >= 1,
        "projection_mode_matches": bool(config.get("trainable_projections"))
        == (projection == "trainable"),
        "kd_mode_matches": (float(config.get("behavior_kd_weight", 0.0)) > 0)
        == (arm == "generation_kd"),
    }
    projection_grad = gradients.get("state_projection") or {}
    if projection == "trainable":
        checks["projection_receives_grad"] = (
            int(projection_grad.get("parameter_tensors", 0)) > 0
            and int(projection_grad.get("tensors_with_nonzero_gradient", 0)) > 0
            and float(projection_grad.get("gradient_abs_sum", 0.0)) > 0
        )
    else:
        checks["identity_has_no_trainable_projection"] = int(
            projection_grad.get("parameter_tensors", 0)
        ) == 0

    expected_hashes = completion.get("sha256") or {}
    for name in (
        "validation_metrics",
        "causal_validation_metrics",
        "run_config",
        "gradient_audit",
    ):
        filename = f"{name}.json"
        checks[f"hash_{name}"] = expected_hashes.get(name) == sha256(root / filename)

    payload = {
        "stage": "M4_selected_diagnostic",
        "arm": arm,
        "projection_mode": projection,
        "pass": all(checks.values()),
        "checks": checks,
        "completion": completion,
        "validation_metrics": validation,
        "causal_validation_metrics": causal,
    }
    if not payload["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"M4 diagnostic failed checks: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.root, args.projection, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
