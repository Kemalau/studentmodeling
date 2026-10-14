#!/usr/bin/env python3
"""Audit one strict CF1 + vision-pad seed from downloaded AMLT artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_INPUT_SHA = {
    "train.jsonl": "929487329576607572aa49b8e2daba53605b3c733749e7e631baa376a9df65f9",
    "valid.jsonl": "4bf7117db760bcaba9790b8f7901ae89428911359f2b483f08aef39dc1fe6a36",
    "test_same_template.jsonl": "079f7634cc54dfd55dfc234503bb5925ab734b1ca74aa2f2a9a17cc4375a2c2f",
    "test_cross_template.jsonl": "405446c371185d23f06b4a0530c8ef4da4d1ac9addb696180f83bdf092402779",
    "c4_stage_a_gate_pairs_seed42_n150_v1.json": "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8",
    "train_c1_lora.py": "4e7e633263539b0a46aaa5172a71e921d418c448f89d67ef1fc671044244b6a7",
    "evaluate_c1.py": "df2451a6e7658cdf4e6ff0e5abe1e3b0708e63685265d46ade0ee97b09846be6",
    "probe_student_state_smoke.py": "dc1cda0b72d4bbe8fdbbf05952ac5237c9e62f1afc0e33dac77b1bdbded8c147",
    "state_swap_misconception_smoke.py": "1b8d4db48f8b228218c735e8fbb900ded7600e81ee9a55ccc40a6d358230aadd",
}
CONDITIONS = ("none", "noop", "samemal", "random", "swap")
PAIR_FIELDS = ("donor_id", "recipient_id", "samemal_id")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            result[Path(parts[1]).name] = parts[0]
    return result


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_dir
    gate = root / "gate"
    audit = root / "audit"
    required = [
        gate / "gate_summary.json",
        gate / "response_same_masked.json",
        gate / "response_cross_masked.json",
        gate / "state_probe_masked.json",
        gate / "state_swap_masked.json",
        audit / "input_manifest.txt",
        audit / "output_manifest.txt",
        audit / "train.log",
        *(audit / f"trainer_state_step{step}.json" for step in (250, 500, 750, 1000, 1250, 1500)),
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]

    checks: dict[str, Any] = {"missing_files": missing}
    if missing:
        result = {"passed": False, "seed": args.seed, "checks": checks}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise SystemExit(json.dumps(result, indent=2))

    input_manifest = parse_manifest(audit / "input_manifest.txt")
    input_text = (audit / "input_manifest.txt").read_text(encoding="utf-8")
    checks["seed_identity"] = f"train_seed={args.seed}" in input_text
    checks["protocol_identity"] = all(
        token in input_text
        for token in (
            "condition=cf1",
            "token_name=visionpad",
            "state_token_alias=<|vision_pad|>",
            "model=Qwen3-32B",
            "max_steps=1500",
            "learning_rate=1e-4",
            "save_steps=250",
        )
    )
    checks["input_hashes"] = {
        name: input_manifest.get(name) == expected
        for name, expected in EXPECTED_INPUT_SHA.items()
    }

    checkpoint_steps: dict[str, Any] = {}
    for step in (250, 500, 750, 1000, 1250, 1500):
        state = read_json(audit / f"trainer_state_step{step}.json")
        checkpoint_steps[str(step)] = {
            "global_step": state.get("global_step"),
            "max_steps": state.get("max_steps"),
            "valid": state.get("global_step") == step and state.get("max_steps") == 1500,
        }
    checks["checkpoint_steps"] = checkpoint_steps

    final_state = read_json(audit / "trainer_state_step1500.json")
    eval_rows = [row for row in final_state.get("log_history", []) if "eval_loss" in row]
    checks["training_complete"] = final_state.get("global_step") == 1500
    checks["final_eval_loss"] = float(eval_rows[-1]["eval_loss"]) if eval_rows else None
    checks["final_eval_finite"] = bool(eval_rows and finite(eval_rows[-1]["eval_loss"]))

    frozen = read_json(args.pair_manifest)
    swap = read_json(gate / "state_swap_masked.json")
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in swap.get("records", []):
        grouped.setdefault(int(record["pair_idx"]), {})[str(record["condition"])] = record
    expected_ids = list(range(1, 151))
    complete_conditions = all(set(grouped.get(index, {})) == set(CONDITIONS) for index in expected_ids)
    pair_match = True
    for index, expected in enumerate(frozen["pairs"], start=1):
        none = grouped.get(index, {}).get("none", {})
        if any(none.get(field) != expected[field] for field in PAIR_FIELDS):
            pair_match = False
            break
    checks["swap_shape"] = {
        "num_pairs": swap.get("num_pairs"),
        "record_count": len(swap.get("records", [])),
        "complete_conditions": complete_conditions,
    }
    checks["frozen_pair_manifest_exact"] = pair_match
    checks["all_64_layers"] = swap.get("layers") == list(range(64))
    checks["state_alias"] = swap.get("state_token_alias") == "<|vision_pad|>"
    checks["bottleneck"] = bool(swap.get("bottleneck_mask")) and swap.get("bottleneck_style") == "late_fusion"

    summary = read_json(gate / "gate_summary.json")
    recompute_path = args.output.with_name(args.output.stem + "_recomputed_gate.json")
    from summarize_stage_a_gate import compact_task_metrics, probe_summary, clean_swap_summary

    recomputed = {
        "response_same_masked": compact_task_metrics(gate / "response_same_masked.json"),
        "response_cross_masked": compact_task_metrics(gate / "response_cross_masked.json"),
        "state_probe_masked": probe_summary(gate / "state_probe_masked.json"),
        "state_swap_masked": clean_swap_summary(gate / "state_swap_masked.json"),
    }
    recompute_path.write_text(json.dumps(recomputed, indent=2, sort_keys=True) + "\n")
    checks["gate_summary_reproduced"] = recomputed == summary

    clean = summary["state_swap_masked"]["clean"]
    directed = float(clean["swap"]["flip_to_A_rate"])
    random_rate = float(clean["random"]["flip_to_A_rate"])
    same_rate = float(clean["samemal"]["flip_to_A_rate"])
    noop_rate = float(clean["noop"]["flip_to_A_rate"])
    state_probe = summary["state_probe_masked"]["positions"]["state"]["best_cross"]["accuracy"]
    metrics = {
        "response_same_accuracy": summary["response_same_masked"]["accuracy"],
        "response_cross_accuracy": summary["response_cross_masked"]["accuracy"],
        "state_probe_best_cross_accuracy": state_probe,
        "clean_n": summary["state_swap_masked"]["clean_n"],
        "directed_flip_rate": directed,
        "random_flip_rate": random_rate,
        "same_rule_flip_rate": same_rate,
        "noop_flip_rate": noop_rate,
        "D_specific": directed - random_rate,
    }

    output_manifest = parse_manifest(audit / "output_manifest.txt")
    checks["downloaded_output_hashes"] = {
        "gate_summary.json": output_manifest.get("gate_summary.json") == sha256(gate / "gate_summary.json"),
        "trainer_state_step1500.json": output_manifest.get("trainer_state_step1500.json")
        == sha256(audit / "trainer_state_step1500.json"),
    }

    def all_true(value: Any) -> bool:
        if isinstance(value, dict):
            return all(all_true(item) for item in value.values())
        if isinstance(value, bool):
            return value
        return True

    passed = all_true(checks)
    result = {
        "passed": passed,
        "seed": args.seed,
        "artifact_dir": str(root),
        "artifact_sha256": {str(path.relative_to(root)): sha256(path) for path in required},
        "checks": checks,
        "metrics": metrics,
        "recomputed_gate": str(recompute_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
