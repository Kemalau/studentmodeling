#!/usr/bin/env python3
"""Select the next recurrent-state action from frozen causal diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def passing_rows(payload: dict) -> list[dict]:
    return [
        row
        for row in payload.get("rows") or []
        if row.get("scientific_gate_pass") is True
        and float(row.get("D_specific", -1.0)) >= 0.15
        and int(row.get("clean_n", 0)) >= 50
        and float(row.get("noop_flip_to_A", 1.0)) <= 0.05
        and float(row.get("same_rule_flip_to_A", 1.0)) <= 0.10
    ]


def select(
    scalar_transition: dict,
    layerwise_bridge: dict,
    cross_adapter_bridge: dict,
    layerwise_recurrent: dict,
) -> dict:
    scalar_rows = scalar_transition.get("rows") or []
    if len(scalar_rows) < 2:
        raise ValueError("Scalar transition audit is incomplete")
    scalar_pass = any(row.get("pass") is True for row in scalar_rows)
    bridge_pass = passing_rows(layerwise_bridge)
    cross_pass = passing_rows(cross_adapter_bridge)
    recurrent_pass = passing_rows(layerwise_recurrent)
    full_task_bridge = [row for row in cross_pass if row.get("decoder") == "full_task"]
    full_task_recurrent = [
        row for row in recurrent_pass if row.get("decoder") == "full_task"
    ]
    if full_task_recurrent:
        decision = "formal_eval_layerwise_recurrent_full_task"
        rationale = (
            "The zero-training closed recurrence and complete-task decoder pass "
            "the frozen C4 gate; evaluate T1/T2/T3/T5/T6 before any weight update."
        )
        selected = full_task_recurrent[0]
    elif bridge_pass:
        decision = "train_layerwise_transition_only"
        rationale = (
            "A history-free decoder can consume the layerwise state, but the "
            "zero-training recurrent update is not closed; train only the transition."
        )
        selected = (full_task_bridge or cross_pass or bridge_pass)[0]
    else:
        decision = "continue_causal_objective_diagnosis"
        rationale = (
            "Even full-history layerwise transport misses frozen C4; representation "
            "space alone is insufficient, so full recurrent training is unauthorized."
        )
        selected = None
    payload = {
        "stage": "recurrent_state_architecture_selection",
        "decision": decision,
        "rationale": rationale,
        "scalar_transition_pass": scalar_pass,
        "layerwise_bridge_pass_count": len(bridge_pass),
        "cross_adapter_bridge_pass_count": len(cross_pass),
        "layerwise_recurrent_pass_count": len(recurrent_pass),
        "selected_row": selected,
        "full_unified_fit_authorized": decision
        == "formal_eval_layerwise_recurrent_full_task",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-transition", type=Path, required=True)
    parser.add_argument("--layerwise-bridge", type=Path, required=True)
    parser.add_argument("--cross-adapter-bridge", type=Path, required=True)
    parser.add_argument("--layerwise-recurrent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = select(
        load(args.scalar_transition),
        load(args.layerwise_bridge),
        load(args.cross_adapter_bridge),
        load(args.layerwise_recurrent),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
