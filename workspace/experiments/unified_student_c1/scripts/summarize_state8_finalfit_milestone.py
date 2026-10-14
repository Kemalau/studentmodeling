#!/usr/bin/env python3
"""Join the task, semantic, and causal gates for one final-fit milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROUNDS = {3249, 6497, 9745, 12993}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def summarize(
    receipt_path: Path,
    formal_path: Path,
    semantic_path: Path,
    c4_path: Path,
) -> dict[str, Any]:
    receipt = read(receipt_path)
    formal = read(formal_path)
    semantic = read(semantic_path)
    c4 = read(c4_path)
    round_index = int(receipt.get("round", -1))
    variant = str(receipt.get("variant") or "")
    if receipt.get("stage") != "state8_finalfit_milestone_eval_v1":
        raise ValueError("Milestone receipt stage drift")
    if receipt.get("status") != "complete" or round_index not in ROUNDS:
        raise ValueError("Milestone receipt is incomplete")
    if variant not in {"t3_heavy", "highkd", "balanced"}:
        raise ValueError("Milestone variant drift")
    expected = receipt.get("evaluated_artifact_sha256")
    if set(expected or {}) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Milestone evaluated artifact hashes are incomplete")
    if formal.get("checkpoint_round") != round_index:
        raise ValueError("Formal checkpoint coordinate drift")
    if semantic.get("checkpoint_round") != round_index:
        raise ValueError("Semantic checkpoint coordinate drift")
    if formal.get("stage") != f"layerwise_finalfit_checkpoint{round_index}_formal_screen":
        raise ValueError("Formal stage drift")
    if semantic.get("stage") != f"layerwise_finalfit_checkpoint{round_index}_T3_semantic_judge":
        raise ValueError("Semantic stage drift")
    if c4.get("state_shape") != [64, 8, 5120]:
        raise ValueError("C4 state shape drift")
    if c4.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("C4 decoder history route is open")
    if c4.get("persistent_history_kv_cache") is not False:
        raise ValueError("C4 decoder persists history KV cache")
    if c4.get("artifact_sha256") != expected:
        raise ValueError("C4 evaluated artifact hash drift")
    checks = {
        "T1_T2_T5_T6_small_screen": bool(formal.get("diagnostic_gate_pass")),
        "T3_semantic": bool(semantic.get("pass")),
        "C4_seed42": bool((c4.get("summary") or {}).get("scientific_gate_pass")),
    }
    return {
        "stage": "state8_finalfit_milestone_joint_gate_v1",
        "diagnostic_only": True,
        "formal_final_evaluation_required": True,
        "variant": variant,
        "round": round_index,
        "pass": all(checks.values()),
        "checks": checks,
        "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
        "evaluated_artifact_sha256": expected,
        "formal_screen": formal,
        "T3_semantic": semantic,
        "C4": {
            "pair_seed": int(c4.get("pair_seed", c4.get("seed", -1))),
            "clean_n": int((c4.get("summary") or {}).get("clean_n", 0)),
            "D_specific": float((c4.get("summary") or {})["D_specific"]),
            "scientific_gate_pass": checks["C4_seed42"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.receipt, args.formal, args.semantic, args.c4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["pass"] else 2)


if __name__ == "__main__":
    main()
