#!/usr/bin/env python3
"""Aggregate independent final-fit, formal/C4, and T3 gates fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def validate(root: Path) -> dict[str, Any]:
    required_markers = {
        "final_fit_complete": root / "final_fit_complete",
        "final_eval_complete": root / "final_eval_complete",
        "t3_semantic_judge_complete": root / "t3_semantic_judge_complete",
    }
    missing = [name for name, path in required_markers.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Final pipeline markers missing: {missing}")

    gate_paths = {
        "final_fit": root / "final_fit_gate.json",
        "formal_and_c4": root / "final_scientific_gate_without_t3_semantic.json",
        "t3_semantic": root / "t3_semantic_judge/t3_semantic_gate.json",
    }
    gates = {name: read_object(path) for name, path in gate_paths.items()}
    if gates["formal_and_c4"].get("stage") != "M4_strict_recurrent_pilot":
        raise ValueError("Formal/C4 gate stage drift")
    if gates["t3_semantic"].get("stage") != "final_T3_semantic_judge":
        raise ValueError("T3 semantic gate stage drift")
    failed = [name for name, gate in gates.items() if gate.get("pass") is not True]
    if failed:
        raise ValueError(f"Final scientific gates failed: {failed}")

    final_models = list((root / "final_fit_download").rglob("adapter_model.safetensors"))
    recurrent_states = list((root / "final_fit_download").rglob("recurrent_state.pt"))
    if len(final_models) != 1 or len(recurrent_states) != 1:
        raise ValueError(
            f"Expected one downloaded final adapter/state, got {final_models}, {recurrent_states}"
        )
    return {
        "version": 1,
        "status": "complete",
        "objective": "full_unified_recurrent_student_model",
        "checks": {
            "final_fit": True,
            "formal_T1_T2_T3_T5_T6": True,
            "causal_C4": True,
            "T3_semantic_fixed_panel": True,
            "downloaded_adapter_and_recurrent_state": True,
        },
        "artifacts": {
            "adapter_model": str(final_models[0]),
            "recurrent_state": str(recurrent_states[0]),
        },
        "gates": {name: str(path) for name, path in gate_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
