#!/usr/bin/env python3
"""Select the preregistered qualified State8 promotion arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = {
    "t3_heavy": {
        "stage": "state8_qualified_promotion_gate_v1",
        "immutable_root": "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_v1_immutable",
        "immutable_stage": "state8_qualified_teacher_curriculum_immutable_v1",
    },
    "balanced": {
        "stage": "state8_qualified_balanced_promotion_gate_v1",
        "immutable_root": "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable",
        "immutable_stage": "state8_qualified_teacher_balanced_curriculum_immutable_v1",
    },
    "highkd": {
        "stage": "state8_qualified_highkd_promotion_gate_v1",
        "immutable_root": "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable",
        "immutable_stage": "state8_qualified_teacher_curriculum_highkd_immutable_v1",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def audit(name: str, gate_path: Path) -> dict[str, Any]:
    value = read(gate_path); expected = ARMS[name]
    if value.get("stage") != expected["stage"] or value.get("promotion_variant") != name:
        raise ValueError(f"Promotion identity drift: {name}")
    if set(value.get("checks") or {}) != {"C4_multiseed", "T3_semantic", "T1_T2_T5_T6_small_screen"}:
        raise ValueError(f"Promotion check-set drift: {name}")
    if bool(value.get("pass")) != all(bool(item) for item in value["checks"].values()):
        raise ValueError(f"Promotion pass inconsistency: {name}")
    artifacts = value.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
        len(str(digest)) != 64 for digest in artifacts.values()
    ):
        raise ValueError(f"Promotion artifact hashes missing: {name}")
    artifact = ((value.get("C4") or {}).get("rows") or [])
    # The merged promotion artifact hashes are carried by the T3 merge; the
    # controller stores them beside the gate in merged/t3_metrics.json.
    return {
        "variant": name,
        "pass": bool(value["pass"]),
        "checks": value["checks"],
        "promotion_gate": str(gate_path),
        "promotion_gate_sha256": sha256(gate_path),
        "immutable_root": expected["immutable_root"],
        "immutable_stage": expected["immutable_stage"],
        "artifact_sha256": artifacts,
        "c4_pair_seeds": sorted(int(row["pair_seed"]) for row in artifact),
    }


def pending(name: str, reason: str) -> dict[str, Any]:
    return {
        "variant": name,
        "status": "not_required_by_preregistered_priority",
        "reason": reason,
    }


def select(
    primary_path: Path,
    highkd_path: Path | None = None,
    balanced_path: Path | None = None,
) -> dict[str, Any]:
    primary = audit("t3_heavy", primary_path)
    if primary["pass"]:
        chosen = primary
        rule = "primary_t3_heavy_passed"
        highkd = pending("highkd", "primary_joint_gate_passed")
        balanced = pending("balanced", "primary_joint_gate_passed")
    else:
        if highkd_path is None:
            raise ValueError("Primary failed; qualified high-KD gate is required")
        highkd = audit("highkd", highkd_path)
        if highkd["pass"]:
            chosen = highkd
            rule = "primary_failed_highkd_hedge_passed"
            balanced = pending("balanced", "highkd_joint_gate_passed")
        else:
            if balanced_path is None:
                raise ValueError(
                    "Primary and high-KD failed; qualified balanced gate is required"
                )
            balanced = audit("balanced", balanced_path)
            if balanced["pass"]:
                chosen = balanced
                rule = "primary_failed_balanced_fallback_passed"
            else:
                chosen = None
                rule = "no_arm_passed"
    return {
        "stage": "state8_qualified_promotion_selection_v1",
        "pass": chosen is not None,
        "selection_rule": "prefer_t3_heavy_then_highkd_then_balanced_if_joint_gate_passes",
        "decision": rule,
        "selected": chosen,
        "arms": {"t3_heavy": primary, "highkd": highkd, "balanced": balanced},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--highkd", type=Path)
    parser.add_argument("--balanced", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = select(args.primary, args.highkd, args.balanced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["pass"] else 2)


if __name__ == "__main__":
    main()
