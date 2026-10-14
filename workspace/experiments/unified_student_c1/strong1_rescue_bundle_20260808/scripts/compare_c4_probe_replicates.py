#!/usr/bin/env python3
"""Compare two identical-protocol C4 probes across execution environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("summary"), dict):
        raise ValueError(f"Invalid C4 result: {path}")
    return value


def compact(value: dict) -> dict:
    summary = value["summary"]
    clean = summary["clean"]
    return {
        "D_specific": float(summary["D_specific"]),
        "D_damage": float(summary["D_damage"]),
        "clean_n": int(summary["clean_n"]),
        "swap_flip": float(clean["swap"]["flip_to_A_rate"]),
        "random_flip": float(clean["random"]["flip_to_A_rate"]),
        "noop_flip": float(clean["noop"]["flip_to_A_rate"]),
        "same_rule_flip": float(clean["samemal"]["flip_to_A_rate"]),
        "scientific_gate_pass": bool(summary["scientific_gate_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_raw, remote_raw = load(args.local), load(args.remote)
    invariants = (
        "intervention_target",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "random_control",
    )
    for key in invariants:
        if local_raw.get(key) != remote_raw.get(key):
            raise ValueError(f"Protocol drift in {key}")
    local, remote = compact(local_raw), compact(remote_raw)
    payload = {
        "protocol_match": True,
        "authoritative_environment": "remote_locked_container",
        "local": local,
        "remote": remote,
        "absolute_delta_D_specific": abs(local["D_specific"] - remote["D_specific"]),
        "absolute_delta_D_damage": abs(local["D_damage"] - remote["D_damage"]),
        "gate_agreement": local["scientific_gate_pass"] == remote["scientific_gate_pass"],
    }
    if not payload["gate_agreement"]:
        raise ValueError("Local and remote scientific gates disagree")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
