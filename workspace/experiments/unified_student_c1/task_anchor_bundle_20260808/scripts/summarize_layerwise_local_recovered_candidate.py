#!/usr/bin/env python3
"""Summarize a locally completed, checkpoint-provenance-audited candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_layerwise_checkpoint250_formal_screen import summarize as summarize_formal
from validate_checkpoint_t3_semantic_result import validate as validate_t3


T6_REFERENCE_MACRO_F1 = 0.5959866676584252
T6_MIN_RATIO = 0.85


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def weighted(groups: dict[str, Any], metric: str) -> float:
    support = sum(int(row["n"]) for row in groups.values())
    if support <= 0:
        raise ValueError("Metric support must be positive")
    return sum(float(row[metric]) * int(row["n"]) for row in groups.values()) / support


def summarize(
    provenance_path: Path,
    c4_path: Path,
    formal_root: Path,
    t6_path: Path,
    t3_path: Path,
) -> dict[str, Any]:
    provenance = read(provenance_path)
    c4 = read(c4_path)
    formal = summarize_formal(formal_root, checkpoint_round=750)
    t6 = read(t6_path)
    t3 = validate_t3(t3_path, checkpoint_round=750)
    seen_unseen_path = t3_path.parent / "seen_unseen.json"
    seen_unseen = read(seen_unseen_path) if seen_unseen_path.is_file() else None
    c4_summary = c4["summary"]
    clean = c4_summary["clean"]
    t6_groups = t6["metrics_by_group"]
    t6_macro = weighted(t6_groups, "macro_f1")
    t6_recall = weighted(t6_groups, "positive_recall")
    provenance_pass = bool(
        provenance.get("status") == "passed"
        and provenance.get("branch") == "equal_t3_t6_open_t3"
        and provenance.get("loss_profile") == "state_path"
        and provenance.get("learning_rate") == 5e-7
        and provenance.get("causal_weight") == 2.0
        and provenance.get("history_rows") == 1000
        and provenance.get("history_global_position_start") == 12000
        and provenance.get("history_global_position_end") == 12999
        and provenance.get("history_unique_examples") == 1000
        and provenance.get("causal_pairs_seen") == 50
        and provenance.get("formal_example_id_overlap") == 0
        and provenance.get("training_view") == "open_t3_no_candidate_enumeration"
        and provenance.get("optimizer_present") is True
        and provenance.get("scheduler_present") is True
        and provenance.get("rank_rng_states") == 4
        and provenance.get("full_real_order_rows") == 259845
        and provenance.get("causal_schedule_size") == 12992
    )
    checks = {
        "diagnostic_only": provenance_pass,
        "matched_candidate": provenance_pass,
        "train_support": provenance_pass,
        "train_boundary": provenance_pass,
        "causal_exposure": provenance_pass,
        "causal_recipe": provenance_pass,
        "formal_test_overlap_zero": provenance_pass,
        "closed_layerwise": c4.get("state_architecture") == "layerwise"
        and c4.get("raw_history_visible_to_decoder") is False
        and c4.get("persistent_history_kv_cache") is False,
        "C4": bool(c4_summary.get("scientific_gate_pass")),
        "formal_screen": bool(formal.get("diagnostic_gate_pass")),
        "T3_semantic": bool(t3.get("pass")),
        "T6_retention": t6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO
        and t6_recall > 0.0,
        "open_t3_training_view": provenance_pass,
    }
    return {
        "stage": "layerwise_single_task_retention_candidate",
        "candidate_id": "equal_t3_t6_open_t3__state_path",
        "branch": "equal_t3_t6_open_t3",
        "loss_profile": "state_path",
        "learning_rate": 5e-7,
        "causal_weight": 2.0,
        "diagnostic_only": True,
        "recovered_from_hash_verified_checkpoint": True,
        "provenance_audit": str(provenance_path),
        "checks": checks,
        "all_gates_pass": all(checks.values()),
        "task_counts": provenance["task_counts"],
        "C4": {
            "clean_n": c4_summary["clean_n"],
            "swap": clean["swap"]["flip_to_A_rate"],
            "random": clean["random"]["flip_to_A_rate"],
            "same_rule": clean["samemal"]["flip_to_A_rate"],
            "noop": clean["noop"]["flip_to_A_rate"],
            "D_specific": c4_summary["D_specific"],
        },
        "formal_screen": formal,
        "T3_semantic": t3,
        "T3_seen_unseen": seen_unseen,
        "T6": {
            "support": int(t6["sampled_rows"]),
            "macro_f1": t6_macro,
            "reference_macro_f1": T6_REFERENCE_MACRO_F1,
            "retention_ratio": t6_macro / T6_REFERENCE_MACRO_F1,
            "positive_recall": t6_recall,
            "by_group": t6_groups,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--t6", type=Path, required=True)
    parser.add_argument("--t3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.provenance, args.c4, args.formal_root, args.t6, args.t3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
