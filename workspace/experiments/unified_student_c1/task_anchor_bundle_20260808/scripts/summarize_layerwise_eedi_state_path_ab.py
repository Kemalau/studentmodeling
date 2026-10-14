#!/usr/bin/env python3
"""Summarize the matched Eedi-dense standard versus state-path diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_layerwise_checkpoint250_formal_screen import summarize as summarize_formal
from validate_checkpoint_t3_semantic_result import validate as validate_t3


PROFILES = {"standard", "state_path"}
T6_REFERENCE_MACRO_F1 = 0.5959866676584252
T6_MIN_RATIO = 0.85


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def profile_root(root: Path, profile: str) -> Path:
    receipts = [
        path
        for path in root.rglob("task_retention_training_receipt.json")
        if read(path).get("loss_profile") == profile
    ]
    if len(receipts) != 1:
        raise ValueError(f"Expected one receipt for profile={profile}, got {receipts}")
    for ancestor in receipts[0].parents:
        if (ancestor / "eval/formal_screen/metrics.json").is_file():
            return ancestor
        if ancestor == root:
            break
    raise ValueError(f"Could not locate evaluation root for profile={profile}")


def weighted(groups: dict[str, Any], metric: str) -> float:
    support = sum(int(row["n"]) for row in groups.values())
    if support <= 0:
        raise ValueError("Metric support must be positive")
    return sum(float(row[metric]) * int(row["n"]) for row in groups.values()) / support


def summarize(root: Path, t3_root: Path) -> dict[str, Any]:
    rows = []
    for profile in sorted(PROFILES):
        job = profile_root(root, profile)
        receipt = read(next(job.rglob("task_retention_training_receipt.json")))
        completion = read(next(job.rglob("metrics/completion.json")))
        run_config = read(next(job.rglob("metrics/run_config.json")))
        c4 = read(job / "eval/c4/recurrent_causal_gate.json")
        formal = summarize_formal(job / "eval/formal_screen", checkpoint_round=750)
        t6 = read(job / "eval/t6_full/metrics.json")
        t3 = validate_t3(t3_root / profile / "metrics.json", checkpoint_round=750)
        clean = c4["summary"]["clean"]
        t6_groups = t6["metrics_by_group"]
        t6_macro = weighted(t6_groups, "macro_f1")
        t6_recall = weighted(t6_groups, "positive_recall")
        checks = {
            "diagnostic_only": receipt.get("diagnostic_only") is True and receipt.get("exact_resume") is False,
            "matched_branch": receipt.get("branch") == "eedi_dense" and receipt.get("loss_profile") == profile,
            "train_support": completion.get("micro_steps") == 1000 and completion.get("optimizer_steps") == 250,
            "train_boundary": completion.get("history_global_position_start") == 12000 and completion.get("history_global_position_end") == 12999,
            "causal_exposure": completion.get("causal_pairs_seen") == 50,
            "causal_recipe": float(run_config.get("causal_weight", -1.0)) == 2.0
            and float(receipt.get("causal_weight", run_config.get("causal_weight", -1.0))) == 2.0,
            "formal_test_overlap_zero": receipt.get("formal_example_id_overlap") == 0,
            "closed_layerwise": c4.get("state_architecture") == "layerwise" and c4.get("raw_history_visible_to_decoder") is False and c4.get("persistent_history_kv_cache") is False,
            "C4": bool(c4["summary"].get("scientific_gate_pass")),
            "formal_screen": bool(formal.get("diagnostic_gate_pass")),
            "T3_semantic": bool(t3.get("pass")),
            "T6_retention": t6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO and t6_recall > 0.0,
        }
        rows.append({
            "loss_profile": profile,
            "causal_weight": float(run_config["causal_weight"]),
            "checks": checks,
            "all_gates_pass": all(checks.values()),
            "task_counts": receipt["task_counts"],
            "C4": {
                "clean_n": c4["summary"]["clean_n"],
                "swap": clean["swap"]["flip_to_A_rate"],
                "random": clean["random"]["flip_to_A_rate"],
                "same_rule": clean["samemal"]["flip_to_A_rate"],
                "noop": clean["noop"]["flip_to_A_rate"],
                "D_specific": c4["summary"]["D_specific"],
            },
            "formal_screen": formal,
            "T3_semantic": t3,
            "T6": {
                "support": int(t6["sampled_rows"]),
                "macro_f1": t6_macro,
                "reference_macro_f1": T6_REFERENCE_MACRO_F1,
                "retention_ratio": t6_macro / T6_REFERENCE_MACRO_F1,
                "positive_recall": t6_recall,
                "by_group": t6_groups,
            },
        })
    eligible = [row for row in rows if row["all_gates_pass"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["T3_semantic"]["retention_ratio"],
            row["T6"]["retention_ratio"],
            row["C4"]["D_specific"],
        ),
    ) if eligible else None
    return {
        "stage": "layerwise_eedi_state_path_ab_comparison",
        "diagnostic_only": True,
        "source_checkpoint_round": 750,
        "matched_data": True,
        "profiles": rows,
        "any_all_gates_pass": bool(eligible),
        "selected_profile": selected["loss_profile"] if selected else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--t3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.root, args.t3_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
