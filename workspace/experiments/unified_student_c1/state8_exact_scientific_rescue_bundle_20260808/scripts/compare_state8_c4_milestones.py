#!/usr/bin/env python3
"""Compare pair-level C4 transitions between two hash-bound milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = ("native", "noop", "samemal", "random", "swap")


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("state_shape") != [64, 8, 5120]:
        raise ValueError(f"State-shape drift: {path}")
    if value.get("raw_history_visible_to_decoder") is not False:
        raise ValueError(f"History-route drift: {path}")
    records = value.get("records") or []
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[int(record["pair_idx"])][str(record["condition"])] = record
    if any(set(row) != set(CONDITIONS) for row in grouped.values()):
        raise ValueError(f"Incomplete condition set: {path}")
    manifest = Path(str(value.get("pair_manifest") or ""))
    if not manifest.is_file():
        raise ValueError(f"Pair manifest missing: {manifest}")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return {
        **value,
        "_grouped": dict(grouped),
        "_pair_manifest_sha256": manifest_sha256,
    }


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def compare(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = read(before_path)
    after = read(after_path)
    if before["_pair_manifest_sha256"] != after["_pair_manifest_sha256"]:
        raise ValueError("Milestones use different frozen pair manifests")
    before_clean = {
        index: row
        for index, row in before["_grouped"].items()
        if row["native"]["category"] == "stay_B"
    }
    after_clean = {
        index: row
        for index, row in after["_grouped"].items()
        if row["native"]["category"] == "stay_B"
    }
    common = sorted(set(before_clean) & set(after_clean))
    same_rule_regressions = [
        index
        for index in common
        if before_clean[index]["samemal"]["category"] == "stay_B"
        and after_clean[index]["samemal"]["category"] != "stay_B"
    ]
    regression_rows = []
    for index in same_rule_regressions:
        old = before_clean[index]
        new = after_clean[index]
        native = old["native"]
        regression_rows.append(
            {
                "pair_idx": index,
                "family": native["family"],
                "recipient_label": native["recipient_label"],
                "donor_label": native["donor_label"],
                "before_category": old["samemal"]["category"],
                "after_category": new["samemal"]["category"],
                "before_margin_A_minus_B": old["samemal"]["margin_A_minus_B"],
                "after_margin_A_minus_B": new["samemal"]["margin_A_minus_B"],
            }
        )
    transitions = {
        condition: counter_dict(
            Counter(
                (
                    before_clean[index][condition]["category"],
                    after_clean[index][condition]["category"],
                )
                for index in common
            )
        )
        for condition in CONDITIONS
    }
    return {
        "stage": "state8_c4_pairwise_milestone_comparison_v1",
        "before": str(before_path),
        "after": str(after_path),
        "pair_manifest_sha256": before["_pair_manifest_sha256"],
        "before_artifact_sha256": before["artifact_sha256"],
        "after_artifact_sha256": after["artifact_sha256"],
        "before_clean_n": len(before_clean),
        "after_clean_n": len(after_clean),
        "common_clean_n": len(common),
        "condition_transitions": transitions,
        "same_rule_regression_n": len(regression_rows),
        "same_rule_regression_family_counts": counter_dict(
            Counter(row["family"] for row in regression_rows)
        ),
        "same_rule_regression_recipient_label_counts": counter_dict(
            Counter(row["recipient_label"] for row in regression_rows)
        ),
        "same_rule_regressions": regression_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
