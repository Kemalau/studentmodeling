#!/usr/bin/env python3
"""Summarize the seven-job state-alias factorial after AMLT result download."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


JOB_RE = re.compile(r"state-(?P<condition>cf1|original)-(?P<token>\w+)-seed(?P<seed>\d+)")


def clean_arm(swap: dict, name: str) -> dict:
    arm = swap.get("clean", {}).get(name, {})
    return {
        "changed_top1_rate": arm.get("changed_top1_rate"),
        "flip_to_A_rate": arm.get("flip_to_A_rate"),
        "mean_delta_margin_A_minus_B": arm.get("mean_delta_margin_A_minus_B"),
        "n": arm.get("n"),
    }


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("gate_summary.json")):
        match = next((JOB_RE.search(part) for part in path.parts if JOB_RE.search(part)), None)
        if match is None:
            continue
        data = json.loads(path.read_text())
        probe = data.get("state_probe_masked", {}).get("positions", {}).get("state", {})
        swap = data.get("state_swap_masked", {})
        arms = {name: clean_arm(swap, name) for name in ("swap", "random", "samemal", "noop")}
        swap_flip = arms["swap"]["flip_to_A_rate"]
        random_flip = arms["random"]["flip_to_A_rate"]
        noop_flip = arms["noop"]["flip_to_A_rate"]
        rows.append(
            {
                **match.groupdict(),
                "seed": int(match.group("seed")),
                "response_same_accuracy": data.get("response_same_masked", {}).get("accuracy"),
                "response_cross_accuracy": data.get("response_cross_masked", {}).get("accuracy"),
                "state_probe_best_cross_accuracy": probe.get("best_cross", {}).get("accuracy"),
                "state_probe_best_cross_layer": probe.get("best_cross_layer"),
                "clean_n": swap.get("clean_n"),
                "arms": arms,
                "swap_minus_random_flip": (
                    swap_flip - random_flip
                    if swap_flip is not None and random_flip is not None
                    else None
                ),
                "random_minus_noop_flip": (
                    random_flip - noop_flip
                    if random_flip is not None and noop_flip is not None
                    else None
                ),
                "source": str(path),
            }
        )
    return rows


def fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# State-alias factorial summary",
        "",
        "| condition | token | seed | same acc | cross acc | probe acc | probe layer | clean n | swap flip | random flip | same-mal flip | noop flip | swap-random | random-noop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x["condition"], x["token"], x["seed"])):
        arms = row["arms"]
        lines.append(
            "| {condition} | {token} | {seed} | {same} | {cross} | {probe} | {layer} | {clean_n} | {swap} | {random} | {samemal} | {noop} | {specific} | {damage} |".format(
                condition=row["condition"],
                token=row["token"],
                seed=row["seed"],
                same=fmt(row["response_same_accuracy"]),
                cross=fmt(row["response_cross_accuracy"]),
                probe=fmt(row["state_probe_best_cross_accuracy"]),
                layer=fmt(row["state_probe_best_cross_layer"]),
                clean_n=fmt(row["clean_n"]),
                swap=fmt(arms["swap"]["flip_to_A_rate"]),
                random=fmt(arms["random"]["flip_to_A_rate"]),
                samemal=fmt(arms["samemal"]["flip_to_A_rate"]),
                noop=fmt(arms["noop"]["flip_to_A_rate"]),
                specific=fmt(row["swap_minus_random_flip"]),
                damage=fmt(row["random_minus_noop_flip"]),
            )
        )
    lines.extend(
        [
            "",
            "`swap-random` is the directed causal-specificity contrast. `random-noop` estimates nonspecific activation damage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input_root)
    if not rows:
        raise SystemExit(f"No gate_summary.json files found under {args.input_root}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"jobs": rows}, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(rows))
    print(f"summarized {len(rows)} jobs -> {args.output_md}")


if __name__ == "__main__":
    main()
