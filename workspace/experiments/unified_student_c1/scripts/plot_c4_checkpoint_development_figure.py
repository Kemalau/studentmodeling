#!/usr/bin/env python3
"""Plot strict CF1 checkpoint development curves from frozen summaries.

The figure keeps task learning, state decodability, and causal use separate.
Each line must come from one fully seeded training run evaluated at the same
six checkpoints.  The script intentionally refuses historical endpoint-only
artifacts, which cannot establish temporal ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_PROTOCOL = "cf1_visionpad_strict_checkpoint_development_v1"
EXPECTED_STEPS = [250, 500, 750, 1000, 1250, 1500]
SEED_PATTERN = re.compile(r"(?:seed|train_seed)[-_]?(13|42|97)(?:\D|$)", re.I)
COLORS = {13: "#0072B2", 42: "#D55E00", 97: "#009E73"}
INK = "#202124"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def infer_seed(path: Path, payload: dict[str, Any]) -> int:
    if "train_seed" in payload:
        seed = int(payload["train_seed"])
    else:
        matches = {int(value) for value in SEED_PATTERN.findall(str(path))}
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer exactly one training seed from path: {path}; "
                f"matches={sorted(matches)}"
            )
        seed = matches.pop()
    if seed not in COLORS:
        raise ValueError(f"Unexpected training seed {seed}: {path}")
    return seed


def validate_summary(path: Path, payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    protocol = payload.get("protocol")
    if protocol != EXPECTED_PROTOCOL:
        raise ValueError(f"Unexpected protocol {protocol!r}: {path}")
    steps = [int(value) for value in payload.get("steps", [])]
    if steps != EXPECTED_STEPS:
        raise ValueError(f"Unexpected checkpoints {steps}: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_STEPS):
        raise ValueError(f"Expected {len(EXPECTED_STEPS)} rows: {path}")
    row_steps = [int(row["step"]) for row in rows]
    if row_steps != EXPECTED_STEPS:
        raise ValueError(f"Row/checkpoint mismatch {row_steps}: {path}")

    unit_metrics = (
        "response_cross_accuracy",
        "state_probe_best_cross_accuracy",
        "swap_flip_to_A",
        "random_flip_to_A",
        "same_rule_flip_to_A",
        "noop_flip_to_A",
    )
    for row in rows:
        for key in unit_metrics:
            value = float(row[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} outside [0,1] at step {row['step']}: {value}")
        expected_d = float(row["swap_flip_to_A"]) - float(row["random_flip_to_A"])
        if not np.isclose(float(row["D_specific"]), expected_d, atol=5e-8, rtol=0):
            raise ValueError(
                f"D_specific mismatch at step {row['step']}: "
                f"reported={row['D_specific']} recomputed={expected_d}"
            )
        if int(row["clean_n"]) <= 0:
            raise ValueError(f"Non-positive clean_n at step {row['step']}: {row['clean_n']}")
        adapter_hash = str(row["checkpoint_adapter_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", adapter_hash):
            raise ValueError(f"Invalid adapter SHA256 at step {row['step']}: {adapter_hash}")
    return infer_seed(path, payload), rows


def discover(input_root: Path, required_seeds: set[int]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    candidates = sorted(input_root.rglob("curve_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No curve_summary.json below {input_root}")

    curves: dict[int, list[dict[str, Any]]] = {}
    sources: dict[str, Any] = {}
    for path in candidates:
        payload = load_json(path)
        if payload.get("protocol") != EXPECTED_PROTOCOL:
            continue
        seed, rows = validate_summary(path, payload)
        if seed not in required_seeds:
            continue
        if seed in curves:
            raise ValueError(
                f"Duplicate strict development summaries for seed {seed}: "
                f"{sources[str(seed)]['path']} and {path}"
            )
        curves[seed] = rows
        sources[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }

    missing = required_seeds - curves.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing strict development summaries for seeds {sorted(missing)} below {input_root}"
        )
    return curves, sources


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(length=3, width=0.8, color="#777777", labelcolor=INK)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7, zorder=0)


def render(source_data: dict[str, Any], output_pdf: Path, output_png: Path) -> None:
    curves = {
        int(seed): rows for seed, rows in source_data["curves"].items()
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharex=True)
    steps = np.asarray(EXPECTED_STEPS)
    panels = [
        (
            "response_cross_accuracy",
            "a   Predictive competence",
            "Cross-template response accuracy",
            (0.0, 1.03),
        ),
        (
            "state_probe_best_cross_accuracy",
            "b   State decodability",
            "Best cross-template probe accuracy",
            (0.0, 1.03),
        ),
        (
            "D_specific",
            "c   Causal use",
            "$D_{specific}$ (directed − noise)",
            (-0.13, 0.36),
        ),
    ]

    for ax, (metric, title, ylabel, ylim) in zip(axes, panels):
        style_axis(ax)
        for seed in sorted(curves):
            values = np.asarray([float(row[metric]) for row in curves[seed]])
            ax.plot(
                steps,
                values,
                color=COLORS[seed],
                marker="o",
                markersize=3.7,
                linewidth=1.25,
                label=f"seed {seed}",
                zorder=3,
            )
        ax.set_xlim(210, 1540)
        ax.set_ylim(*ylim)
        ax.set_xticks([250, 500, 750, 1000, 1250, 1500])
        ax.tick_params(axis="x", rotation=35)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")

    axes[1].axhline(1 / 22, color="#999999", linestyle=(0, (3, 2)), linewidth=0.9)
    axes[1].text(
        1532,
        1 / 22 + 0.018,
        "22-way chance",
        ha="right",
        va="bottom",
        color="#777777",
        fontsize=6.6,
    )
    axes[2].axhline(0.0, color="#777777", linewidth=0.75)
    axes[2].axhline(0.15, color="#CC79A7", linestyle=(0, (4, 2)), linewidth=1.0)
    axes[2].text(
        220,
        0.162,
        "preregistered gate = 0.15",
        ha="left",
        va="bottom",
        color="#CC79A7",
        fontsize=6.6,
    )
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.8)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.29, wspace=0.36)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    default_project = Path(__file__).resolve().parent.parent
    default_input = (
        default_project
        / "results"
        / "c4_strict3seed_and_curves_monitor_20260824"
        / "download"
    )
    default_output = default_project.parents[2] / "paper_iclr" / "figures"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--required-seeds", default="13,42,97")
    args = parser.parse_args()

    required_seeds = {int(value) for value in args.required_seeds.split(",") if value}
    if not required_seeds or not required_seeds <= COLORS.keys():
        raise ValueError(f"Unsupported required seeds: {sorted(required_seeds)}")
    curves, sources = discover(args.input_root.resolve(), required_seeds)
    source_data = {
        "definition": {
            "protocol": EXPECTED_PROTOCOL,
            "checkpoints": EXPECTED_STEPS,
            "response": "cross-template closed-candidate response accuracy",
            "probe": "best 22-way cross-template linear-probe accuracy across predeclared layers",
            "D_specific": "flip_to_A(directed donor trajectory) - flip_to_A(norm-matched Gaussian noise)",
            "gate": 0.15,
            "clean_subset": "checkpoint-specific native prediction equals recipient response B",
        },
        "curves": {str(seed): curves[seed] for seed in sorted(curves)},
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "c4_checkpoint_development.source.json"
    pdf_path = args.output_dir / "c4_checkpoint_development.pdf"
    png_path = args.output_dir / "c4_checkpoint_development.png"
    source_path.write_text(
        json.dumps(source_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render(source_data, pdf_path, png_path)
    print(
        json.dumps(
            {
                "source": str(source_path.resolve()),
                "pdf": str(pdf_path.resolve()),
                "png": str(png_path.resolve()),
                "seeds": sorted(curves),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
