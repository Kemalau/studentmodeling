#!/usr/bin/env python3
"""Render the paper C4 figure directly from frozen experiment artifacts.

The figure deliberately separates three questions:

1. Does a matched donor-state intervention move behavior more than controls?
2. Does the effect replicate across seeds and require the CF1 curriculum?
3. Is the effective history-free carrier a final vector or a depth-indexed
   residual trajectory?

No values are hand-entered into the plot.  The script reads the original JSON
artifacts and writes a source-data receipt beside the vector/raster figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#8A8A8A",
    "light_gray": "#D5D5D5",
    "ink": "#202124",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_flip(summary: dict[str, Any], arm: str) -> float:
    return float(summary["clean"][arm]["flip_to_A_rate"])


def require_close(actual: float, expected: float, label: str) -> None:
    """Protect the paper figure from silently reading a different artifact."""
    if not np.isclose(actual, expected, atol=5e-7, rtol=0):
        raise ValueError(f"Unexpected {label}: got {actual:.9f}, expected {expected:.9f}")


def extract(project_root: Path) -> dict[str, Any]:
    results = project_root / "results"
    paths = {
        "canonical_seed42": results
        / "state_curriculum_comparison"
        / "state-curriculum-stage-a-alias-gate-20260727"
        / "qwen3-32b-state-curriculum-stage-a-cf1-alias-gate"
        / "gate"
        / "stage_a_alias_gate_summary.json",
        "factorial": results
        / "state_alias_factorial_monitor_20260729"
        / "summary.json",
        "layerwise_bridge": results
        / "layerwise_state_bridge"
        / "raw_20260823"
        / "recurrent-cf1-layerwise-state-bridge-c4-seed42-20260731"
        / "cf1-layerwise-state-bridge-compact-n150-seed42"
        / "gate"
        / "layerwise_state_bridge.json",
        "final_vector_bridge": results
        / "single_layer_state_bridge"
        / "raw_20260823"
        / "recurrent-cf1-single-vector-layer-bridge-c4-seed42-20260731"
        / "cf1-single-vector-layer63-c4-n150-seed42"
        / "gate"
        / "single_layer_state_bridge.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifact(s):\n" + "\n".join(missing))

    canonical = load_json(paths["canonical_seed42"])
    factorial = load_json(paths["factorial"])
    layerwise = load_json(paths["layerwise_bridge"])
    final_vector = load_json(paths["final_vector_bridge"])

    seed42_summary = canonical["state_swap_masked"]
    seed42_arms = {
        arm: clean_flip(seed42_summary, arm)
        for arm in ("swap", "random", "samemal", "noop")
    }
    seed42_n = int(seed42_summary["clean_n"])
    seed42_specificity = seed42_arms["swap"] - seed42_arms["random"]

    jobs = factorial["jobs"]
    if not isinstance(jobs, list):
        raise TypeError("factorial.jobs must be a list")
    vision_jobs = {
        (str(job["condition"]), int(job["seed"])): job
        for job in jobs
        if job.get("token") == "visionpad"
    }
    cf1 = {
        13: vision_jobs[("cf1", 13)],
        42: {
            "seed": 42,
            "condition": "cf1",
            "clean_n": seed42_n,
            "swap_minus_random_flip": seed42_specificity,
            "response_cross_accuracy": canonical["response_cross_masked"]["accuracy"],
            "state_probe_best_cross_accuracy": canonical["state_probe_masked"]
            ["positions"]["state"]["best_cross"]["accuracy"],
        },
        97: vision_jobs[("cf1", 97)],
    }
    original = {seed: vision_jobs[("original", seed)] for seed in (13, 42, 97)}

    layer_summary = layerwise["summary"]
    final_summary = final_vector["summary"]
    bridge = {
        "trajectory": {
            "D_specific": float(layer_summary["D_specific"]),
            "clean_n": int(layer_summary["clean_n"]),
            "state_shape": list(layerwise["state_shape"]),
            "raw_history_visible_to_decoder": bool(
                layerwise["raw_history_visible_to_decoder"]
            ),
            "persistent_history_kv_cache": bool(
                layerwise["persistent_history_kv_cache"]
            ),
        },
        "final_vector": {
            "D_specific": float(final_summary["D_specific"]),
            "clean_n": int(final_summary["clean_n"]),
            "state_shape": list(final_vector["state_shape"]),
            "raw_history_visible_to_decoder": bool(
                final_vector["raw_history_visible_to_decoder"]
            ),
            "persistent_history_kv_cache": bool(
                final_vector["persistent_history_kv_cache"]
            ),
        },
    }

    # These checks are intentional: a recovery run must be incorporated by
    # updating the source path and the figure text, not by silently overwriting
    # the historical preregistered result.
    require_close(seed42_arms["swap"], 0.4105263158, "seed-42 directed flip")
    require_close(seed42_arms["random"], 0.1894736842, "seed-42 random flip")
    require_close(seed42_arms["samemal"], 0.0421052632, "seed-42 same-rule flip")
    require_close(seed42_arms["noop"], 0.0, "seed-42 no-op flip")
    require_close(bridge["trajectory"]["D_specific"], 0.2061855670, "trajectory D")
    require_close(bridge["final_vector"]["D_specific"], 0.0, "final-vector D")
    if bridge["trajectory"]["state_shape"] != [64, 5120]:
        raise ValueError("The layerwise carrier is not the expected 64 x 5120 trajectory")
    if bridge["final_vector"]["state_shape"] != [5120]:
        raise ValueError("The single-layer carrier is not the expected 5120-vector")
    for carrier in bridge.values():
        if carrier["raw_history_visible_to_decoder"] or carrier["persistent_history_kv_cache"]:
            raise ValueError("History-free bridge invariant is violated")

    return {
        "definition": {
            "outcome": "flip_to_A_rate on native-stay_B clean recipients",
            "D_specific": "flip_to_A(swap) - flip_to_A(norm-matched Gaussian noise)",
            "gate": 0.15,
            "random_control": "independent norm-matched Gaussian residual noise, not a random student's state",
        },
        "canonical_seed42": {
            "clean_n": seed42_n,
            "arms": seed42_arms,
            "D_specific": seed42_specificity,
            "response_cross_accuracy": float(canonical["response_cross_masked"]["accuracy"]),
            "state_probe_cross_accuracy": float(
                canonical["state_probe_masked"]["positions"]["state"]["best_cross"]
                ["accuracy"]
            ),
        },
        "seed_replication": {
            "cf1": {
                str(seed): {
                    "D_specific": float(job["swap_minus_random_flip"]),
                    "clean_n": int(job["clean_n"]),
                    "response_cross_accuracy": float(job["response_cross_accuracy"]),
                    "state_probe_cross_accuracy": float(
                        job["state_probe_best_cross_accuracy"]
                    ),
                }
                for seed, job in cf1.items()
            },
            "original_only": {
                str(seed): {
                    "D_specific": float(job["swap_minus_random_flip"]),
                    "clean_n": int(job["clean_n"]),
                    "response_cross_accuracy": float(job["response_cross_accuracy"]),
                    "state_probe_cross_accuracy": float(
                        job["state_probe_best_cross_accuracy"]
                    ),
                }
                for seed, job in original.items()
            },
        },
        "history_free_bridge": bridge,
        "sources": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(length=3, width=0.8, color="#777777", labelcolor=COLORS["ink"])
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7, zorder=0)


def render(data: dict[str, Any], output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.3,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.38),
        gridspec_kw={"width_ratios": [1.15, 1.18, 0.92], "wspace": 0.44},
    )

    # (a) Frozen seed-42 causal controls.
    ax = axes[0]
    style_axis(ax)
    arm_keys = ["swap", "random", "samemal", "noop"]
    arm_labels = ["Directed", "Noise", "Same-rule", "No-op"]
    arm_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["gray"]]
    values = [data["canonical_seed42"]["arms"][key] for key in arm_keys]
    bars = ax.bar(np.arange(4), values, color=arm_colors, width=0.68, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["ink"],
        )
    ax.set_xticks(np.arange(4), arm_labels)
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylabel("Flip to donor response")
    ax.set_title("a   Seed-42 intervention controls", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.96,
        f"clean n={data['canonical_seed42']['clean_n']}\n$D_{{specific}}$={data['canonical_seed42']['D_specific']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=COLORS["ink"],
    )

    # (b) Curriculum x seed interaction, keeping the failed seed visible.
    ax = axes[1]
    style_axis(ax)
    conditions = ["original_only", "cf1"]
    x = np.array([0.0, 1.0])
    rng = np.random.default_rng(42097)
    jitter = {seed: float(rng.uniform(-0.035, 0.035)) for seed in (13, 42, 97)}
    for seed in (13, 42, 97):
        ys = [
            data["seed_replication"][condition][str(seed)]["D_specific"]
            for condition in conditions
        ]
        xs = x + jitter[seed]
        ax.plot(xs, ys, color="#BDBDBD", linewidth=1.0, zorder=2)
        ax.scatter(xs[0], ys[0], s=30, color=COLORS["gray"], edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(xs[1], ys[1], s=34, color=COLORS["blue"], edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(
            str(seed),
            (xs[1], ys[1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.9,
            color=COLORS["ink"],
        )
    gate = float(data["definition"]["gate"])
    ax.axhline(gate, color=COLORS["purple"], linestyle=(0, (4, 2)), linewidth=1.1, zorder=1)
    ax.text(0.02, gate + 0.008, "preregistered gate = 0.15", color=COLORS["purple"], fontsize=6.7)
    ax.axhline(0, color="#777777", linewidth=0.7, zorder=1)
    ax.set_xlim(-0.32, 1.38)
    ax.set_ylim(-0.055, 0.255)
    ax.set_xticks(x, ["Original-only", "CF1 curriculum"])
    ax.set_ylabel("$D_{specific}$")
    ax.set_title("b   Curriculum $\\times$ seed", loc="left", fontweight="bold")
    ax.text(0.0, 0.018, "0/3 pass", ha="center", va="bottom", fontsize=7.1, color=COLORS["gray"])
    ax.text(0.72, 0.237, "2/3 pass", ha="left", va="top", fontsize=7.1, color=COLORS["blue"])

    # (c) Carrier ablation with no history and no persistent history KV.
    ax = axes[2]
    style_axis(ax)
    carriers = ["trajectory", "final_vector"]
    carrier_values = [
        data["history_free_bridge"][name]["D_specific"] for name in carriers
    ]
    bars = ax.bar(
        [0, 1],
        carrier_values,
        color=[COLORS["blue"], COLORS["gray"]],
        width=0.62,
        zorder=3,
    )
    for i, (bar, value) in enumerate(zip(bars, carrier_values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value + 0.010, 0.010),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["ink"],
        )
    ax.axhline(gate, color=COLORS["purple"], linestyle=(0, (4, 2)), linewidth=1.1, zorder=1)
    trajectory = data["history_free_bridge"]["trajectory"]
    final_vector = data["history_free_bridge"]["final_vector"]
    ax.set_xticks(
        [0, 1],
        [
            f"64-layer\ntrajectory\n(n={trajectory['clean_n']})",
            f"Layer-63\nvector\n(n={final_vector['clean_n']})",
        ],
    )
    ax.set_ylim(-0.03, 0.255)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_ylabel("$D_{specific}$")
    ax.set_title("c   History-free carrier", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.27)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    default_project = Path(__file__).resolve().parent.parent
    default_output = default_project.parents[2] / "paper_iclr" / "figures"
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    data = extract(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "c4_causal_state.source.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_pdf = output_dir / "c4_causal_state.pdf"
    output_png = output_dir / "c4_causal_state.png"
    render(data, output_pdf, output_png)
    print(
        json.dumps(
            {
                "data": str(data_path),
                "pdf": str(output_pdf),
                "png": str(output_png),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
