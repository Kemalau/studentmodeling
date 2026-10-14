#!/usr/bin/env python3
"""Render the user-supplied transfer table as a six-panel performance comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


FIGURE_DIR = Path(__file__).resolve().parents[4] / "paper_iclr" / "figures"
STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.edgecolor": "#B8BDC2",
    "axes.linewidth": 0.6,
    "text.color": "#202020",
    "axes.labelcolor": "#202020",
    "xtick.color": "#202020",
    "ytick.color": "#404040",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}


def build_figure(source: Path):
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["rows"]
    if len(rows) != 6:
        raise ValueError("The 2-by-3 figure requires exactly six capability rows.")
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharey=True)
        figure.subplots_adjust(
            left=0.075, right=0.985, bottom=0.085, top=0.87,
            wspace=0.35, hspace=0.85,
        )
        for panel_index, (axis, row) in enumerate(zip(axes.flat, rows)):
            scores = [row["task_only"], row["ussm"]]
            bars = axis.bar(
                [0, 1], scores, width=0.55,
                color=["#F2F5B2", "#ABE2EF"],
                edgecolor=["#A3A66F", "#679BA9"],
                linewidth=0.65, zorder=3,
            )
            for bar, score in zip(bars, scores):
                axis.annotate(
                    f"{score:.3f}",
                    (bar.get_x() + bar.get_width() / 2, score),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                )
            axis.text(
                0.5, 1.035, rf"$\Delta = {row['gain']:+.3f}$",
                transform=axis.transAxes, ha="center", va="bottom",
                fontsize=9, color="#343A3E",
            )
            axis.set_title(row["task"], pad=18)
            axis.set_axisbelow(True)
            axis.yaxis.grid(True, color="#E9ECEE", linewidth=0.5)
            axis.set_ylim(0, 1)
            axis.set_xlim(-0.6, 1.6)
            axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.1f}"))
            axis.set_xticks([0, 1], ["Task-only", "USSM"])
            axis.tick_params(axis="x", length=0, pad=5, labelsize=9)
            axis.tick_params(axis="y", length=2, width=0.6, labelsize=8, labelleft=True)
            axis.spines[["top", "right"]].set_visible(False)
            if panel_index % 3 == 0:
                axis.set_ylabel("Performance", labelpad=5)
    return figure, axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=FIGURE_DIR / "cross_task_transfer.source.json")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    figure, _ = build_figure(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for extension in ("pdf", "svg", "png"):
            output = args.output_dir / f"cross_task_transfer.{extension}"
            figure.savefig(output, dpi=300)
            print(output)
    plt.close(figure)


if __name__ == "__main__":
    main()