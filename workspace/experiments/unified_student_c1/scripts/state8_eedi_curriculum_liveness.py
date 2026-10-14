#!/usr/bin/env python3
"""Fail-safe liveness decision for final-quintile-causal State8 curricula."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


UNIVERSAL_HEALTH_CHECKS = (
    "finite_losses",
    "no_runtime_error_signature",
    "memory_below_a100_80gb",
)


def should_cancel(
    progress: dict[str, Any],
    *,
    stall_seconds: float,
    startup_stall_seconds: float,
    posttrain_stall_seconds: float,
) -> bool:
    """Cancel only on a universal fatal signal or a phase-aware real stall.

    ``causal_path_observed_when_round1_present`` is intentionally excluded:
    these curricula schedule their first causal pair at the final 20% boundary,
    unlike the position-zero schedule assumed by the generic monitor.
    """

    health = progress.get("health") or {}
    checks = health.get("checks") or {}
    unhealthy = any(checks.get(name) is False for name in UNIVERSAL_HEALTH_CHECKS)
    complete = (
        float(progress.get("progress_fraction") or 0.0) >= 1.0
        or int(progress.get("round") or -1)
        >= int(progress.get("total_rounds") or 5000)
    )
    awaiting = progress.get("phase") == "awaiting_first_training_record"
    threshold = (
        posttrain_stall_seconds
        if complete
        else startup_stall_seconds
        if awaiting
        else stall_seconds
    )
    stalled = float(progress.get("seconds_since_progress") or 0.0) >= threshold
    return unhealthy or stalled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("progress", type=Path)
    parser.add_argument("stall_seconds", type=float)
    parser.add_argument("startup_stall_seconds", type=float)
    parser.add_argument("posttrain_stall_seconds", type=float)
    args = parser.parse_args()
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    raise SystemExit(
        0
        if should_cancel(
            progress,
            stall_seconds=args.stall_seconds,
            startup_stall_seconds=args.startup_stall_seconds,
            posttrain_stall_seconds=args.posttrain_stall_seconds,
        )
        else 1
    )


if __name__ == "__main__":
    main()
