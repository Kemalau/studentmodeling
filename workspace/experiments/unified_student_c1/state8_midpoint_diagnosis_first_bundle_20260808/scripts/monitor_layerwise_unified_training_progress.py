#!/usr/bin/env python3
"""Extract the newest structured training record from an AMLT rank log."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def canonical_training_record(value: Any) -> dict[str, Any] | None:
    """Normalize distributed ``round`` and single-GPU ``step`` log schemas."""
    if not isinstance(value, dict) or "optimizer_step" not in value:
        return None
    if "round" in value:
        return value
    if "step" in value:
        result = dict(value)
        result["round"] = value["step"]
        result["progress_unit"] = "step"
        return result
    return None


def latest_record(text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = canonical_training_record(value)
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError("No structured training record found")
    return max(records, key=lambda row: (int(row["round"]), int(row.get("rank", 0))))


def structured_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = canonical_training_record(value)
        if record is not None:
            records.append(record)
    records.sort(key=lambda row: (int(row["round"]), int(row.get("rank", 0))))
    return records


def health_summary(text: str) -> dict[str, Any]:
    records = structured_records(text)
    if not records:
        raise ValueError("No structured training record found")
    loss_fields = (
        "loss",
        "generation_loss",
        "behavior_kd_loss",
        "consistency_loss",
        "direct_loss",
        "causal_loss",
    )
    nonfinite: list[dict[str, Any]] = []
    for row in records:
        for field in loss_fields:
            value = row.get(field)
            if value is not None and not math.isfinite(float(value)):
                nonfinite.append({"round": int(row["round"]), "field": field, "value": str(value)})
    causal = [row for row in records if row.get("causal_pair_id") is not None]
    max_allocated = max(float(row.get("peak_cuda_allocated_gib") or 0.0) for row in records)
    max_reserved = max(float(row.get("peak_cuda_reserved_gib") or 0.0) for row in records)
    lower = text.lower()
    runtime_errors = [
        pattern
        for pattern in (
            "cuda out of memory",
            "nccl watchdog",
            "traceback (most recent call last)",
            "runtimeerror:",
        )
        if pattern in lower
    ]
    checks = {
        "finite_losses": not nonfinite,
        "no_runtime_error_signature": not runtime_errors,
        "memory_below_a100_80gb": max_allocated < 79.0 and max_reserved < 79.0,
        # The deterministic schedule starts at global position zero. Rank-zero
        # log tails that contain round one must therefore show at least one
        # causal record; later tails may legitimately omit it.
        "causal_path_observed_when_round1_present": (
            any(int(row["round"]) == 1 for row in records) is False or bool(causal)
        ),
    }
    return {
        "record_count": len(records),
        "round_min": min(int(row["round"]) for row in records),
        "round_max": max(int(row["round"]) for row in records),
        "causal_record_count": len(causal),
        "causal_pair_ids": sorted({str(row["causal_pair_id"]) for row in causal}),
        # AMLT progress monitoring reads a bounded rank-zero log tail.  In a
        # distributed run the causal row at a scheduled global position may be
        # owned by any rank, and old causal rows may also fall outside the
        # downloaded tail.  Therefore zero observed rows is never evidence of
        # a missing causal schedule.  The authoritative check is the frozen
        # distributed-checkpoint audit, which validates every history row
        # against the complete causal_schedule map.
        "causal_observation_scope": "bounded_rank_zero_log_tail",
        "causal_absence_is_schedule_failure": False,
        "causal_schedule_authority": "distributed_checkpoint_audit",
        "peak_cuda_allocated_gib": max_allocated,
        "peak_cuda_reserved_gib": max_reserved,
        "nonfinite": nonfinite,
        "runtime_error_signatures": runtime_errors,
        "checks": checks,
        "healthy": all(checks.values()),
        "advisory_only": True,
    }


def attach_fatal_log_health(
    result: dict[str, Any], text: str
) -> dict[str, Any]:
    """Make deterministic terminal logs visible even without a JSON row.

    A failed resume can terminate before emitting a structured training record.
    In that case the log-gap path intentionally preserves the previous round,
    but it must not preserve a healthy status when the launcher has already
    reported a fail-closed scientific-recipe error.
    """

    lower = text.lower()
    signatures = [
        pattern
        for pattern in (
            "distributed resume scientific recipe drift",
            "state8_curriculum_failed line=",
            "failed line=",
        )
        if pattern in lower
    ]
    if not signatures:
        return result
    health = dict(result.get("health") or {})
    checks = dict(health.get("checks") or {})
    checks["no_runtime_error_signature"] = False
    existing = list(health.get("runtime_error_signatures") or [])
    health.update(
        {
            "checks": checks,
            "healthy": False,
            "runtime_error_signatures": sorted(set(existing + signatures)),
            "advisory_only": True,
        }
    )
    result["health"] = health
    result["phase"] = "fatal_runtime_error"
    return result


def awaiting_first_record(
    *,
    experiment: str,
    log_file: str,
    status: str,
    previous: dict[str, Any] | None = None,
    now: datetime | None = None,
    total_rounds: int | None = None,
) -> dict[str, Any]:
    """Describe a live run whose rank log has not emitted a training row yet."""

    now = now or datetime.now(timezone.utc)
    same_run = bool(previous) and previous.get("experiment") == experiment
    if same_run:
        attempt_value = (
            previous.get("attempt_started_at")
            or previous.get("observed_at")
            or now.isoformat()
        )
        attempt_started_at = datetime.fromisoformat(str(attempt_value))
        if attempt_started_at.tzinfo is None:
            attempt_started_at = attempt_started_at.replace(tzinfo=timezone.utc)
    else:
        attempt_started_at = now
    running_started_at: datetime | None = None
    if status == "running":
        if same_run and previous.get("running_started_at") is not None:
            running_started_at = datetime.fromisoformat(
                str(previous["running_started_at"])
            )
            if running_started_at.tzinfo is None:
                running_started_at = running_started_at.replace(tzinfo=timezone.utc)
        else:
            running_started_at = now
    seconds_since_progress = (
        max(0.0, (now - running_started_at).total_seconds())
        if running_started_at is not None
        else 0.0
    )
    result: dict[str, Any] = {
        "experiment": experiment,
        "log_file": log_file,
        "amlt_status": status,
        "observed_at": now.isoformat(),
        "attempt_started_at": attempt_started_at.isoformat(),
        "attempt_elapsed_seconds": max(
            0.0, (now - attempt_started_at).total_seconds()
        ),
        "phase": "awaiting_first_training_record",
        "round": None,
        "optimizer_step": None,
        "advanced": False,
        "seconds_since_progress": seconds_since_progress,
        "health": None,
    }
    if running_started_at is not None:
        result["running_started_at"] = running_started_at.isoformat()
    if total_rounds is not None:
        if total_rounds <= 0:
            raise ValueError("Invalid total round count")
        result["total_rounds"] = total_rounds
    return result


def preserve_progress_during_log_gap(
    previous: dict[str, Any],
    *,
    experiment: str,
    log_file: str,
    status: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Keep the last real round when AMLT briefly returns an empty log tail.

    AMLT log retrieval can succeed while returning no structured rows during
    retry transitions or blob refreshes.  Replacing a real progress receipt
    with an ``awaiting_first`` receipt would reset the stall clock and could
    hide a genuinely wedged training attempt.  Preserve the last evidence and
    advance only the observation/stall timestamps until a newer row appears.
    """

    if previous.get("experiment") != experiment or previous.get("round") is None:
        raise ValueError("No prior structured training progress to preserve")
    now = now or datetime.now(timezone.utc)
    result = dict(previous)
    value = previous.get("last_progress_at") or previous.get("observed_at")
    last_progress_at = datetime.fromisoformat(str(value))
    if last_progress_at.tzinfo is None:
        last_progress_at = last_progress_at.replace(tzinfo=timezone.utc)
    result.update(
        {
            "experiment": experiment,
            "log_file": log_file,
            "amlt_status": status,
            "observed_at": now.isoformat(),
            "advanced": False,
            "phase": "temporary_log_gap",
            "log_gap": True,
            "seconds_since_progress": max(
                0.0, (now - last_progress_at).total_seconds()
            ),
        }
    )
    attempt_value = previous.get("attempt_started_at")
    if attempt_value is not None:
        attempt_started_at = datetime.fromisoformat(str(attempt_value))
        if attempt_started_at.tzinfo is None:
            attempt_started_at = attempt_started_at.replace(tzinfo=timezone.utc)
        result["attempt_elapsed_seconds"] = max(
            0.0, (now - attempt_started_at).total_seconds()
        )
        if "max_run_seconds" in result:
            result["remaining_attempt_seconds"] = max(
                0.0,
                float(result["max_run_seconds"])
                - result["attempt_elapsed_seconds"],
            )
    return result


def summarize(
    text: str,
    *,
    experiment: str,
    log_file: str,
    status: str,
    previous: dict[str, Any] | None = None,
    now: datetime | None = None,
    total_rounds: int | None = None,
    max_run_seconds: int | None = None,
    validation_reserve_seconds: int = 0,
) -> dict[str, Any]:
    row = latest_record(text)
    now = now or datetime.now(timezone.utc)
    fields = (
        "round",
        "optimizer_step",
        "step",
        "global_position",
        "rank",
        "loss",
        "generation_loss",
        "behavior_kd_loss",
        "consistency_loss",
        "direct_loss",
        "causal_loss",
        "causal_pair_id",
        "peak_cuda_allocated_gib",
        "peak_cuda_reserved_gib",
        "progress_unit",
    )
    result = {key: row.get(key) for key in fields}
    result.update(
        {
            "experiment": experiment,
            "log_file": log_file,
            "amlt_status": status,
            "observed_at": now.isoformat(),
        }
    )
    result["health"] = health_summary(text)
    same_run = bool(previous) and previous.get("experiment") == experiment
    previous_round = previous.get("round") if same_run else None
    has_previous_round = previous_round is not None
    advanced = not has_previous_round or int(row["round"]) > int(previous_round)
    if advanced:
        last_progress_at = now
    else:
        value = previous.get("last_progress_at") or previous.get("observed_at")
        last_progress_at = datetime.fromisoformat(str(value))
        if last_progress_at.tzinfo is None:
            last_progress_at = last_progress_at.replace(tzinfo=timezone.utc)
    result["last_progress_at"] = last_progress_at.isoformat()
    result["seconds_since_progress"] = max(0.0, (now - last_progress_at).total_seconds())
    result["advanced"] = advanced
    if same_run:
        attempt_value = (
            previous.get("attempt_started_at")
            or previous.get("observed_at")
            or previous.get("last_progress_at")
            or now.isoformat()
        )
        attempt_started_at = datetime.fromisoformat(
            str(attempt_value)
        )
        if attempt_started_at.tzinfo is None:
            attempt_started_at = attempt_started_at.replace(tzinfo=timezone.utc)
    else:
        attempt_started_at = now
    result["attempt_started_at"] = attempt_started_at.isoformat()
    result["attempt_elapsed_seconds"] = max(
        0.0, (now - attempt_started_at).total_seconds()
    )
    if total_rounds is not None:
        if total_rounds <= 0 or int(row["round"]) > total_rounds:
            raise ValueError("Invalid total round count")
        result["total_rounds"] = total_rounds
        result["progress_fraction"] = int(row["round"]) / total_rounds
        if same_run and has_previous_round:
            baseline_value = previous.get("rate_baseline_round")
            baseline_round = int(
                previous_round if baseline_value is None else baseline_value
            )
            baseline_at = datetime.fromisoformat(
                str(previous.get("rate_baseline_at", previous["observed_at"]))
            )
            if baseline_at.tzinfo is None:
                baseline_at = baseline_at.replace(tzinfo=timezone.utc)
        else:
            baseline_round = int(row["round"])
            baseline_at = now
        result["rate_baseline_round"] = baseline_round
        result["rate_baseline_at"] = baseline_at.isoformat()
        elapsed = max(0.0, (now - baseline_at).total_seconds())
        completed = int(row["round"]) - baseline_round
        if elapsed > 0 and completed > 0:
            rounds_per_hour = completed * 3600.0 / elapsed
            eta_seconds = (total_rounds - int(row["round"])) * 3600.0 / rounds_per_hour
            result["rounds_per_hour"] = rounds_per_hour
            result["eta_seconds"] = eta_seconds
            result["eta_at"] = (now + timedelta(seconds=eta_seconds)).isoformat()
            if max_run_seconds is not None:
                if max_run_seconds <= 0:
                    raise ValueError("Invalid maximum run duration")
                if validation_reserve_seconds < 0:
                    raise ValueError("Invalid validation reserve")
                result["max_run_seconds"] = max_run_seconds
                result["validation_reserve_seconds"] = validation_reserve_seconds
                remaining_attempt_seconds = max(
                    0.0, max_run_seconds - result["attempt_elapsed_seconds"]
                )
                result["remaining_attempt_seconds"] = remaining_attempt_seconds
                result["single_attempt_duration_risk"] = (
                    eta_seconds + validation_reserve_seconds
                    > remaining_attempt_seconds
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument("--awaiting-first-record", action="store_true")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-rounds", type=int)
    parser.add_argument("--max-run-seconds", type=int)
    parser.add_argument("--validation-reserve-seconds", type=int, default=0)
    args = parser.parse_args()
    previous = None
    if args.output.is_file():
        value = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            previous = value
    if args.awaiting_first_record:
        result = awaiting_first_record(
            experiment=args.experiment,
            log_file=args.log_file,
            status=args.status,
            previous=previous,
            total_rounds=args.total_rounds,
        )
    else:
        if args.log is None:
            parser.error("--log is required unless --awaiting-first-record is set")
        text = args.log.read_text(encoding="utf-8", errors="replace")
        try:
            result = summarize(
                text,
                experiment=args.experiment,
                log_file=args.log_file,
                status=args.status,
                previous=previous,
                total_rounds=args.total_rounds,
                max_run_seconds=args.max_run_seconds,
                validation_reserve_seconds=args.validation_reserve_seconds,
            )
        except ValueError as error:
            if str(error) != "No structured training record found":
                raise
            if (
                previous
                and previous.get("experiment") == args.experiment
                and previous.get("round") is not None
            ):
                result = preserve_progress_during_log_gap(
                    previous,
                    experiment=args.experiment,
                    log_file=args.log_file,
                    status=args.status,
                )
            else:
                result = awaiting_first_record(
                    experiment=args.experiment,
                    log_file=args.log_file,
                    status=args.status,
                    previous=previous,
                    total_rounds=args.total_rounds,
                )
        result = attach_fatal_log_health(result, text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
