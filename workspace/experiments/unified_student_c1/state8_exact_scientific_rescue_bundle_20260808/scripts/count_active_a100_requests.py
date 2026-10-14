#!/usr/bin/env python3
"""Count currently running A100 requests in the checked-out AMLT project."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def json_table(command: list[str]) -> list[dict]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after 90s: {command}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {command}\n{result.stderr}")
    start = result.stdout.find("[")
    if start < 0:
        raise ValueError(f"missing JSON table: {command}\n{result.stdout}")
    value = json.loads(result.stdout[start:])
    if not isinstance(value, list):
        raise ValueError(f"invalid JSON table: {command}")
    return value


def count(limit: int, recent: int) -> dict:
    experiments = json_table(
        ["amlt", "--json-tables", "list", "-s", "running", "-n", str(recent)]
    )
    names = sorted({str(row["experiment_name"]) for row in experiments})
    jobs: list[dict] = []
    unreadable: list[dict[str, str]] = []

    def inspect(name: str) -> tuple[str, list[dict] | None, Exception | None]:
        command = [
            "amlt",
            "--json-tables",
            "status",
            name,
            "-s",
            "running",
            "--hide-urls",
        ]
        try:
            return name, json_table(command), None
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            return name, None, exc

    # Status calls are independent and dominate this audit's wall time when
    # the account has many active experiments.  Bound concurrency so topology
    # selection remains responsive without creating a burst of unbounded AMLT
    # processes.  ``executor.map`` preserves the sorted input order, keeping
    # the receipt deterministic even though requests execute concurrently.
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(names)))) as executor:
        inspected = list(executor.map(inspect, names))

    for name, rows, error in inspected:
        if error is not None:
            # One stale or malformed unrelated AMLT experiment must not make
            # the complete-pipeline supervisor crash.  It is also unsafe to
            # assume that an unreadable experiment consumes zero A100s, so
            # preserve the known-job audit while closing the available-slot
            # estimate until every running experiment is readable again.
            unreadable.append(
                {
                    "experiment": name,
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                }
            )
            continue
        assert rows is not None
        for row in rows:
            sku = str(row.get("sku") or "")
            match = re.fullmatch(r"(?:\d+x)?80G(\d+)-A100(?:-.*)?", sku)
            gpu_count = int(match.group(1)) if match else 0
            if gpu_count:
                jobs.append(
                    {
                        "experiment": name,
                        "job_name": row.get("job_name"),
                        "job_id": row.get("job_id"),
                        "sku": sku,
                        "gpu_count": gpu_count,
                    }
                )
    jobs.sort(key=lambda row: (str(row["experiment"]), str(row["job_name"])))
    running = sum(int(row["gpu_count"]) for row in jobs)
    auditable = not unreadable
    return {
        "stage": "active_a100_request_audit_v1",
        "user_limit": limit,
        "running_a100_requests": running,
        "available_user_slots": max(0, limit - running) if auditable else 0,
        "auditable": auditable,
        "unreadable_experiments": unreadable,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-limit", type=int, default=32)
    parser.add_argument("--recent", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.user_limit <= 0 or args.recent <= 0:
        raise ValueError("limits must be positive")
    result = count(args.user_limit, args.recent)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
