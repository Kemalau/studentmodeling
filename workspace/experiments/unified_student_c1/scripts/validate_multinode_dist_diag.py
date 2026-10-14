#!/usr/bin/env python3
"""Validate the complete two-node 16-rank NCCL diagnostic receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(root: Path) -> dict:
    paths = sorted(root.rglob("rank-*.json"))
    if len(paths) != 16:
        raise ValueError(f"Expected 16 rank receipts, found {len(paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ranks = {int(row.get("rank", -1)) for row in rows}
    hosts = {str(row.get("hostname", "")) for row in rows}
    masters = {str(row.get("master_address", "")) for row in rows}
    checks = {
        "all_status_pass": all(row.get("status") == "pass" for row in rows),
        "all_global_ranks": ranks == set(range(16)),
        "world_size_16": all(int(row.get("world_size", -1)) == 16 for row in rows),
        "two_hosts": len(hosts) == 2 and "" not in hosts,
        "eight_local_ranks_per_host": all(
            {int(row["local_rank"]) for row in rows if str(row["hostname"]) == host}
            == set(range(8))
            for host in hosts
        ),
        "one_master": len(masters) == 1 and "" not in masters,
        "all_reduce_136": all(float(row.get("all_reduce_sum", -1)) == 136.0 for row in rows),
        "all_a100": all("A100" in str(row.get("device", "")) for row in rows),
    }
    payload = {
        "stage": "multinode_16rank_nccl_diagnostic",
        "pass": all(checks.values()),
        "checks": checks,
        "hosts": sorted(hosts),
        "master_address": next(iter(masters)) if len(masters) == 1 else None,
        "receipts": [str(path) for path in paths],
    }
    if not payload["pass"]:
        raise ValueError(f"16-rank diagnostic failed: {[k for k,v in checks.items() if not v]}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
