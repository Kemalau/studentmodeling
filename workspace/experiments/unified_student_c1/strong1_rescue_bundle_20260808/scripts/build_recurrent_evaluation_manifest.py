#!/usr/bin/env python3
"""Build one audited recurrent-state evaluation manifest from a frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_unified_recurrent_manifest import audit_split
from build_unified_recurrent_manifest import build_split, sha256, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    rows, construction = build_split(
        args.input_jsonl, args.max_updates, args.seed, limit=None
    )
    write_jsonl(args.output_jsonl, rows)
    audit, _, _ = audit_split(args.output_jsonl)
    payload = {
        "version": 1,
        "status": "passed",
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256(args.input_jsonl),
        "construction": construction,
        "audit": audit,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
