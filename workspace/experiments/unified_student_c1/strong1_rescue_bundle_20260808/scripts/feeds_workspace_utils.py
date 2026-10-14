#!/usr/bin/env python3
"""Validate the fixed Feeds workspace in human-readable AMLT tables."""

from __future__ import annotations

import argparse
import re
import sys


def workspace_row_exists(text: str, name: str, resource_group: str) -> bool:
    """Return true only for a Singularity row with the exact name and RG."""

    pattern = re.compile(
        rf"^{re.escape(name)}\s+sing\s+\S+\s+{re.escape(resource_group)}(?:\s+\S+)*\s*$"
    )
    return any(pattern.fullmatch(line.strip()) for line in text.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--resource-group", required=True)
    args = parser.parse_args()
    text = sys.stdin.read()
    if not workspace_row_exists(text, args.name, args.resource_group):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
