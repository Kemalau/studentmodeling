#!/usr/bin/env python3
"""Add the trainer's capacity keyword to the frozen one-slot wrapper.

The current trainer always passes ``state_token_count``.  Historical C3 runs
intentionally use the frozen one-address implementation, whose constructor
predates that keyword.  This materializer keeps the frozen implementation
byte-for-byte as its source and adds only a fail-closed compatibility shim:
``state_token_count=1`` is accepted, while every other capacity is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca"

OLD = """        operation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_token_id = int(state_token_id)
"""

NEW = """        operation_checkpointing: bool = False,
        state_token_count: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_token_id = int(state_token_id)
        if int(state_token_count) != 1:
            raise ValueError(
                "Frozen one-address wrapper requires state_token_count=1"
            )
        self.state_token_count = 1
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def materialize(source: Path, output: Path) -> dict[str, object]:
    raw = source.read_bytes()
    observed = digest(raw)
    if observed != SOURCE_SHA256:
        raise ValueError(
            f"Frozen wrapper source hash drift: {observed} != {SOURCE_SHA256}"
        )
    text = raw.decode("utf-8")
    if text.count(OLD) != 1:
        raise ValueError("Frozen wrapper constructor contract drift")
    rendered = text.replace(OLD, NEW)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    result = {
        "stage": "legacy_layerwise_state_count_compat_v1",
        "source_sha256": observed,
        "output_sha256": digest(rendered.encode("utf-8")),
        "accepted_state_token_count": 1,
        "capacity_change_allowed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = materialize(args.source, args.output)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
