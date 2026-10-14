#!/usr/bin/env python3
"""Insert the exact constant eight-token marker used by the Stage-B teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MARKER = "<|vision_pad|>" * 8


def transform(prompt: str) -> str:
    if "<|vision_pad|>" in prompt or "<STUDENT_STATE>" in prompt:
        raise ValueError("Teacher prompt already contains a state marker")
    anchor = "</student_history>"
    if prompt.count(anchor) != 1:
        raise ValueError("Teacher prompt must contain exactly one history close tag")
    return prompt.replace(anchor, f"{anchor}\n{MARKER}", 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    count = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open(encoding="utf-8") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row["prompt"] = transform(str(row["prompt"]))
            if "direct_text" in row:
                row["direct_text"] = row["prompt"]
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                teacher_constant_marker_count=8,
                teacher_constant_marker="vision_pad",
                marker_contains_sample_information=False,
            )
            row["metadata"] = metadata
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if count != 1494:
        raise ValueError(f"Expected the frozen 1494-row teacher panel, got {count}")
    receipt = {
        "stage": "state8_teacher_exact_marker_view_v1",
        "rows": count,
        "constant_marker_count": 8,
        "constant_marker": "vision_pad",
        "marker_contains_sample_information": False,
        "input_sha256": sha256(args.input_jsonl),
        "output_sha256": sha256(args.output_jsonl),
    }
    args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
