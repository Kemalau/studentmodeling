#!/usr/bin/env python3
"""Rewrite frozen teacher prompts to the unambiguous State8 natural schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from state8_natural_curriculum_view import natural_prompt_text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changed = 0
    sources: dict[str, int] = {}
    seen: set[str] = set()
    with source.open(encoding="utf-8") as input_handle, destination.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {source}:{line_number}")
            example_id = str(value.get("example_id") or "")
            if not example_id or example_id in seen:
                raise ValueError(f"Missing or duplicate example_id: {example_id!r}")
            seen.add(example_id)
            before = str(value.get("prompt") or value.get("direct_text") or "")
            after = natural_prompt_text(before)
            if "Output format: <misconception>" in after:
                raise ValueError(f"Legacy output contract survived: {example_id}")
            target = str(value.get("target") or "")
            for tag in ("label", "diagnosis", "evidence"):
                if target.count(f"<{tag}>") != 1 or target.count(f"</{tag}>") != 1:
                    raise ValueError(f"Target schema drift for {example_id}: {tag}")
            value["prompt"] = after
            if "direct_text" in value:
                value["direct_text"] = after
            metadata = dict(value.get("metadata") or {})
            metadata.update(
                output_schema="label_diagnosis_evidence_v1",
                legacy_answer_adjacent_contract_removed=True,
            )
            value["metadata"] = metadata
            output_handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
            changed += int(after != before)
            source_name = str(value.get("source") or "unknown")
            sources[source_name] = sources.get(source_name, 0) + 1
    if rows != 1494 or sources != {"eedi_mining": 894, "malrule": 600}:
        raise ValueError(f"Frozen teacher panel drift: rows={rows} sources={sources}")
    receipt = {
        "stage": "state8_teacher_clean_prompt_v2",
        "rows": rows,
        "changed_rows": changed,
        "source_counts": dict(sorted(sources.items())),
        "output_schema": "label_diagnosis_evidence_v1",
        "legacy_answer_adjacent_contracts": 0,
        "input_sha256": sha256(source),
        "output_sha256": sha256(destination),
    }
    destination.with_suffix(destination.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rewrite(args.input_jsonl, args.output_jsonl), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
