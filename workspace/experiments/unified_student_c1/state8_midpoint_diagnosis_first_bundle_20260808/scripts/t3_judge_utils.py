#!/usr/bin/env python3
"""Torch-free JSON/text helpers for the fixed T3 semantic judge."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return html.unescape(match.group(1).strip()) if match else None


def extract_diagnosis(text: str) -> str | None:
    """Accept the new natural-language schema and the legacy misconception tag."""
    return extract_tag(text, "diagnosis") or extract_tag(text, "misconception")


def natural_schema_valid(text: str) -> bool:
    """Require exactly one non-empty label, diagnosis, and evidence tag."""
    value = str(text or "")
    if re.search(r"<\s*misconception\b", value, flags=re.I):
        return False
    for tag in ("label", "diagnosis", "evidence"):
        matches = re.findall(
            rf"<{tag}>\s*(.*?)\s*</{tag}>", value, flags=re.I | re.S
        )
        if len(matches) != 1 or not html.unescape(matches[0]).strip():
            return False
    return True


def normalize_text(value: str | None) -> str:
    text = html.unescape(str(value or "")).casefold().strip()
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    return re.sub(r"\s+", "", text)


def summarize_t3(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("task") != "diagnose_misconception":
            raise ValueError("Torch-free T3 summary received a non-T3 record")
        grouped[
            (str(record["task"]), str(record["source"]), str(record["protocol"]))
        ].append(record)
    result: dict[str, Any] = {}
    for (task, source, protocol), group in sorted(grouped.items()):
        parsed = [extract_diagnosis(str(record.get("raw_output") or "")) for record in group]
        gold = [extract_diagnosis(str(record.get("target") or "")) for record in group]
        row: dict[str, Any] = {
            "n": len(group),
            "format_valid_rate": sum(value is not None for value in parsed) / max(1, len(group)),
            "normalized_exact_match": sum(
                normalize_text(prediction) == normalize_text(target)
                for prediction, target in zip(parsed, gold, strict=True)
            ) / max(1, len(group)),
            "full_taxonomy_retrieval_pending": True,
            "natural_schema_valid_rate": sum(
                natural_schema_valid(str(record.get("raw_output") or ""))
                for record in group
            )
            / max(1, len(group)),
        }
        judged = [record for record in group if "llm_judge_correct" in record]
        if judged:
            row.update(
                {
                    "llm_judge_accuracy": sum(bool(record["llm_judge_correct"]) for record in judged) / len(judged),
                    "llm_judge_n": len(judged),
                    "llm_judge_mean_confidence": sum(float(record["llm_judge_confidence"]) for record in judged) / len(judged),
                    "llm_judge_exact_shortcuts": sum(record.get("llm_judge_mode") == "exact_match_shortcut" for record in judged),
                    "llm_judge_api_or_cache": sum(record.get("llm_judge_mode") in {"api", "cache"} for record in judged),
                }
            )
        result[f"{task}/{source}/{protocol}"] = row
    return result


def redacted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = {
            key: value
            for key, value in record.items()
            if key not in {"raw_output", "target"}
        }
        if "raw_output" in record:
            raw = str(record["raw_output"])
            target = str(record.get("target") or "")
            item["raw_output_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            item["raw_output_length"] = len(raw)
            item["target_sha256"] = hashlib.sha256(target.encode("utf-8")).hexdigest()
        output.append(item)
    return output
