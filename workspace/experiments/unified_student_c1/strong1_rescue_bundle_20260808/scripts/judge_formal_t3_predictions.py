#!/usr/bin/env python3
"""Judge previously generated T3 predictions without loading the evaluated model."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from t3_judge_utils import (
    extract_diagnosis,
    normalize_text,
    read_jsonl,
    redacted_records,
    summarize_t3,
    write_jsonl,
)
from t3_llm_judge import (
    JUDGE_INSTRUCTIONS,
    RUBRIC_VERSION,
    config_from_args,
    judge_t3_records,
    write_audit_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-base-url", default="http://1.13.190.7:25242")
    parser.add_argument("--judge-workers", type=int, default=12)
    parser.add_argument(
        "--judge-backend", choices=("auto", "responses", "trapi"), default="auto"
    )
    parser.add_argument("--trapi-deployment")
    parser.add_argument("--judge-max-retries", type=int, default=12)
    parser.add_argument("--judge-timeout-seconds", type=int, default=60)
    parser.add_argument("--judge-cache", type=Path)
    parser.add_argument("--judge-audit-samples", type=int, default=100)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [record for path in args.predictions_jsonl for record in read_jsonl(path)]
    rows = read_jsonl(args.eval_jsonl)
    rows_by_id = {str(row["example_id"]): row for row in rows}
    missing = sorted(str(record["example_id"]) for record in records if str(record["example_id"]) not in rows_by_id)
    if missing:
        raise ValueError(f"Predictions contain {len(missing)} unknown example IDs, first={missing[0]}")
    if len({str(record["example_id"]) for record in records}) != len(records):
        raise ValueError("Duplicate example IDs across prediction files")
    records = [record for record in records if record.get("task") == "diagnose_misconception"]
    records.sort(
        key=lambda record: hashlib.sha256(
            f"{args.seed}:{record['example_id']}".encode("utf-8")
        ).hexdigest()
    )
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        records = records[: args.max_records]

    config = config_from_args(
        args.judge_model,
        args.judge_base_url,
        args.judge_workers,
        args.judge_backend,
        args.trapi_deployment,
    )
    if args.judge_max_retries <= 0 or args.judge_timeout_seconds <= 0:
        raise ValueError("Judge retries and timeout must be positive")
    config = replace(
        config,
        max_retries=args.judge_max_retries,
        timeout_seconds=args.judge_timeout_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    judge_t3_records(
        records,
        rows_by_id,
        config,
        args.judge_cache or args.output_dir / "t3_judge_cache.jsonl",
        extract_diagnosis,
        normalize_text,
    )
    result = {
        "judge_model": args.judge_model,
        "judge_effective_model": config.model,
        "judge_backend": config.backend,
        "judge_identity": config.identity,
        "judge_instance": config.trapi_instance if config.backend == "trapi" else None,
        "judge_api_version": config.trapi_api_version if config.backend == "trapi" else None,
        "judge_rubric": RUBRIC_VERSION,
        "judge_rubric_sha256": hashlib.sha256(
            JUDGE_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "prediction_files": [str(path) for path in args.predictions_jsonl],
        "eval_jsonl": str(args.eval_jsonl),
        "generation_rows": len(records),
        "judged_rows": sum("llm_judge_correct" in record for record in records),
        "judge_errors": sum("llm_judge_correct" not in record for record in records),
        "metrics_by_group": summarize_t3(records),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "predictions_redacted.jsonl", redacted_records(records))
    write_audit_sample(
        args.output_dir / "t3_judge_audit_sample.jsonl",
        records,
        args.judge_audit_samples,
        args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
