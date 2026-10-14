#!/usr/bin/env python3
"""Evaluate Base and Unified T3 generations with one model load and an LLM judge."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from evaluate_c1 import load_model
from evaluate_formal_protocols import (
    extract_tag,
    generate_rows,
    normalize_text,
    read_jsonl,
    redacted_records,
    sample_rows,
    summarize,
    write_jsonl,
)
from t3_llm_judge import config_from_args, judge_t3_records, write_audit_sample


def save_result(
    output_dir: Path,
    model_path: str,
    adapter_path: str | None,
    eval_path: Path,
    rows: list[dict],
    records: list[dict],
    audit_samples: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_path,
        "adapter": adapter_path,
        "eval_jsonl": str(eval_path),
        "sampled_rows": len(rows),
        "classification_rows": 0,
        "generation_rows": len(records),
        "metrics_by_group": summarize(records),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "predictions_redacted.jsonl", redacted_records(records))
    write_audit_sample(output_dir / "t3_judge_audit_sample.jsonl", records, audit_samples, seed)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


def judge_and_save(
    name: str,
    records: list[dict],
    rows: list[dict],
    args,
    adapter_path: str | None,
) -> None:
    output_dir = args.output_root / name
    config = config_from_args(args.judge_model, args.judge_base_url, args.judge_workers)
    judge_t3_records(
        records,
        {str(row["example_id"]): row for row in rows},
        config,
        output_dir / "t3_judge_cache.jsonl",
        lambda value: extract_tag(value, "misconception"),
        normalize_text,
    )
    save_result(
        output_dir,
        args.model,
        adapter_path,
        args.eval_jsonl,
        rows,
        records,
        args.judge_audit_samples,
        args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples-per-group", type=int)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-base-url", default="http://1.13.190.7:25242")
    parser.add_argument("--judge-workers", type=int, default=12)
    parser.add_argument("--judge-audit-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.eval_jsonl) if row["task"] == "diagnose_misconception"]
    rows = sample_rows(rows, args.max_samples_per_group, args.seed)
    tokenizer, base_model = load_model(args.model, None, load_4bit=True)

    base_records = generate_rows(
        tokenizer,
        base_model,
        rows,
        args.max_length,
        args.max_new_tokens,
        args.generation_batch_size,
        args.state_token_alias,
    )
    judge_and_save("base_diagnose_misconception", base_records, rows, args, None)
    del base_records
    gc.collect()

    adapter_model = PeftModel.from_pretrained(base_model, args.adapter)
    adapter_model.eval()
    adapter_records = generate_rows(
        tokenizer,
        adapter_model,
        rows,
        args.max_length,
        args.max_new_tokens,
        args.generation_batch_size,
        args.state_token_alias,
    )
    judge_and_save("adapter_diagnose_misconception", adapter_records, rows, args, args.adapter)
    del adapter_records, adapter_model, base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
