#!/usr/bin/env python3
"""Build a sanitized, protocol-aware T1/T2/T3/T5/T6 test manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


STATE_TOKEN = "<STUDENT_STATE>"
TASK_ALIASES = {
    "malrule_next_correctness": "predict_correctness",
    "malrule_next_response": "predict_response",
    "malrule_identity": "diagnose_misconception",
}
EVALUATION_TASKS = {
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
}
QUERY_GOLD_FIELDS = {
    "correctness",
    "correct_response",
    "correct_response_id",
    "first_response_correctness",
    "unsupported_first_attempt_success",
    "hint_count",
    "saw_answer",
    "gold_misconception_id",
    "gold_misconception_name",
    "malrule_answer",
}
METADATA_GOLD_FIELDS = {
    "target_correctness",
    "target_response",
    "target_response_id",
    "target_response_hash",
    "misconception_id",
    "malrule_id",
    "query_answers_by_malrule",
}


def source_files(processed_dir: Path) -> list[tuple[str, str, Path]]:
    return [
        ("dbe_kt22", "temporal_test", processed_dir / "dbe_kt22_t1_t6/test.jsonl"),
        ("malrule", "cross_template", processed_dir / "malrule_c1_clean/test_cross_template.jsonl"),
        ("malrule", "same_template", processed_dir / "malrule_c1_clean/test_same_template.jsonl"),
        ("eedi_mining", "official_fold_4", processed_dir / "eedi_mining_c1_ir/test.jsonl"),
        ("eedi_response", "profile_heldout", processed_dir / "eedi_response_t1_t2/test.jsonl"),
        ("xes3g5m", "official_temporal_test", processed_dir / "xes3g5m_t1_t6/test.jsonl"),
        ("foundationalassist", "temporal_test", processed_dir / "foundationalassist_core_tasks/test.jsonl"),
    ]


def protocol(task: str, source: str, source_protocol: str) -> str | None:
    if task == "predict_correctness":
        if source == "eedi_response":
            return "profile_heldout_binary"
        if source == "malrule":
            return f"synthetic_{source_protocol}_binary"
        return "temporal_warm_item_binary"
    if task == "predict_response":
        if source in {"dbe_kt22", "eedi_response"}:
            return "closed_choice_generation"
        if source == "malrule":
            return f"synthetic_{source_protocol}_open_generation"
        if source == "foundationalassist":
            return "structured_math_open_generation"
    if task == "diagnose_misconception":
        if source == "eedi_mining":
            return "open_taxonomy_generation"
        if source == "malrule":
            return f"synthetic_{source_protocol}_open_generation"
    if task == "predict_prospective_error":
        if source == "dbe_kt22":
            return "delta3_closed_choice_generation"
        if source == "foundationalassist":
            return "delta_open_generation"
    if task == "predict_error_persistence":
        if source in {"dbe_kt22", "foundationalassist"}:
            return "strict_exact_recurrence_binary"
        return None
    return None


def malrule_open_prompt(row: dict[str, Any], task: str) -> str:
    history = row.get("history") or []
    query = row.get("query") or {}
    if task == "predict_response":
        instruction = (
            "You are modeling one student's mathematical behavior. Predict the exact answer this same student "
            "would give on the next problem. Return exactly one <predicted_response> tag."
        )
        output_rule = "Output format: <predicted_response>student's exact response</predicted_response>"
    elif task == "diagnose_misconception":
        instruction = (
            "You are modeling one student's mathematical behavior. Diagnose which systematic misconception "
            "best explains the student's history. Return exactly one <misconception> tag."
        )
        output_rule = "Output format: <misconception>systematic misconception</misconception>"
    else:
        return str(row["prompt"])
    history_lines = []
    for index, event in enumerate(history, 1):
        history_lines.extend(
            [
                f"Trial {index}:",
                f"Problem: {event['problem_text']}",
                f"Student answer: {event['student_answer']}",
                f"Correct answer: {event['correct_answer']}",
                f"Outcome: {'correct' if event['is_correct'] else 'incorrect'}",
                "",
            ]
        )
    history_text = "\n".join(history_lines).rstrip()
    return (
        f"{instruction}\n"
        "<student_history>\n"
        f"{history_text}\n"
        "</student_history>\n"
        f"{STATE_TOKEN}\n"
        "<query>\n"
        f"Problem: {query['problem_text']}\n"
        f"Correct answer: {query['correct_answer']}\n"
        "</query>\n"
        f"{output_rule}\n"
        "Answer:"
    )


def sanitize(row: dict[str, Any], source: str, source_protocol: str) -> dict[str, Any] | None:
    original_task = str(row.get("task") or "")
    task = TASK_ALIASES.get(original_task, original_task)
    if task not in EVALUATION_TASKS:
        return None
    task_protocol = protocol(task, source, source_protocol)
    if task_protocol is None:
        return None
    result = dict(row)
    original_id = str(result["example_id"])
    result["example_id"] = f"formaleval:{source}:{source_protocol}:{original_id}"
    result["dataset"] = source
    result["split"] = "test"
    result["task"] = task
    result["student_id"] = f"{source}::{result.get('student_id', '')}"
    query = dict(result.get("query") or {})
    for field in QUERY_GOLD_FIELDS:
        query.pop(field, None)
    if task != "predict_error_persistence":
        query.pop("student_response", None)
        query.pop("student_response_id", None)
    result["query"] = query
    metadata = dict(result.get("metadata") or {})
    for field in METADATA_GOLD_FIELDS:
        metadata.pop(field, None)
    metadata.update(
        {
            "eval_source": source,
            "eval_protocol": task_protocol,
            "source_test_protocol": source_protocol,
            "original_example_id": original_id,
            "original_task": original_task,
            "visible_input_sanitized": True,
        }
    )
    result["metadata"] = metadata
    if source == "malrule" and task in {"predict_response", "diagnose_misconception"}:
        result["prompt"] = malrule_open_prompt(row, task)
        result["candidate_labels"] = []
        result["candidate_targets"] = {}
    elif task == "diagnose_misconception":
        result["candidate_labels"] = []
        result["candidate_targets"] = {}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    counts = Counter()
    seen_ids: set[str] = set()
    pending: list[dict[str, Any]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def flush(output_handle) -> None:
        if not pending:
            return
        prompts = [str(row["prompt"]).replace(STATE_TOKEN, args.state_token_alias) for row in pending]
        prompt_lengths = tokenizer(prompts, add_special_tokens=False, return_length=True)["length"]
        for row, prompt_length in zip(pending, prompt_lengths):
            source = str(row["metadata"]["eval_source"])
            task = str(row["task"])
            if prompt_length >= args.max_length:
                counts[f"dropped_prompt_length/{task}/{source}"] += 1
                continue
            example_id = str(row["example_id"])
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example_id: {example_id}")
            seen_ids.add(example_id)
            output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts["kept"] += 1
            counts[f"kept_task/{task}"] += 1
            counts[f"kept_source/{source}"] += 1
            counts[f"kept_protocol/{row['metadata']['eval_protocol']}"] += 1
            counts[f"kept_task_source/{task}/{source}"] += 1
        pending.clear()

    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for source, source_protocol, path in source_files(args.processed_dir):
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    if not line.strip():
                        continue
                    counts["input_rows"] += 1
                    row = sanitize(json.loads(line), source, source_protocol)
                    if row is None:
                        counts["excluded_rows"] += 1
                        continue
                    counts["eligible_rows"] += 1
                    pending.append(row)
                    if len(pending) >= args.batch_size:
                        flush(output_handle)
        flush(output_handle)

    stats = {
        "output_jsonl": str(args.output_jsonl),
        "tokenizer": args.tokenizer,
        "max_prompt_length": args.max_length - 1,
        "included_tasks": sorted(EVALUATION_TASKS),
        "excluded_protocols": ["all T4 rows", "XES3G5M repeat-incorrectness proxy"],
        "counts": dict(sorted(counts.items())),
        "unique_example_ids": len(seen_ids),
    }
    args.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
