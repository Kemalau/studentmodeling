#!/usr/bin/env python3
"""Validate and reuse the completed diagnosis-first State8 curriculum.

The exact pipeline is deliberately resumable across Feeds preemptions.  Once
the immutable curriculum has been published, a recovery must not spend hours
re-running the 5,000-step curriculum and 487-row formal screen.  This validator
binds the published model, state, curriculum receipts, and formal predictions
before allowing the pipeline to continue directly to state-only C4 repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_STAGE = (
    "state8_qualified_teacher_curriculum_highkd_immutable_v1_"
    "taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4"
)
EXPECTED_HISTORY_DROPOUT = {
    "0.0": 1000,
    "0.25": 1000,
    "0.5": 1000,
    "0.75": 1000,
    "1.0": 1000,
}


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prediction_rows(path: Path) -> tuple[int, int]:
    rows = 0
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Prediction row {line_number} is not an object")
            example_id = str(value.get("example_id") or "")
            if not example_id:
                raise ValueError(f"Prediction row {line_number} has no example_id")
            if example_id in identifiers:
                raise ValueError(f"Duplicate prediction example_id={example_id}")
            identifiers.add(example_id)
            rows += 1
    return rows, len(identifiers)


def validate(root: Path) -> dict[str, Any]:
    model = root / "model"
    metrics = root / "metrics"
    formal = root / "eval/formal_screen"
    paths = {
        "adapter": model / "adapter_model.safetensors",
        "adapter_config": model / "adapter_config.json",
        "state": model / "recurrent_state.pt",
        "immutable": metrics / "immutable_receipt.json",
        "completion": metrics / "completion.json",
        "qualified": metrics / "qualified_teacher_receipt.json",
        "diagnosis": metrics / "diagnosis_first_teacher_curriculum.json",
        "formal_metrics": formal / "metrics.json",
        "formal_predictions": formal / "predictions.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete immutable curriculum: missing={missing}")

    immutable = read_object(paths["immutable"])
    completion = read_object(paths["completion"])
    qualified = read_object(paths["qualified"])
    diagnosis = read_object(paths["diagnosis"])
    formal_metrics = read_object(paths["formal_metrics"])

    expected_artifacts = immutable.get("sha256") or {}
    observed_artifacts = {
        "adapter_model": digest(paths["adapter"]),
        "recurrent_state": digest(paths["state"]),
    }
    checks = {
        "immutable_contract": immutable.get("status") == "complete"
        and immutable.get("stage") == EXPECTED_STAGE
        and immutable.get("replay_variant") == "t3_heavy"
        and immutable.get("state_shape") == [64, 8, 5120]
        and int(immutable.get("micro_steps", -1)) == 5000
        and immutable.get("raw_history_visible_to_decoder") is False
        and immutable.get("persistent_history_kv_cache") is False
        and immutable.get("semantic_judge_authoritative") is True
        and immutable.get("exact_label_metric_authoritative") is False
        and immutable.get("diagnosis_field_order")
        == ["diagnosis", "evidence", "label"]
        and immutable.get("canonical_label_role") == "auxiliary_ordinary_weight"
        and float(immutable.get("label_auxiliary_weight", -1)) == 1.0,
        "artifact_hashes": expected_artifacts == observed_artifacts
        and set(observed_artifacts) == {"adapter_model", "recurrent_state"},
        "completion_contract": completion.get("status") == "complete"
        and completion.get("diagnostic") is False
        and int(completion.get("micro_steps", -1)) == 5000
        and completion.get("real_examples_seen_once") is True
        and completion.get("all_causal_pairs_covered") is True
        and completion.get("causal_exposure_count_matches_plan") is True
        and {
            key: (completion.get("sha256") or {}).get(key)
            for key in observed_artifacts
        }
        == observed_artifacts,
        "qualified_contract": qualified.get("status") == "complete"
        and qualified.get("stage") == "state8_qualified_teacher_curriculum_v1"
        and qualified.get("history_dropout_counts") == EXPECTED_HISTORY_DROPOUT
        and qualified.get("state_shape") == [64, 8, 5120]
        and qualified.get("raw_history_visible_to_decoder") is False
        and qualified.get("semantic_judge_authoritative") is True
        and qualified.get("exact_label_metric_authoritative") is False
        and qualified.get("adapter_sha256") == observed_artifacts["adapter_model"]
        and qualified.get("recurrent_state_sha256")
        == observed_artifacts["recurrent_state"],
        "diagnosis_contract": diagnosis.get("status") == "complete"
        and diagnosis.get("stage") == "state8_diagnosis_first_teacher_curriculum_v1"
        and diagnosis.get("history_dropout_schedule") == [0.0, 0.25, 0.5, 0.75, 1.0]
        and diagnosis.get("field_order") == ["diagnosis", "evidence", "label"]
        and diagnosis.get("semantic_fields_primary") is True
        and diagnosis.get("canonical_label_role") == "auxiliary_ordinary_weight"
        and diagnosis.get("semantic_judge_authoritative") is True
        and diagnosis.get("exact_label_metric_authoritative") is False
        and diagnosis.get("raw_history_visible_to_decoder") is False
        and diagnosis.get("final_inference_route") == "strict_state_only",
        "formal_contract": formal_metrics.get("artifact_sha256") == observed_artifacts
        and formal_metrics.get("state_shape") == [64, 8, 5120]
        and formal_metrics.get("raw_history_visible_to_decoder") is False
        and formal_metrics.get("persistent_history_kv_cache") is False
        and int(formal_metrics.get("unsharded_filtered_rows", -1)) == 21676
        and int(formal_metrics.get("sampled_rows", -1)) == 487
        and int(immutable.get("formal_screen_rows", -1)) == 487,
    }
    rows, unique_rows = prediction_rows(paths["formal_predictions"])
    checks["formal_predictions_complete"] = rows == unique_rows == 487
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Immutable curriculum validation failed: {failed}")
    return {
        "stage": "state8_exact_immutable_curriculum_reuse_v1",
        "status": "complete",
        "root": str(root),
        "artifact_sha256": observed_artifacts,
        "formal_screen_rows": rows,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        partial.write_text(rendered, encoding="utf-8")
        partial.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
