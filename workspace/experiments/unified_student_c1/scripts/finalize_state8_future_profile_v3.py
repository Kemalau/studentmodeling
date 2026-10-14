#!/usr/bin/env python3
"""Create a fail-closed closeout receipt for the State8 ownership-v3 run.

Experiment completion and scientific-gate success are deliberately separate:
a completed, independently audited experiment may provide a negative C4 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarize_state8_future_profile_experiment import build, markdown


PASS_STATES = {"pass", "passed"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_f1(predicted: list[str], gold: list[str]) -> float:
    predicted_set, gold_set = set(predicted), set(gold)
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    overlap = len(predicted_set & gold_set)
    return 2 * overlap / (len(predicted_set) + len(gold_set))


def job_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        require(isinstance(payload, list) and payload, f"empty status file: {path}")
        for row in payload:
            rows.append(
                {
                    "experiment": row.get("experiment"),
                    "job_id": row.get("job_id"),
                    "job_name": row.get("job_name"),
                    "status": row.get("status"),
                    "duration": row.get("duration"),
                    "cluster": row.get("cluster"),
                    "sku": row.get("sku"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--lineage-status-json", type=Path, action="append", default=[])
    parser.add_argument("--profile-metrics", type=Path, required=True)
    parser.add_argument("--interface-metrics", type=Path, required=True)
    parser.add_argument("--interface-profiles", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--retention-metrics", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--adapter-model", type=Path, required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    args = parser.parse_args()

    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    status = read_json(args.status_json)
    require(isinstance(status, list) and status, "final Feeds status is empty")
    require(
        all(str(row.get("status", "")).casefold() in PASS_STATES for row in status),
        "final Feeds job did not pass",
    )
    require(
        all(row.get("experiment") == args.experiment for row in status),
        "status belongs to a different experiment",
    )

    profile = read_json(args.profile_metrics)
    interface = read_json(args.interface_metrics)
    retention = read_json(args.retention_metrics)
    audit = read_json(args.independent_audit)
    require(audit.get("status") == "pass" and not audit.get("failures"), "independent audit did not pass")
    require(interface.get("status") == "complete", "reusable-interface analysis is incomplete")
    require(interface.get("student_disjoint") is True, "reusable-interface split is not student-disjoint")
    require(interface.get("future_support_visible_to_readout") is False, "reusable readout exposes future support")
    require(
        interface.get("weakness_candidate_protocol") == "fixed_train_valid_supported_kcs",
        "weakness candidates were not frozen from train/validation support",
    )
    require(retention.get("sampled_rows") == 200, "retention evaluation does not contain 200 rows")
    require(retention.get("raw_history_visible_to_decoder") is False, "retention decoder exposes history")
    require(retention.get("persistent_history_kv_cache") is False, "retention decoder keeps history cache")

    expected_hashes = profile.get("artifact_sha256", {})
    actual_hashes = {
        "adapter_model": sha256(args.adapter_model),
        "recurrent_state": sha256(args.recurrent_state),
    }
    require(actual_hashes == expected_hashes, "downloaded checkpoint hashes do not match evaluated artifacts")
    require(audit.get("artifact_sha256") == expected_hashes, "audit hashes do not match evaluated artifacts")
    require(retention.get("artifact_sha256") == expected_hashes, "retention metrics use another checkpoint")

    profile_rows = read_jsonl(args.interface_profiles)
    require(
        len(profile_rows) == interface["students"]["test"],
        "reusable-profile row count does not match held-out students",
    )
    require(
        len({row.get("example_id") for row in profile_rows}) == len(profile_rows),
        "reusable-profile example IDs are missing or duplicated",
    )
    for index, row in enumerate(profile_rows, start=1):
        require(row.get("raw_history_visible_to_readout") is False, f"profile row {index} exposes history")
        require(row.get("student_id_visible_to_readout") is False, f"profile row {index} exposes student ID")
        require(row.get("future_support_visible_to_readout") is False, f"profile row {index} exposes future support")
        require(
            row.get("candidate_protocol") == "fixed_train_valid_supported_kcs",
            f"profile row {index} uses the wrong weakness-candidate protocol",
        )
    profile_f1 = sum(
        set_f1(row["predicted_weaknesses"], row["gold_future_weaknesses"])
        for row in profile_rows
    ) / len(profile_rows)
    require(
        abs(profile_f1 - interface["metrics"]["state8"]["weakness"]["weak_top2_macro_f1"]) <= 1e-12,
        "reusable-profile rows do not reproduce the reported weakness F1",
    )

    for split in ("train", "valid", "test"):
        feature = interface["feature_artifacts"][split]
        require(
            feature["checkpoint"]
            == {
                "adapter": expected_hashes["adapter_model"],
                "recurrent_state": expected_hashes["recurrent_state"],
            },
            f"{split} features belong to another checkpoint",
        )
        require(sha256(args.feature_dir / f"{split}.npz") == feature["npz_sha256"], f"{split}.npz hash mismatch")
        require(
            sha256(args.feature_dir / f"{split}.json") == feature["metadata_sha256"],
            f"{split}.json hash mismatch",
        )

    summary = build(profile, interface, retention, audit, args.experiment)
    require(summary["experiment_complete"] is True, "summary does not meet experiment-completion contract")

    legacy_files = {
        "independent_audit.json": "independent_audit_legacy_future_support_protocol.json",
        "reusable_interface_metrics.json": "reusable_interface_metrics_legacy_future_support_protocol.json",
        "terminal_summary.json": "terminal_summary_legacy_future_support_protocol.json",
    }
    for current_name, legacy_name in legacy_files.items():
        current = state / current_name
        legacy = state / legacy_name
        if current.is_file() and not legacy.exists():
            shutil.copy2(current, legacy)

    canonical_files = {
        "independent_audit.json": args.independent_audit,
        "reusable_interface_metrics.json": args.interface_metrics,
        "reusable_interface_readable_profiles.jsonl": args.interface_profiles,
    }
    for name, source in canonical_files.items():
        temporary = state / f"{name}.partial"
        shutil.copy2(source, temporary)
        os.replace(temporary, state / name)

    write_json_atomic(state / "final_experiment_summary.json", summary)
    markdown_path = state / "final_experiment_summary.md"
    markdown_partial = markdown_path.with_suffix(markdown_path.suffix + ".partial")
    markdown_partial.write_text(markdown(summary), encoding="utf-8")
    os.replace(markdown_partial, markdown_path)

    lineage = job_rows(args.lineage_status_json + [args.status_json])
    receipt = {
        "schema": "state8_future_profile_v3_closeout_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "completion_ready": True,
        "scientific_gate_pass": summary["c4_scientific_gate_pass"],
        "scientific_conclusion": summary["scientific_conclusion"],
        "completion_semantics": (
            "The run, download, hash verification, corrected-protocol reanalysis, "
            "and independent audit completed; this does not imply that C4 passed."
        ),
        "lineage": lineage,
        "final_jobs": status,
        "protocol": summary["protocol"],
        "artifact_sha256": actual_hashes,
        "artifact_sizes": {
            "adapter_model": args.adapter_model.stat().st_size,
            "recurrent_state": args.recurrent_state.stat().st_size,
        },
        "source_artifact_sha256": {
            "profile_metrics": sha256(args.profile_metrics),
            "interface_metrics": sha256(args.interface_metrics),
            "interface_profiles": sha256(args.interface_profiles),
            "retention_metrics": sha256(args.retention_metrics),
            "independent_audit": sha256(args.independent_audit),
            "evaluator_script": sha256(args.evaluator_script),
            "manifest_train": sha256(args.manifest_dir / "train.jsonl"),
            "manifest_valid": sha256(args.manifest_dir / "valid.jsonl"),
            "manifest_test": sha256(args.manifest_dir / "test.jsonl"),
        },
        "protocol_correction": {
            "reason": "The remote weakness readout selected candidates from each test student's future support.",
            "repair": "Reanalyze immutable downloaded features with one KC candidate mask fixed from train/validation support only.",
            "future_support_visible_to_readout": False,
            "training_or_feature_extraction_rerun": False,
        },
        "independent_audit": audit,
        "task_performance_scope": summary["task_performance_scope"],
    }
    write_json_atomic(state / "completion_receipt.json", receipt)

    terminal = {
        "experiment": args.experiment,
        "rows": status,
        "all_passed": True,
        "downloaded_artifact_hashes_match_metrics": True,
        "independent_audit_exit_code": 0,
        "independent_audit": audit,
        "scientific_gate_pass": summary["c4_scientific_gate_pass"],
        "completion_ready": True,
        "interpretation": "experiment_complete_with_negative_c4_result",
    }
    write_json_atomic(state / "terminal_summary.json", terminal)
    (state / "needs_attention").unlink(missing_ok=True)
    (state / "complete").touch()
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
