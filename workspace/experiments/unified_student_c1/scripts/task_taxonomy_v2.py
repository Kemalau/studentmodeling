#!/usr/bin/env python3
"""Canonical public task taxonomy for the unified student-model benchmark.

Internal task keys remain stable so historical manifests, checkpoint hashes,
and already-running jobs stay reproducible.  Public T labels follow the
capability progression requested for the paper: current behavior, diagnosis,
future behavior, persistence, then the appendix-only KC-performance proxy.
"""

from __future__ import annotations

from typing import Any


TASK_TAXONOMY_VERSION = "unified_student_tasks_v2_20260813"

TASKS: dict[str, dict[str, Any]] = {
    "T1": {
        "task_key": "predict_correctness",
        "name": "correctness prediction",
        "headline": True,
        "legacy_label": "T1",
    },
    "T2": {
        "task_key": "predict_response",
        "name": "concrete response prediction",
        "headline": True,
        "legacy_label": "T2",
    },
    "T3": {
        "task_key": "diagnose_misconception",
        "name": "misconception diagnosis",
        "headline": True,
        "legacy_label": "T3",
    },
    "T4": {
        "task_key": "predict_prospective_error",
        "name": "conditional prospective behavior prediction",
        "headline": True,
        "legacy_label": "T5",
    },
    "T5": {
        "task_key": "predict_error_persistence",
        "name": "strict error-persistence prediction",
        "headline": True,
        "legacy_label": "T6",
    },
    "T6": {
        "task_key": "estimate_mastery",
        "name": "future-KC-performance band",
        "headline": False,
        "legacy_label": "T4",
        "role": "appendix_proxy_not_latent_mastery",
    },
}

HEADLINE_TASKS = tuple(label for label, row in TASKS.items() if row["headline"])
APPENDIX_TASKS = tuple(label for label, row in TASKS.items() if not row["headline"])
TASK_KEY_TO_PUBLIC = {row["task_key"]: label for label, row in TASKS.items()}
LEGACY_TO_PUBLIC = {row["legacy_label"]: label for label, row in TASKS.items()}


def taxonomy_receipt() -> dict[str, Any]:
    return {
        "version": TASK_TAXONOMY_VERSION,
        "headline_tasks": list(HEADLINE_TASKS),
        "appendix_tasks": list(APPENDIX_TASKS),
        "public_to_internal": {
            label: row["task_key"] for label, row in TASKS.items()
        },
        "legacy_to_public": dict(LEGACY_TO_PUBLIC),
        "internal_keys_stable_for_artifact_compatibility": True,
    }


def public_task_metrics(task_metrics: dict[str, Any]) -> dict[str, Any]:
    missing = [
        TASKS[label]["task_key"]
        for label in HEADLINE_TASKS
        if TASKS[label]["task_key"] not in task_metrics
        and label != "T3"
    ]
    if missing:
        raise ValueError(f"Missing headline task metrics: {missing}")
    return {
        label: task_metrics[TASKS[label]["task_key"]]
        for label in HEADLINE_TASKS
        if label != "T3"
    }

