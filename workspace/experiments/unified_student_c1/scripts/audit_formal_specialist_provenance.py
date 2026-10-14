#!/usr/bin/env python3
"""Audit the frozen formal specialist baseline artifacts and denominators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {"T1": 10_913, "T2": 3_693, "T3": 1_494, "T4": 4_600, "T5": 976}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict[str, Any]:
    paths = {
        "irt_pfa": root / "irt_pfa.json",
        "bkt": root / "bkt.json",
        "dkt": root / "dkt.json",
        "akt": root / "akt.json",
        "t3_specialists": root / "t3_specialists.json",
        "task_specialists": root / "task_specialists.json",
    }
    values = {name: load(path) for name, path in paths.items()}
    assert int(values["irt_pfa"]["data"]["test"]) == EXPECTED["T1"]
    assert int(values["irt_pfa"]["rasch_irt"]["test"]["n"]) == EXPECTED["T1"]
    assert int(values["irt_pfa"]["pfa"]["test"]["n"]) == EXPECTED["T1"]
    assert int(values["dkt"]["data"]["test_rows"]) == EXPECTED["T1"]
    assert int(values["akt"]["data"]["test"]) == EXPECTED["T1"]
    assert int(values["akt"]["test"]["n"]) == EXPECTED["T1"]
    assert values["akt"]["architecture"] == "official pyKT AKT"
    t3 = values["t3_specialists"]
    assert int(t3["data"]["test"]) == EXPECTED["T3"]
    assert int(t3["bge_full_taxonomy_eedi"]["n"]) == 894
    assert int(t3["bge_full_taxonomy_eedi"]["n_taxonomy_labels"]) == 2586
    assert "no oracle candidates" in t3["bge_full_taxonomy_eedi"]["protocol"]
    assert int(t3["tfidf_linearsvc"]["by_source"]["malrule"]["n"]) == 600
    tasks = values["task_specialists"]
    assert int(tasks["T2"]["metrics"]["overall"]["n"]) == EXPECTED["T2"]
    assert int(tasks["T5"]["metrics"]["overall"]["n"]) == EXPECTED["T4"]
    assert int(tasks["T6"]["metrics"]["overall"]["n"]) == EXPECTED["T5"]
    return {
        "stage": "formal_specialist_provenance_audit_v1",
        "status": "complete",
        "task_taxonomy": {
            "T1": "predict_correctness",
            "T2": "predict_response",
            "T3": "diagnose_misconception",
            "T4": "predict_prospective_error",
            "T5": "predict_error_persistence",
        },
        "frozen_test_denominators": EXPECTED,
        "artifact_sha256": {name: digest(path) for name, path in paths.items()},
        "checks": {
            "t1_classical_full_support": True,
            "akt_official_pykt": True,
            "t3_eedi_full_taxonomy_no_oracle_candidates": True,
            "t3_malrule_full_support": True,
            "task_specific_t2_t4_t5_full_support": True,
        },
        "metric_compatibility_warning": (
            "T3 taxonomy specialists use exact label accuracy; open generations use "
            "independent semantic judgment and are not an identical-score comparison."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
