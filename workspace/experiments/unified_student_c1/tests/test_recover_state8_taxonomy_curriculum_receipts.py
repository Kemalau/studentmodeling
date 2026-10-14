from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recover_state8_taxonomy_curriculum_receipts import (  # noqa: E402
    EXPECTED_DROPOUT_COUNTS,
    audit_history,
)


def history_rows() -> list[dict[str, object]]:
    rows = []
    schedule = (0.0, 0.25, 0.5, 0.75, 1.0)
    for index in range(5000):
        row: dict[str, object] = {
            "global_position": index,
            "example_id": f"example-{index}",
            "history_dropout": schedule[index // 1000],
            "task": "diagnose_misconception",
            "causal_pair_id": None,
        }
        if index < 250:
            row["causal_pair_id"] = f"pair-{index}"
            row["causal_objective_phase"] = "specificity_triplet"
        rows.append(row)
    return rows


def test_history_audit_requires_exact_epoch_curriculum_and_causal_support() -> None:
    audit = audit_history(history_rows())
    assert audit["history_rows"] == 5000
    assert audit["history_unique_examples"] == 5000
    assert audit["history_dropout_counts"] == EXPECTED_DROPOUT_COUNTS
    assert audit["causal_rows"] == 250
    assert audit["unique_causal_pairs"] == 250


@pytest.mark.parametrize("mutation", ("duplicate", "dropout", "causal"))
def test_history_audit_fails_closed_on_scientific_drift(mutation: str) -> None:
    rows = history_rows()
    if mutation == "duplicate":
        rows[-1]["example_id"] = rows[-2]["example_id"]
    elif mutation == "dropout":
        rows[-1]["history_dropout"] = 0.0
    else:
        rows[0]["causal_objective_phase"] = "directional"
    with pytest.raises(AssertionError):
        audit_history(rows)
