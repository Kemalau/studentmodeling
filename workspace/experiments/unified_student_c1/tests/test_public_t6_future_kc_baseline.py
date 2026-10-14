from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_public_t6_baselines import analyze, metrics as bootstrap_metrics  # noqa: E402
from train_public_t6_future_kc_baseline import audit_splits, features, metric_block  # noqa: E402


def row(source: str, student: str, example: str, label: str = "medium") -> dict:
    return {
        "dataset": source,
        "student_id": student,
        "example_id": example,
        "task": "estimate_mastery",
        "target_label": label,
        "history": [{"correctness": 1, "kc_ids": ["k"]}],
        "query": {"kc_ids": ["k"], "correctness": 0},
        "metadata": {
            "source": source,
            "question_id": "q",
            "skill_id": "k",
            "future_same_kc_outcomes": [0, 1],
        },
    }


def test_features_exclude_query_target_and_future_outcomes() -> None:
    value = features(row("a", "1", "train"))
    assert "future_same_kc_outcomes" not in value
    assert "correctness" not in value
    assert value["same_kc_rate"] == 1.0


def test_split_audit_prefixes_student_by_source() -> None:
    value = audit_splits(
        [row("a", "1", "train")],
        [row("b", "1", "valid")],
        [row("c", "1", "test")],
    )
    assert not any(value["split_leakage_checks"].values())
    assert value["future_outcomes_used_as_features"] is False


def test_metric_block_reports_ordinal_metrics() -> None:
    value = metric_block(["low", "medium", "high"], ["low", "high", "high"])
    assert value["accuracy"] == 2 / 3
    assert value["ordinal_mae"] == 1 / 3
    assert "quadratic_weighted_kappa" in value


def test_cluster_bootstrap_reports_paired_intervals() -> None:
    rows = []
    for index, gold in enumerate(("low", "medium", "high", "high")):
        rows.append(
            {
                "student_id": f"s{index // 2}",
                "source": "a",
                "gold": gold,
                "predictions": {
                    "source_majority": "high",
                    "better": gold,
                },
            }
        )
    value = analyze(rows, replicates=100, seed=1)
    assert value["rows"] == 4
    assert value["students"] == 2
    assert value["estimates"]["better"]["macro_f1"]["value"] == 1.0
    assert "better" in value["paired_vs_source_majority"]


def test_fast_bootstrap_metrics_match_sklearn_definitions() -> None:
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

    gold = ["low", "low", "medium", "high", "high", "high"]
    prediction = ["low", "medium", "medium", "medium", "high", "low"]
    value = bootstrap_metrics(gold, prediction)
    ordinal = {"low": 0, "medium": 1, "high": 2}
    assert value["macro_f1"] == f1_score(
        gold, prediction, labels=("low", "medium", "high"), average="macro", zero_division=0
    )
    assert value["balanced_accuracy"] == balanced_accuracy_score(gold, prediction)
    assert abs(
        value["quadratic_weighted_kappa"]
        - cohen_kappa_score(
            [ordinal[item] for item in gold],
            [ordinal[item] for item in prediction],
            weights="quadratic",
        )
    ) < 1e-12
