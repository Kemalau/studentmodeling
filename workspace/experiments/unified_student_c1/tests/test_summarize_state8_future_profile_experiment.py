from scripts.summarize_state8_future_profile_experiment import weighted_task_summary


def test_weighted_task_summary_uses_sample_counts() -> None:
    result = weighted_task_summary({
        "metrics_by_group": {
            "predict_correctness/a": {"n": 2, "accuracy": 0.5, "auroc": 0.25},
            "predict_correctness/b": {"n": 6, "accuracy": 1.0, "auroc": 0.75},
            "diagnose_misconception/a": {
                "n": 4, "normalized_exact_match": 0.25,
                "full_taxonomy_retrieval_pending": True,
            },
        }
    })
    assert result["predict_correctness"]["n"] == 8
    assert result["predict_correctness"]["accuracy"] == 0.875
    assert result["predict_correctness"]["auroc"] == 0.625
    assert result["diagnose_misconception"]["full_evaluation_pending_groups"] == 1
