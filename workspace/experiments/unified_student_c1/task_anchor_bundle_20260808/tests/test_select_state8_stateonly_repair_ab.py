from scripts.select_state8_stateonly_repair_ab import score


def test_repair_score_prefers_semantic_t3_and_causal_specificity():
    groups = {
        name: {
            "n": 100, "llm_judge_n": 100, "llm_judge_accuracy": acc,
            "natural_schema_valid_rate": 1.0,
        }
        for name, acc in {
            "diagnose_misconception/eedi_mining/open_taxonomy_generation": 0.3,
            "diagnose_misconception/malrule/synthetic_cross_template_open_generation": 0.4,
            "diagnose_misconception/malrule/synthetic_same_template_open_generation": 0.5,
        }.items()
    }
    t3 = {"judge_backend": "trapi", "judge_errors": 0, "metrics_by_group": groups}
    c4 = {
        "semantic_judge_authoritative": True,
        "generation": {"num_pairs": 50},
        "semantic_D_specific": 0.3,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": 0.6}},
        "noop_recipient_retention": 0.8,
        "same_rule_recipient_retention": 0.7,
        "random_matches_either_stable_target_rate": 0.1,
    }
    value, checks = score(t3, c4)
    assert value > 0
    assert checks["t3_semantic_accuracy"] == 0.4
