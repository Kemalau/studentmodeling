from scripts.select_state8_semantic_candidates import GROUPS, score


def t3(eedi=0.2, cross=0.5, same=0.6):
    values = dict(zip(GROUPS, (eedi, cross, same), strict=True))
    return {
        "judge_backend": "trapi",
        "judge_errors": 0,
        "metrics_by_group": {
            name: {
                "n": 100,
                "llm_judge_n": 100,
                "llm_judge_api_or_cache": 100,
                "llm_judge_accuracy": value,
                "llm_judge_exact_shortcuts": 0,
                "natural_schema_valid_rate": 1.0,
            }
            for name, value in values.items()
        },
    }


def c4(d=0.2, swap=0.5, random=0.2):
    return {
        "semantic_judge_authoritative": True,
        "generation": {"num_pairs": 50},
        "semantic_D_specific": d,
        "cells": {"swap_to_donor": {"semantic_equivalence_rate": swap}},
        "noop_recipient_retention": 0.8,
        "same_rule_recipient_retention": 0.7,
        "random_matches_either_stable_target_rate": random,
    }


def test_qualified_candidate_maps_to_finalfit_job() -> None:
    value = score("eedi_g025", t3(), c4())
    assert value["qualified"] is True
    assert value["selection_eligible"] is True
    assert value["finalfit_variant"] == "semantic_eedi_g025"
    assert value["finalfit_job"] == "state8-semantic-eedi-g025-unified-finalfit-seed42"


def test_single_a100_candidate_keeps_its_seed46_checkpoint_identity() -> None:
    value = score("eedi_g025_single", t3(), c4())
    assert value["qualified"] is True
    assert value["selection_eligible"] is False
    assert value["finalfit_variant"] == "semantic_eedi_g025_single_seed46"
    assert value["finalfit_job"] == (
        "state8-semantic-eedi-g025-single-seed46-unified-finalfit-seed42"
    )


def test_candidate_fails_if_eedi_or_c4_missing() -> None:
    assert score("c4_g010", t3(eedi=0.05), c4())["qualified"] is False
    assert score("c4_g010", t3(eedi=0.16), c4())["qualified"] is False
    assert score("c4_g010", t3(), c4(d=0.05))["qualified"] is False
    assert score("c4_g010", t3(), c4(swap=0.39))["qualified"] is False
    assert score("c4_g010", t3(), c4(random=0.31))["qualified"] is False


def test_candidate_rejects_exact_match_or_missing_semantic_judgments() -> None:
    exact_shortcut = t3()
    group = next(iter(exact_shortcut["metrics_by_group"].values()))
    group["llm_judge_api_or_cache"] = 99
    group["llm_judge_exact_shortcuts"] = 1
    try:
        score("c4_g010", exact_shortcut, c4())
    except ValueError as error:
        assert "independent semantic judgments" in str(error)
    else:
        raise AssertionError("Exact-match shortcut must fail closed")

    missing = t3()
    next(iter(missing["metrics_by_group"].values())).pop(
        "llm_judge_api_or_cache"
    )
    try:
        score("c4_g010", missing, c4())
    except ValueError as error:
        assert "independent semantic judgments" in str(error)
    else:
        raise AssertionError("Missing semantic judgments must fail closed")
