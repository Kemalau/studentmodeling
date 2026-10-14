import copy

from scripts.summarize_state8_eedi_label_readout import summarize


def result(tag, swap, random, native=0.8, noop=0.8, same=0.8, passed=False):
    by_condition = {
        condition: {
            "n": 25,
            "to_donor_rate": 0.0,
            "to_recipient_rate": 0.0,
            "other_rate": 1.0,
            "valid_identifier_rate": 1.0,
        }
        for condition in ("native", "noop", "samemal", "random", "swap")
    }
    by_condition["swap"]["to_donor_rate"] = swap
    by_condition["random"]["to_donor_rate"] = random
    return {
        "stage": "state8_eedi_label_readout_c4_v1",
        "status": "complete",
        "diagnostic_only": True,
        "semantic_judge_authoritative": True,
        "exact_identifier_authoritative": False,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "student_response_visible_to_decoder": False,
        "target_candidates_visible_to_decoder": False,
        "num_pairs": 25,
        "readout_tag": tag,
        "diagnostic_pass": passed,
        "D_specific": swap - random,
        "swap_to_donor_rate": swap,
        "native_recipient_retention": native,
        "noop_recipient_retention": noop,
        "same_rule_recipient_retention": same,
        "random_matches_either_target_rate": random,
        "by_condition": by_condition,
        "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }


def test_summary_routes_successful_identity_readout_to_decoder_interface():
    values = [
        result("early-label-low-cp20", 0.72, 0.08, passed=True),
        result("early-label-high-cp20", 0.60, 0.10),
        result("causal-teacher-low-cp100-control", 0.12, 0.08),
    ]
    value = summarize(values)
    assert value["best_early_label_tag"] == "early-label-low-cp20"
    assert value["interface_diagnosis"] == (
        "state_identity_decodable_decoder_interface_is_next_bottleneck"
    )
    assert value["semantic_judge_authoritative"] is True


def test_summary_fails_closed_on_duplicate_matrix_coordinate():
    values = [
        result("early-label-low-cp20", 0.2, 0.1),
        result("early-label-low-cp20", 0.3, 0.1),
        result("causal-teacher-low-cp100-control", 0.1, 0.1),
    ]
    try:
        summarize(copy.deepcopy(values))
    except ValueError as error:
        assert "matrix drift" in str(error)
    else:
        raise AssertionError("Duplicate readout matrix should fail closed")
