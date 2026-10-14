from scripts.evaluate_state8_eedi_label_readout_c4 import (
    categorize,
    normalize_identifier,
    summarize,
)


def record(pair, condition, category, output=None):
    return {
        "pair_idx": pair,
        "condition": condition,
        "category": category,
        "predicted_identifier": "x" if category != "other" else "",
        "raw_output": output or f"{condition}-{category}",
    }


def test_identifier_normalization_is_fail_closed():
    assert normalize_identifier("2016") == "2016"
    assert normalize_identifier("<label>fractions.rule</label>") == "fractions.rule"
    assert normalize_identifier("2016.") == "2016"
    assert normalize_identifier("2016 because the state says so") == ""
    assert categorize("b", "a", "b") == "donor"
    assert categorize("a", "a", "b") == "recipient"


def test_summary_detects_clean_label_transport():
    rows = []
    for pair in range(1, 5):
        rows.extend(
            [
                record(pair, "native", "recipient", "a"),
                record(pair, "noop", "recipient", "a"),
                record(pair, "samemal", "recipient", "a"),
                record(pair, "random", "other", "z"),
                record(pair, "swap", "donor", "b"),
            ]
        )
    value = summarize(rows)
    assert value["diagnostic_only"] is True
    assert value["semantic_judge_authoritative"] is True
    assert value["exact_identifier_authoritative"] is False
    assert value["swap_to_donor_rate"] == 1.0
    assert value["D_specific"] == 1.0
    assert value["diagnostic_pass"] is True


def test_summary_rejects_incomplete_pair():
    rows = [record(1, condition, "other") for condition in ("native", "noop", "random", "swap")]
    try:
        summarize(rows)
    except ValueError as error:
        assert "condition drift" in str(error) or "incomplete" in str(error)
    else:
        raise AssertionError("Incomplete readout panel should fail closed")
