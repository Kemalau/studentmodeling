import json
from pathlib import Path

import pytest

from scripts.compare_state8_c4_milestones import compare


def write(path: Path, records: list[dict], *, manifest_content: str = "frozen") -> None:
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(manifest_content)
    path.write_text(
        json.dumps(
            {
                "state_shape": [64, 8, 5120],
                "raw_history_visible_to_decoder": False,
                "pair_manifest": str(manifest),
                "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
                "records": records,
            }
        )
    )


def pair(index: int, samemal: str, *, native: str = "stay_B") -> list[dict]:
    output = []
    for condition in ("native", "noop", "samemal", "random", "swap"):
        output.append(
            {
                "pair_idx": index,
                "condition": condition,
                "category": samemal if condition == "samemal" else native,
                "family": "subtraction",
                "recipient_label": "subtraction.borrow_from_bottom",
                "donor_label": "subtraction.smaller_from_larger",
                "margin_A_minus_B": -0.5 if samemal == "stay_B" else 0.5,
            }
        )
    return output


def test_compare_finds_same_rule_regression_and_family(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    write(before, pair(1, "stay_B"))
    write(after, pair(1, "flip_to_A"))
    result = compare(before, after)
    assert result["common_clean_n"] == 1
    assert result["same_rule_regression_n"] == 1
    assert result["same_rule_regression_family_counts"] == {"subtraction": 1}


def test_compare_rejects_pair_manifest_drift(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    write(before, pair(1, "stay_B"), manifest_content="a")
    write(after, pair(1, "stay_B"), manifest_content="b")
    with pytest.raises(ValueError, match="different frozen pair manifests"):
        compare(before, after)
