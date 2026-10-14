from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_state8_semantic_label_view import materialize, transform_row
from scripts.state8_semantic_label import (
    eedi_semantic_label,
    replace_natural_label,
    semantic_label,
)


def natural(identifier: str, diagnosis: str) -> str:
    return (
        f"<label>{identifier}</label>\n"
        f"<diagnosis>{diagnosis}</diagnosis>\n"
        "<evidence>observed wrong response</evidence>"
    )


def test_eedi_semantic_label_is_readable_unique_and_taxonomy_bound() -> None:
    a = eedi_semantic_label("1847", "Confuses the part and the whole in fraction problems")
    b = eedi_semantic_label("9", "Confuses the part and the whole in fraction problems")
    assert a == "eedi.confuses_the_part_and_the_whole_in_fraction_problems_id_1847"
    assert a != b
    assert semantic_label(
        {
            "source": "eedi_mining",
            "target_label": "1847",
            "target": "<misconception>Confuses the part and the whole in fraction problems</misconception>",
        }
    ) == a


def test_natural_label_replacement_does_not_change_diagnosis_or_evidence() -> None:
    source = natural("1847", "Confuses part and whole")
    changed = replace_natural_label(source, "eedi.confuses_part_and_whole_id_1847")
    assert "<label>eedi.confuses_part_and_whole_id_1847</label>" in changed
    assert "<diagnosis>Confuses part and whole</diagnosis>" in changed
    assert "<evidence>observed wrong response</evidence>" in changed


def test_transform_preserves_numeric_ids_and_rewrites_pair_targets() -> None:
    row = {
        "source": "eedi_mining",
        "task": "diagnose_misconception",
        "target_label": "1847",
        "target": natural("1847", "Confuses part and whole"),
        "causal_target_kind": "diagnosis_natural_schema",
        "native_target_label": "1847",
        "donor_target_label": "952",
        "native_target": natural("1847", "Confuses part and whole"),
        "donor_target": natural("952", "Adds the index to the number"),
    }
    transformed, changed = transform_row(row)
    assert changed is True
    assert transformed["target_label"] == "1847"
    assert transformed["taxonomy_id"] == "1847"
    assert transformed["native_taxonomy_id"] == "1847"
    assert transformed["donor_taxonomy_id"] == "952"
    assert transformed["native_semantic_label"].endswith("_id_1847")
    assert transformed["donor_semantic_label"].endswith("_id_952")
    assert transformed["native_semantic_label"] in transformed["native_target"]
    assert transformed["donor_semantic_label"] in transformed["donor_target"]


def test_materialized_receipt_binds_source_and_output(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    receipt = tmp_path / "receipt.json"
    source.write_text(
        json.dumps(
            {
                "source": "eedi_mining",
                "task": "diagnose_misconception",
                "target_label": "1847",
                "target": natural("1847", "Confuses part and whole"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    value = materialize(source, output, receipt)
    assert value["status"] == "passed"
    assert value["changed_rows"] == 1
    assert value["numeric_taxonomy_id_preserved"] is True
    assert json.loads(receipt.read_text()) == value

