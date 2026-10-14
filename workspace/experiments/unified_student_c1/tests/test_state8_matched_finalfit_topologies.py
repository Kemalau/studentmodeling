from __future__ import annotations

from pathlib import Path

import yaml

from scripts.materialize_state8_matched_finalfit_topologies import materialize
from scripts.validate_state8_matched_finalfit_topology_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_materialized_matched_topologies_are_valid(tmp_path: Path) -> None:
    source = ROOT / "amlt/feeds_state8_unified_final_fit_20260805.yaml"
    for topology, sku, launcher in (
        ("world4", "80G4-A100", "run_state8_unified_final_fit_world4.sh"),
        ("world2", "80G2-A100", "run_state8_unified_final_fit_world2.sh"),
    ):
        output = tmp_path / f"{topology}.yaml"
        materialize(source, output, topology)
        value = yaml.safe_load(output.read_text())
        validate(value)
        assert len(value["jobs"]) == 1
        assert value["jobs"][0]["sku"] == sku
        assert launcher in str(value["jobs"][0]["command"])
