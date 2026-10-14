from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_equal_state_kd_open_t3_lr1e6_amlt import (  # noqa: E402
    load,
    validate_midpoint,
)


def test_midpoint_is_one_matched_4xa100_arm() -> None:
    path = ROOT / "amlt/feeds_layerwise_equal_state_kd_open_t3_lr1e6_20260803.yaml"
    value = load(path)
    validate_midpoint(value)
    job = value["jobs"][0]
    env = job["submit_args"]["env"]
    assert job["sku"] == "80G4-A100"
    assert env["LEARNING_RATE"] == "1e-6"
    assert env["SEGMENT_ROWS"] == "1000"
    assert env["TRAIN_ENTRYPOINT"].endswith("open_t3_selective_kd.py")


def test_midpoint_watcher_preserves_lr_provenance() -> None:
    text = (ROOT / "scripts/watch_layerwise_equal_state_kd_open_t3_lr1e6.sh").read_text()
    assert "SUMMARY_LEARNING_RATE=1e-6" in text
    assert "EXPERIMENT_DATE=20260803" in text
    assert "recovery1-seed42-20260803" in text
    assert "feeds_layerwise_equal_state_kd_open_t3_lr1e6_20260803.yaml" in text
    shared = (ROOT / "scripts/watch_layerwise_equal_state_path.sh").read_text()
    assert 'summary_lr_args=(--learning-rate "$summary_learning_rate")' in shared


def test_midpoint_rejects_unavailable_multinode_sku() -> None:
    path = ROOT / "amlt/feeds_layerwise_equal_state_kd_open_t3_lr1e6_20260803.yaml"
    value = copy.deepcopy(load(path))
    value["jobs"][0]["sku"] = "4x80G1-A100"
    value["jobs"][0]["process_count_per_node"] = 1
    with pytest.raises(ValueError, match="GPU/MPI/SLA drift"):
        validate_midpoint(value)
