import json
import sys
from pathlib import Path

import pytest

from scripts.select_state8_exact_consolidation_milestones import (
    STEPS,
    candidate_config,
    select,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_state8_exact_diagnosis_curriculum_selection import (  # noqa: E402
    JOB as PARENT_JOB,
    fixture,
)


def test_milestone_candidates_are_diagnostic_until_empirically_qualified() -> None:
    assert STEPS == (200, 400, 600, 800)
    rows = [candidate_config(step) for step in STEPS]
    assert len({row["job"] for row in rows}) == 4
    assert len({row["stable_root"] for row in rows}) == 4
    assert all(row["finalfit_variant"] is None for row in rows)


def materialize_candidate(root: Path, step: int) -> None:
    fixture(root)
    parent = next((root / "download").glob(f"**/{PARENT_JOB}"))
    job = str(candidate_config(step)["job"])
    parent.rename(parent.with_name(job))
    artifact = next((root / "download").glob(f"**/{job}"))
    artifacts = json.loads(
        (artifact / "metrics/publish_receipt.json").read_text()
    )["artifact_sha256"]
    (artifact / "metrics/checkpoint_audit.json").write_text(json.dumps({
        "stage": "state8_exact_consolidation_milestone_audit_v1",
        "status": "complete",
        "checkpoint_step": step,
        "completed_optimizer_steps": step // 4,
        "planned_micro_steps": 1000,
        "planned_optimizer_steps": 250,
        "consolidation_fraction": step / 1000,
        "history_dropout_counts": {"1.0": step},
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "artifact_sha256": artifacts,
    }))
    (artifact / "metrics/completion.json").write_text(json.dumps({
        "stage": "state8_exact_consolidation_milestone_completion_v1",
        "status": "checkpoint_complete",
        "diagnostic": True,
        "early_stop_candidate_only": True,
        "real_examples_seen_once": False,
        "checkpoint_step": step,
        "sha256": artifacts,
    }))


def test_milestone_selector_passes_only_bound_early_stop_artifacts(tmp_path: Path) -> None:
    for step in STEPS:
        materialize_candidate(tmp_path / f"step{step}", step)
    result = select(tmp_path)
    assert result["pass"] is True
    assert result["selected"]["checkpoint_step"] == 800
    assert result["selected"]["diagnostic_only"] is True
    assert result["selected"]["finalfit_route_materialized"] is False


def test_milestone_selector_rejects_coordinate_drift(tmp_path: Path) -> None:
    for step in STEPS:
        materialize_candidate(tmp_path / f"step{step}", step)
    path = next((tmp_path / "step400/download").glob(
        "**/state8-exact-consolidation-step400-eval/metrics/checkpoint_audit.json"
    ))
    value = json.loads(path.read_text()); value["checkpoint_step"] = 200
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="coordinate drift"):
        select(tmp_path)
