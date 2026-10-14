from pathlib import Path

import yaml

from scripts.validate_state8_exact_consolidation_milestones_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_exact_milestones_are_parallel_read_only_and_strict() -> None:
    config = ROOT / "amlt/feeds_state8_exact_checkpoint200_eval_20260808.yaml"
    validate(config)
    value = yaml.safe_load(config.read_text())
    assert {job["submit_args"]["env"]["MILESTONE_STEP"] for job in value["jobs"]} == {
        "200", "400", "600", "800",
    }
    assert all(job["sku"] == "80G1-A100" for job in value["jobs"])
    assert all(job["mpi"] is False for job in value["jobs"])


def test_exact_milestone_launcher_preserves_early_stop_provenance() -> None:
    source = (ROOT / "scripts/run_state8_exact_consolidation_milestone_eval.sh").read_text()
    for fragment in (
        "consolidation_fraction",
        "real_examples_seen_once':False",
        "early_stop_candidate_only':True",
        "consolidation_early_stop",
        "checkpoint_audit.json",
        "semantic_judgment_pending",
    ):
        assert fragment in source
