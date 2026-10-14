from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_matched_checkpoint2000_screen_amlt import validate  # noqa: E402


CONFIG = ROOT / "amlt/feeds_state8_matched_checkpoint2000_screen_20260810.yaml"
WATCHER = ROOT / "scripts/watch_state8_matched_checkpoint2000_screen.sh"


def test_template_and_each_recovery_subset_validate(tmp_path: Path) -> None:
    validate(CONFIG, allow_template=True)
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for job in value["jobs"]:
        selected = copy.deepcopy(value)
        selected["jobs"] = [job]
        amlt_dir = tmp_path / "amlt"
        amlt_dir.mkdir(exist_ok=True)
        path = amlt_dir / f"{job['name']}.yaml"
        path.write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
        validate(path, allow_template=True, expected_jobs={job["name"]})


def test_unknown_recovery_job_fails_closed(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["jobs"] = [value["jobs"][0]]
    amlt_dir = tmp_path / "amlt"
    amlt_dir.mkdir()
    path = amlt_dir / "one.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid expected job subset"):
        validate(path, allow_template=True, expected_jobs={"unknown"})


def test_watcher_is_receipt_bound_resumable_and_semantically_judged() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    for fragment in (
        "checkpoint_completion.json",
        "materialize_state8_matched_checkpoint2000_screen.py",
        "amlt/feeds_state8_matched_checkpoint2000_screen_generated_20260810.yaml",
        "passed.$job",
        "amlt resume",
        "classify_experiment_status",
        "merge_state8_diagnosis_c4_shards.py",
        "--judge-backend trapi",
        "summarize_state8_diagnosis_c4.py",
    ):
        assert fragment in text
