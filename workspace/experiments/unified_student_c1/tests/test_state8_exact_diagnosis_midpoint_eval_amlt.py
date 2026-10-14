from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_state8_exact_diagnosis_midpoint_eval_amlt",
    ROOT / "scripts" / "validate_state8_exact_diagnosis_midpoint_eval_amlt.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_config() -> dict:
    return yaml.safe_load(
        (ROOT / "amlt" / "feeds_state8_exact_diagnosis_midpoint_eval_20260809.yaml").read_text()
    )


def write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_full_and_each_single_job_render_validate(tmp_path: Path) -> None:
    config = load_config()
    MODULE.validate(write(tmp_path, config))
    for job in config["jobs"]:
        selected = copy.deepcopy(config)
        selected["jobs"] = [copy.deepcopy(job)]
        MODULE.validate(write(tmp_path, selected))


def test_unknown_or_empty_job_set_fails_closed(tmp_path: Path) -> None:
    config = load_config()
    empty = copy.deepcopy(config)
    empty["jobs"] = []
    with pytest.raises(ValueError, match="Job set drift"):
        MODULE.validate(write(tmp_path, empty))
    unknown = copy.deepcopy(config)
    unknown["jobs"][0]["name"] = "unexpected-job"
    with pytest.raises(ValueError, match="Job set drift"):
        MODULE.validate(write(tmp_path, unknown))


def test_c4_launcher_derives_hash_bound_ordered_prefix() -> None:
    text = (ROOT / "scripts" / "run_state8_exact_diagnosis_midpoint_c4.sh").read_text()
    for fragment in (
        "parent_manifest_sha256",
        "'selection':'ordered_prefix'",
        "value['pairs'][:50]",
        'PAIRS=/tmp/state8_exact_diagnosis_midpoint_pairs_seed42_n50.json',
    ):
        assert fragment in text


def test_watcher_uses_job_selector_without_incompatible_status_filter() -> None:
    text = (ROOT / "scripts" / "watch_state8_exact_diagnosis_midpoint_eval.sh").read_text()
    assert 'amlt results download "$experiment" ":$job" \\' in text
    assert '":$job" --status pass' not in text
