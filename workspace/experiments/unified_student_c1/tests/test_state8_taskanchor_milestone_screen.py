from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_taskanchor_milestone_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_taskanchor_milestone_screen_config_is_hash_bound_and_read_only() -> None:
    validate(ROOT / "amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml")


def test_taskanchor_milestone_selector_dump_requires_explicit_subset_mode(
    tmp_path: Path,
) -> None:
    source = ROOT / "amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml"
    config = yaml.safe_load(source.read_text())
    config["jobs"] = [config["jobs"][0]]
    rendered = tmp_path / "selected.yaml"
    rendered.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="Job set drift"):
        validate(rendered)
    validate(rendered, allow_job_subset=True)


def test_taskanchor_milestone_subset_mode_rejects_unknown_or_duplicate_jobs(
    tmp_path: Path,
) -> None:
    source = ROOT / "amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml"
    config = yaml.safe_load(source.read_text())
    selected = config["jobs"][0]

    config["jobs"] = [selected, selected]
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="Job set drift"):
        validate(duplicate, allow_job_subset=True)

    config["jobs"] = [{**selected, "name": "state8-taskanchor-milestone-999"}]
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="Job set drift"):
        validate(unknown, allow_job_subset=True)


def test_taskanchor_milestone_runner_uses_fixed_semantic_protocol() -> None:
    text = (ROOT / "scripts/run_state8_taskanchor_milestone_screen.sh").read_text()
    for fragment in (
        "--max-samples-per-group 25",
        "--tasks diagnose_misconception",
        "--sources eedi_mining malrule",
        "--num-pairs 50 --seed 42",
        "--num-pairs 20 --seed 42",
        "raw_history_visible_to_decoder",
        "semantic_judge_authoritative",
        "exact_match_authoritative",
    ):
        assert fragment in text
