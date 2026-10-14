from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_matched_checkpoint1200_screen_amlt import validate  # noqa: E402


CONFIG = ROOT / "amlt" / "feeds_state8_matched_checkpoint1200_screen_20260810.yaml"


def load() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "screen.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_full_screen_validates() -> None:
    validate(CONFIG, ROOT)


@pytest.mark.parametrize("index", [0, 1])
def test_targeted_failed_shard_recovery_validates(tmp_path: Path, index: int) -> None:
    value = load()
    value["jobs"] = [value["jobs"][index]]
    validate(write(tmp_path, value), ROOT)


def test_unknown_or_duplicate_jobs_fail_closed(tmp_path: Path) -> None:
    value = load()
    unknown = copy.deepcopy(value)
    unknown["jobs"][0]["name"] = "unknown"
    with pytest.raises(ValueError, match="Job set drift"):
        validate(write(tmp_path, unknown), ROOT)
    duplicate = copy.deepcopy(value)
    duplicate["jobs"] = [duplicate["jobs"][0], duplicate["jobs"][0]]
    with pytest.raises(ValueError, match="Job set drift"):
        validate(write(tmp_path, duplicate), ROOT)

