import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage_t1_t2_resume_checkpoint import REQUIRED, stage


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_checkpoint(root: Path, *, step: int = 20_000) -> Path:
    root.mkdir()
    for name in REQUIRED:
        (root / name).write_bytes((name + "\n").encode())
    state = {
        "global_step": step,
        "max_steps": 40_692,
        "save_steps": 10_000,
        "stateful_callbacks": {"TrainerControl": {"args": {"should_save": True}}},
    }
    (root / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    return root


def test_stage_changes_only_trainer_state(tmp_path: Path) -> None:
    source = make_checkpoint(tmp_path / "checkpoint-20000")
    before = {name: sha(source / name) for name in REQUIRED}
    destination = tmp_path / "staged"
    receipt = stage(source, destination, save_steps=1000, max_steps=40_692)
    state = json.loads((destination / "trainer_state.json").read_text())

    assert receipt["old_save_steps"] == 10_000
    assert receipt["new_save_steps"] == 1000
    assert receipt["only_trainer_state_changed"] is True
    assert state["global_step"] == 20_000
    assert state["save_steps"] == 1000
    assert state["stateful_callbacks"]["TrainerControl"]["args"]["should_save"] is False
    for name in REQUIRED:
        if name != "trainer_state.json":
            assert sha(destination / name) == before[name]


def test_stage_rejects_horizon_drift(tmp_path: Path) -> None:
    source = make_checkpoint(tmp_path / "checkpoint-20000")
    with pytest.raises(ValueError, match="Incompatible Trainer horizon"):
        stage(source, tmp_path / "staged", save_steps=1000, max_steps=50_000)
