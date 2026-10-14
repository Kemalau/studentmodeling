from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from scripts.select_latest_valid_single_checkpoint import select, validate_candidate


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_checkpoint(root: Path, step: int, *, history_rows: int | None = None) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    state = checkpoint / "recurrent_state.pt"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(f"adapter-{step}".encode())
    torch.save(
        {
            "step": step,
            "history": [{}] * (step if history_rows is None else history_rows),
            "config": {
                "distributed": False,
                "distributed_world_size": 1,
                "planned_optimizer_steps": 1250,
                "gradient_accumulation_steps": 4,
                "full_micro_steps": 5000,
            },
        },
        state,
    )
    (checkpoint / "checkpoint_completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "step": step,
                "sha256": {
                    "adapter_model": digest(adapter),
                    "recurrent_state": digest(state),
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_selects_newest_hash_valid_single_checkpoint(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 250)
    make_checkpoint(tmp_path, 500)
    selected = select(
        tmp_path,
        max_step=5000,
        save_steps=250,
        planned_optimizer_steps=1250,
        gradient_accumulation_steps=4,
    )
    assert selected == tmp_path / "checkpoint-500/recurrent_state.pt"


def test_rejects_incomplete_history_and_falls_back(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 500)
    make_checkpoint(tmp_path, 1000, history_rows=999)
    selected = select(
        tmp_path,
        max_step=5000,
        save_steps=500,
        planned_optimizer_steps=1250,
        gradient_accumulation_steps=4,
    )
    assert selected == tmp_path / "checkpoint-500/recurrent_state.pt"


def test_rejects_checkpoint_between_accumulation_boundaries(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path, 250)
    valid, reason = validate_candidate(
        checkpoint,
        max_step=5000,
        save_steps=250,
        planned_optimizer_steps=1250,
        gradient_accumulation_steps=4,
    )
    assert valid is False
    assert "gradient-accumulation boundary" in reason


def test_stateonly_launcher_enables_single_device_remote_resume() -> None:
    source = (ROOT / "scripts/run_state8_matched_checkpoint2000_rescue.sh").read_text()
    assert "Scope automatic single-device recovery" in source
    assert "select_latest_valid_single_checkpoint.py" in source
    assert '--run-dir "$RUN_DIR" --max-step 1000' in source
    assert "--planned-optimizer-steps 250" in source
    assert 'test "$STATE8_REPAIR_SAVE_STEPS" = "$STATE8_REPAIR_RECOVERY_SAVE_STEPS"' in source
