from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_state8_single_resume_checkpoint import audit  # noqa: E402


def fixture(root: Path, step: int = 2000) -> tuple[Path, Path]:
    checkpoint = root / "recurrent_state.pt"
    payload = {
        "step": step,
        "config": {
            "gradient_accumulation_steps": 4,
            "planned_optimizer_steps": 1250,
            "full_micro_steps": 5000,
            "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
            "causal_objective_mode": "specificity_triplet",
            "causal_repeat_fraction": 0.05,
            "state_token_count": 8,
            "final_inference_route": "strict_state_only",
        },
        "optimizer": {},
        "scheduler": {},
        "python_random_state": (),
        "torch_random_state": torch.tensor([1], dtype=torch.uint8),
        "cuda_random_state_all": [torch.tensor([2], dtype=torch.uint8)],
        "real_order": [0],
        "causal_schedule": {0: 0},
        "history": [],
        "transition_in": torch.zeros(1),
        "transition_out": torch.zeros(1),
    }
    torch.save(payload, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    completion = root / "checkpoint_completion.json"
    completion.write_text(
        json.dumps(
            {"status": "complete", "step": step, "sha256": {"recurrent_state": sha}}
        )
    )
    return checkpoint, completion


def test_accepts_complete_optimizer_boundary(tmp_path: Path) -> None:
    checkpoint, completion = fixture(tmp_path)
    result = audit(checkpoint, completion)
    assert result["optimizer_step"] == 500
    assert result["remaining_micro_steps"] == 3000


def test_rejects_non_boundary_checkpoint(tmp_path: Path) -> None:
    checkpoint, completion = fixture(tmp_path, step=2250)
    with pytest.raises(ValueError, match="GA=4 boundary"):
        audit(checkpoint, completion)


def test_rejects_missing_optimizer_state(tmp_path: Path) -> None:
    checkpoint, completion = fixture(tmp_path)
    payload = torch.load(checkpoint, weights_only=False)
    payload["optimizer"] = None
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="optimizer"):
        audit(checkpoint, completion)
