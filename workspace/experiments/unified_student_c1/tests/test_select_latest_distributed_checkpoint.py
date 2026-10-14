from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_latest_distributed_checkpoint import select_latest  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(root: Path, round_index: int, *, complete: bool = True) -> Path:
    path = root / f"checkpoint-round-{round_index}"
    (path / "adapter").mkdir(parents=True)
    state = path / "recurrent_state.pt"
    adapter = path / "adapter/adapter_model.safetensors"
    state.write_bytes(f"state-{round_index}".encode())
    adapter.write_bytes(f"adapter-{round_index}".encode())
    if complete:
        (path / "checkpoint_completion.json").write_text(json.dumps({
            "status": "complete",
            "round": round_index,
            "sha256": {
                "adapter_model": digest(adapter),
                "recurrent_state": digest(state),
            },
        }))
    return state


def test_selects_latest_verified_checkpoint_and_honors_stop_round(tmp_path: Path) -> None:
    checkpoint(tmp_path, 3250)
    expected = checkpoint(tmp_path, 3500)
    checkpoint(tmp_path, 3750)
    assert select_latest(tmp_path, max_round=3600) == expected


def test_ignores_incomplete_and_hash_mismatched_checkpoints(tmp_path: Path) -> None:
    expected = checkpoint(tmp_path, 3000)
    checkpoint(tmp_path, 3250, complete=False)
    corrupted = checkpoint(tmp_path, 3500)
    corrupted.write_bytes(b"corrupt")
    assert select_latest(tmp_path) == expected
