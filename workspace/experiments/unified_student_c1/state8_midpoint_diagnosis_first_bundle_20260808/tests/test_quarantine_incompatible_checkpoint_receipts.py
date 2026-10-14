from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.quarantine_incompatible_checkpoint_receipts import quarantine


def checkpoint(root: Path, round_index: int, planned: int | None) -> Path:
    path = root / f"checkpoint-round-{round_index}"
    path.mkdir(parents=True)
    torch.save({"config": {"planned_optimizer_steps": planned}}, path / "recurrent_state.pt")
    (path / "checkpoint_completion.json").write_text(json.dumps({"status": "complete"}))
    return path


def test_quarantines_only_proven_recipe_mismatch(tmp_path: Path) -> None:
    stale = checkpoint(tmp_path, 500, 800)
    valid = checkpoint(tmp_path, 125, 1250)
    ambiguous = checkpoint(tmp_path, 250, None)
    result = quarantine(tmp_path, 1250)
    assert [row["checkpoint"] for row in result["quarantined"]] == ["checkpoint-round-500"]
    assert not (stale / "checkpoint_completion.json").exists()
    assert (stale / "checkpoint_completion.incompatible-planned-800.json").is_file()
    assert (stale / "recurrent_state.pt").is_file()
    assert (valid / "checkpoint_completion.json").is_file()
    assert (ambiguous / "checkpoint_completion.json").is_file()


def test_missing_run_dir_is_a_noop(tmp_path: Path) -> None:
    result = quarantine(tmp_path / "missing", 1250)
    assert result["inspected"] == 0 and result["quarantined"] == []
