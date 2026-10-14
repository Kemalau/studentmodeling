from __future__ import annotations

from pathlib import Path

import pytest

from scripts.select_state8_finalfit_topology import select


def test_auto_prefers_world4_only_when_four_user_slots_are_available(tmp_path: Path) -> None:
    assert select(tmp_path / "four.json", "auto", 4)["topology"] == "world4"
    assert select(tmp_path / "two.json", "auto", 3)["topology"] == "world2"


def test_selection_pins_global_batch_and_rejects_cross_topology_resume(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    result = select(output, "world2", 2)
    assert result["global_batch_size"] == 16
    assert result["optimizer_steps"] == 16241
    assert result["real_examples_exactly_once"] == 259845
    assert result["cross_topology_resume_allowed"] is False
    assert select(output, "world2", 2) == result
    with pytest.raises(ValueError, match="already pinned"):
        select(output, "world4", 4)
