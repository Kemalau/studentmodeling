from __future__ import annotations

from pathlib import Path

import pytest

from scripts.select_state8_finalfit_topology import select


def test_auto_prefers_world4_only_when_four_user_slots_are_available(tmp_path: Path) -> None:
    assert select(tmp_path / "four.json", "auto", 4)["topology"] == "world4"
    assert select(tmp_path / "two.json", "auto", 3)["topology"] == "world2"


def test_auto_respects_cluster_slots_even_when_user_has_capacity(tmp_path: Path) -> None:
    result = select(tmp_path / "cluster-three.json", "auto", 32, 3)
    assert result["topology"] == "world2"
    assert result["available_cluster_slots_at_selection"] == 3
    assert result["schedulable_slots_at_selection"] == 3
    assert select(tmp_path / "cluster-four.json", "auto", 32, 4)["topology"] == "world4"


def test_selection_pins_global_batch_and_rejects_cross_topology_resume(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    result = select(output, "world2", 2, 3)
    assert result["global_batch_size"] == 16
    assert result["optimizer_steps"] == 16241
    assert result["real_examples_exactly_once"] == 259845
    assert result["cross_topology_resume_allowed"] is False
    assert select(output, "world2", 2, 3) == result
    with pytest.raises(ValueError, match="already pinned"):
        select(output, "world4", 4, 4)


def test_selection_rejects_less_than_two_auditable_slots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two auditable A100 slots"):
        select(tmp_path / "zero.json", "auto", 0, 31)
    with pytest.raises(ValueError, match="at least two auditable A100 slots"):
        select(tmp_path / "cluster-one.json", "auto", 32, 1)
    with pytest.raises(ValueError, match="world4 requires 4"):
        select(tmp_path / "explicit-world4.json", "world4", 2, 2)


def test_matched_profile_selects_matched_configs(tmp_path: Path) -> None:
    world4 = select(
        tmp_path / "matched4.json", "auto", 4, 5, "diagnosis_first_matched"
    )
    assert world4["profile"] == "diagnosis_first_matched"
    assert "diagnosis_first_matched_final_fit_world4" in world4["config"]
    world2 = select(
        tmp_path / "matched2.json", "auto", 2, 2, "diagnosis_first_matched"
    )
    assert "diagnosis_first_matched_final_fit_world2" in world2["config"]
    assert world2["distributed_rounds"] == 129923
