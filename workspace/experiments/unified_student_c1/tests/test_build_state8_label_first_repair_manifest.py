from __future__ import annotations

from scripts.build_state8_label_first_repair_manifest import STAGE


def test_stage_is_versioned() -> None:
    assert STAGE == "state8_label_first_repair_index_manifest_v1"
