from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/run_state8_exact_diagnosis_pipeline.sh").read_text()


def test_pipeline_has_separate_curriculum_and_causal_consolidation() -> None:
    for fragment in (
        "run_state8_exact_diagnosis_curriculum.sh",
        "train_state8_diagnosis_first_causal_repair.py",
        "STATE8_RANDOM_NEUTRAL=1",
        "STATE8_GEOMETRY_WEIGHT=0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT=0.05",
        "STATE8_CAUSAL_REPEAT_FRACTION=0.50",
        "REQUESTED_WORLD_SIZE=1",
        "raw_history_visible_to_decoder",
        "diagnosis_field_order",
        "canonical_label_role",
    ):
        assert fragment in SOURCE
