from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_state8_diagnosis_first_teacher_curriculum.py").read_text()


def test_exact_curriculum_contract_is_explicit() -> None:
    for fragment in (
        "train_state8_taxonomy_teacher_curriculum",
        "apply_diagnosis_first_order",
        '"history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0]',
        '"canonical_label_role": "auxiliary_ordinary_weight"',
        '"semantic_judge_authoritative": True',
        '"exact_label_metric_authoritative": False',
        '"raw_history_visible_to_decoder": False',
        '"final_inference_route": "strict_state_only"',
    ):
        assert fragment in SOURCE
