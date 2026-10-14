from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_state8_diagnosis_first_matched_teacher_curriculum as matched  # noqa: E402


def test_matched_teacher_contract_is_schema_specific() -> None:
    assert matched.PROVENANCE_STAGE == "state8_diagnosis_first_teacher_provenance_gate_v1"
    assert matched.OUTPUT_SCHEMA == "diagnosis_evidence_label_v1"
    source = Path(matched.__file__).read_text(encoding="utf-8")
    for fragment in (
        "teacher_gate_sha256",
        "teacher_provenance_gate_sha256",
        '"diagnosis",',
        '"evidence",',
        '"label",',
        '"history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0]',
        '"raw_history_visible_to_decoder": False',
    ):
        assert fragment in source
