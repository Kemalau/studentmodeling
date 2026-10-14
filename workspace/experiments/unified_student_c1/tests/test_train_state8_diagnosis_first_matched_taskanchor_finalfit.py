from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_matched_taskanchor_finalfit_composes_both_contracts() -> None:
    source = (ROOT / "scripts/train_state8_diagnosis_first_matched_taskanchor_finalfit.py").read_text()
    for fragment in (
        "train_state8_diagnosis_first_task_anchored_causal_repair",
        "state8_diagnosis_first_teacher_full_provenance_gate_v1",
        "diagnosis_evidence_label_v1",
        '"diagnosis", "evidence", "label"',
        "teacher_gate_support = 1494",
        "strict_state_only",
        "diagnosis_anchor_schedule",
        "exact_match_authoritative",
    ):
        assert fragment in source

