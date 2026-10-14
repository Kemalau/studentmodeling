from types import SimpleNamespace
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.state8_natural_curriculum_view import (
    apply_state8_natural_curriculum_view,
    natural_prompt_text,
)
from train_recurrent_student_m4 import (
    curriculum_progress,
    dual_path_curriculum_weights,
    history_dropout_at_progress,
    state_path_gradient_audit_ready,
)


def test_natural_schema_preserves_label_and_adds_diagnosis_evidence():
    row = {
        "task": "diagnose_misconception",
        "source": "malrule",
        "target_label": "fractions.denominator_comparison_error",
        "target": "<misconception>fractions.denominator_comparison_error</misconception>",
        "decode_text": "Return exactly one <misconception> tag.\n<STUDENT_STATE>\n<query>x\nValid outputs:\n<misconception>x</misconception>\n</query>\nAnswer:",
        "update_texts": ["one"],
        "direct_text": "Return exactly one <misconception> tag.\n<student_history>h</student_history>\n<STUDENT_STATE>\n<query>x\nValid outputs:\n<misconception>x</misconception>\n</query>\nAnswer:",
    }
    audit = apply_state8_natural_curriculum_view([row])
    assert audit["changed_t3_rows"] == 1
    assert "<label>fractions.denominator_comparison_error</label>" in row["target"]
    assert "<diagnosis>" in row["target"]
    assert "larger denominator" in row["target"]
    assert "fractions.denominator_comparison_error</diagnosis>" not in row["target"]
    assert "<evidence>" in row["target"]
    assert "Valid outputs:" not in row["decode_text"]
    assert "Valid outputs:" not in row["direct_text"]


def test_history_dropout_schedule_has_five_requested_phases():
    args = SimpleNamespace(history_dropout_curriculum=True, dual_path_curriculum=True)
    assert [history_dropout_at_progress(args, x) for x in (0, .2, .4, .6, .8, 1)] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.0,
    ]
    assert curriculum_progress(12000, 13000, 12000) == 0.0
    assert curriculum_progress(12999, 13000, 12000) == 1.0
    assert curriculum_progress(12000, 259845, 12000, 13000) == 0.0
    assert curriculum_progress(12199, 259845, 12000, 13000) < 0.2
    assert curriculum_progress(12200, 259845, 12000, 13000) >= 0.2
    assert curriculum_progress(12999, 259845, 12000, 13000) == 1.0
    assert [dual_path_curriculum_weights(args, x) for x in (0, .2, .4, .6, .8)] == [
        (1.0, 0.0, 0.0),
        (0.75, 0.25, 1.0),
        (0.5, 0.5, 1.0),
        (0.25, 0.75, 1.0),
        (0.0, 1.0, 0.0),
    ]


def test_gradient_audit_waits_for_state_path_but_remains_strict_when_active():
    args = SimpleNamespace(
        behavior_kd_weight=0.0,
        online_kd_weight=0.5,
        consistency_weight=0.0,
    )
    teacher_only = {
        "recurrent_state_present": True,
        "effective_generation_weight": 0.0,
        "online_kd_phase_weight": 0.0,
        "causal_pair_id": None,
        "causal_effective_weight": 0.0,
    }
    assert state_path_gradient_audit_ready(teacher_only, args) is False

    mixed = dict(
        teacher_only,
        effective_generation_weight=0.25,
        online_kd_phase_weight=1.0,
    )
    assert state_path_gradient_audit_ready(mixed, args) is True

    causal = dict(
        teacher_only,
        recurrent_state_present=False,
        causal_pair_id="pair-1",
        causal_effective_weight=2.0,
    )
    assert state_path_gradient_audit_ready(causal, args) is True


def test_answer_adjacent_legacy_output_contract_is_rewritten():
    prompt = (
        "Return exactly <label>, <diagnosis>, and <evidence> tags.\n"
        "<query>q</query>\n"
        "Output format: <misconception>systematic misconception</misconception>\n"
        "Answer:"
    )
    rewritten = natural_prompt_text(prompt)
    assert "Output format: <misconception>" not in rewritten
    assert rewritten.count("<label>") >= 2
    assert "<diagnosis>natural-language diagnosis</diagnosis>" in rewritten
    assert "<evidence>interaction-grounded evidence</evidence>" in rewritten


def test_training_wrapper_rewrites_train_and_validation():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/train_recurrent_student_m4_state8_natural_curriculum.py"
    ).read_text(encoding="utf-8")
    assert 'valid_path = cli_path("--valid-jsonl")' in source
    assert "{train_path, valid_path}" in source
    assert '"dataset_audits": audits' in source
