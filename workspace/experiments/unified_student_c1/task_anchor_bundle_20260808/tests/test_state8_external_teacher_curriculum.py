from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "scripts/train_recurrent_student_m4_state8_external_teacher_curriculum.py"
).read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "scripts"))

from train_recurrent_student_m4_state8_external_teacher_curriculum import (  # noqa: E402
    ensure_dataset_audits,
)


def test_external_teacher_curriculum_is_qualified_frozen_and_state_only() -> None:
    assert "external_teacher_spec()" in SOURCE
    assert "load_external_teacher(backbone, teacher_path)" in SOURCE
    assert "capture_from_external_teacher" in SOURCE
    assert 'namespace.teacher_gate_required = True' in SOURCE
    assert 'namespace.teacher_trainable = False' in SOURCE
    assert 'namespace.final_inference_route = "strict_state_only"' in SOURCE
    assert "distills verified answer logits, not an unverified carrier trajectory" in SOURCE
    assert '"none_unverified_carrier_trajectory_excluded"' in SOURCE
    assert 'namespace.state_token_count != 8' in SOURCE
    assert "[0.0, 0.25, 0.5, 0.75, 1.0]" in SOURCE
    assert "state8_natural_view_sha256" in SOURCE
    assert "state8_clean_prompt_teacher_provenance_gate_v2" in SOURCE
    assert "teacher_evaluation_input_sha256" in SOURCE
    assert "teacher_clean_prompt_sha256" in SOURCE
    assert "legacy_answer_adjacent_contracts" in SOURCE
    assert "namespace.diagnostic_segment_rounds = None" in SOURCE


def test_external_teacher_curriculum_rewrites_train_and_valid() -> None:
    assert 'valid_path = cli_path("--valid-jsonl")' in SOURCE
    assert "{train_path, valid_path}" in SOURCE
    assert "apply_state8_natural_curriculum_view(rows)" in SOURCE
    assert 'audits.append({"path": str(resolved_path), **audit})' in SOURCE
    assert '"dataset_audits": audits' in SOURCE


def test_complete_checkpoint_finalizer_reconstructs_missing_dataset_audits(
    tmp_path: Path,
) -> None:
    train = (tmp_path / "train.jsonl").resolve()
    valid = (tmp_path / "valid.jsonl").resolve()
    train.write_text("train\n", encoding="utf-8")
    valid.write_text("valid\n", encoding="utf-8")
    calls: list[Path] = []
    audits: list[dict[str, int | str]] = []

    def read_rows(path: Path):
        calls.append(path)
        return [path.name]

    def apply_view(rows):
        assert len(rows) == 1
        return {"changed_t3_rows": 1}

    ensure_dataset_audits(
        audits,
        (train, valid),
        read_rows=read_rows,
        apply_view=apply_view,
    )
    assert calls == [train, valid]
    assert [row["path"] for row in audits] == [str(train), str(valid)]
    assert all(
        row["audit_provenance"]
        == "reconstructed_from_immutable_input_after_exact_resume"
        for row in audits
    )

    ensure_dataset_audits(
        audits,
        (train, valid),
        read_rows=read_rows,
        apply_view=apply_view,
    )
    assert calls == [train, valid]
