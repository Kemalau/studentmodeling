import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from add_state8_teacher_markers import MARKER, transform  # noqa: E402


def test_marker_is_constant_eight_tokens_after_history() -> None:
    prompt = "<student_history>x</student_history>\n<query>q</query>"
    value = transform(prompt)
    assert value.count("<|vision_pad|>") == 8
    assert f"</student_history>\n{MARKER}" in value


def test_marker_transform_rejects_existing_carrier() -> None:
    import pytest

    with pytest.raises(ValueError, match="already contains"):
        transform("<student_history>x</student_history><STUDENT_STATE>")
