from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_state8_matched_checkpoint2000_screen import materialize
from validate_state8_matched_checkpoint_milestone_screen_amlt import validate


def test_checkpoint3000_template_validates() -> None:
    validate(
        ROOT / "amlt/feeds_state8_matched_checkpoint3000_screen_20260810.yaml",
        step=3000,
        allow_template=True,
    )


def test_checkpoint4000_template_validates() -> None:
    validate(
        ROOT / "amlt/feeds_state8_matched_checkpoint4000_screen_20260810.yaml",
        step=4000,
        allow_template=True,
    )


def test_materializer_binds_requested_step(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"complete","step":3000,"sha256":{"adapter_model":"' + 'a'*64 + '","recurrent_state":"' + 'b'*64 + '"}}')
    output = tmp_path / "amlt" / "out.yaml"
    output.parent.mkdir()
    result = materialize(
        ROOT / "amlt/feeds_state8_matched_checkpoint3000_screen_20260810.yaml",
        receipt,
        output,
        3000,
    )
    assert result["adapter_model"] == "a" * 64
    validate(output, step=3000)


def test_watcher_is_receipt_bound_resumable_and_semantically_judged() -> None:
    text = (ROOT / "scripts/watch_state8_matched_checkpoint_milestone_screen.sh").read_text()
    for fragment in (
        "checkpoint_completion.json", "--step \"$step\"", "passed.$job",
        "amlt resume", "--judge-backend trapi", "summarize_state8_diagnosis_c4.py",
        "diagnostic_only", "strict_state_only",
    ):
        assert fragment in text
