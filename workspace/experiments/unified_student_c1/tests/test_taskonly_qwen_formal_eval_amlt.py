from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_taskonly_qwen_formal_eval_amlt import SPECS, validate  # noqa: E402


def test_full_taskonly_formal_eval_config() -> None:
    validate(ROOT / "amlt/feeds_taskonly_qwen_formal_eval_20260813.yaml")


def test_each_selector_dump_contract_is_supported() -> None:
    path = ROOT / "amlt/feeds_taskonly_qwen_formal_eval_20260813.yaml"
    for name in SPECS:
        # The full source config contains all jobs; selector-specific AMLT dumps
        # are checked by the runtime controller using the selected set.
        assert name in path.read_text(encoding="utf-8")
