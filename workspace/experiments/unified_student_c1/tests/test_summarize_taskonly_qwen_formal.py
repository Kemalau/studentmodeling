from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_summary_contract_is_full_headline_and_semantic() -> None:
    source = (ROOT / "scripts/summarize_taskonly_qwen_formal.py").read_text(encoding="utf-8")
    assert '"T1": ("predict_correctness", 10_913)' in source
    assert '"T2": ("predict_response", 3_693)' in source
    assert '"T3": ("diagnose_misconception", 1_494)' in source
    assert '"T4": ("predict_prospective_error", 4_600)' in source
    assert '"T5": ("predict_error_persistence", 976)' in source
    assert 'semantic.get("judge_backend") != "trapi"' in source
    assert '"joint_math_equivalence"' in source
    assert 'binary_metrics(records, "persistent")' in source
