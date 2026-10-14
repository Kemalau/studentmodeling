from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_t3_eedi_specialists_paired import compare  # noqa: E402


def rows(values: list[bool]):
    return {
        f"eedi-{index:04d}": {"llm_judge_correct": value}
        for index, value in enumerate(values)
    }


def test_paired_counts_and_difference() -> None:
    challenger = rows([True] * 450 + [False] * 444)
    reference = rows([True] * 400 + [False] * 494)
    value = compare(challenger, reference, bootstrap_samples=100, seed=42)
    assert value["n"] == 894
    assert value["discordant"]["challenger_only_correct"] == 50
    assert value["discordant"]["reference_only_correct"] == 0
    assert value["challenger_minus_reference"] == 50 / 894
