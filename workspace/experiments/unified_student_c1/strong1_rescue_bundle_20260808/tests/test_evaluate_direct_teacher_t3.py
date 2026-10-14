from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_direct_teacher_t3 import direct_teacher_states_batched  # noqa: E402


def test_direct_teacher_state_requires_layerwise_wrapper() -> None:
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

    with pytest.raises(ValueError, match="requires layerwise"):
        direct_teacher_states_batched(Wrapper(), object(), [], "<state>", 32, 1)


def test_wrapper_is_explicitly_non_gate_and_patches_only_state_builder() -> None:
    source = (ROOT / "scripts/evaluate_direct_teacher_t3.py").read_text()
    assert "evaluator.recurrent_states_batched = direct_teacher_states_batched" in source
    assert '"eligible_for_final_gate": False' in source
    assert '"raw_history_visible_to_state_encoder": True' in source
    assert '"raw_history_visible_to_decoder": False' in source
