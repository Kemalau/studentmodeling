from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_state8_taxonomy_teacher_curriculum import (  # noqa: E402
    schema_weighted_loss_factory,
)


class TinyTokenizer:
    mapping = {"<label>": [10, 11], "</label>": [12, 13]}

    def __call__(self, value, add_special_tokens=False):
        return {"input_ids": self.mapping[value]}


def test_curriculum_weights_only_the_canonical_label_value() -> None:
    loss = schema_weighted_loss_factory(TinyTokenizer(), 4.0)
    labels = torch.tensor([[-100, 10, 11, 99, 12, 13, 42]])
    logits = torch.zeros(1, 7, 128, requires_grad=True)
    logits.data[0, 2, 99] = -4.0
    weighted = loss(logits, labels)
    plain = torch.nn.functional.cross_entropy(
        logits[:, :-1][labels[:, 1:].ne(-100)],
        labels[:, 1:][labels[:, 1:].ne(-100)],
    )
    assert float(weighted.detach()) > float(plain.detach())


def test_curriculum_contract_is_semantic_first_and_state_only() -> None:
    source = (ROOT / "scripts/train_state8_taxonomy_teacher_curriculum.py").read_text()
    assert "apply_state8_taxonomy_contract" in source
    assert '"semantic_judge_authoritative": True' in source
    assert '"exact_label_metric_authoritative": False' in source
    assert '"final_inference_route": "strict_state_only"' in source
    assert '"raw_history_visible_to_decoder": False' in source
