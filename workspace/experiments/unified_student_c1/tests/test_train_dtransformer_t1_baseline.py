from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_dtransformer_t1_baseline.py"
SPEC = importlib.util.spec_from_file_location("train_dtransformer_t1_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(history_length: int = 6, source: str = "source_a") -> dict:
    return {
        "task": "predict_correctness",
        "dataset": source,
        "target_label": "correct",
        "history": [
            {
                "kc_ids": [f"kc-{index}"],
                "question_id": f"q-{index}",
                "correctness": index % 2,
            }
            for index in range(history_length)
        ],
        "metadata": {"mix_source": source, "skill_id": "target-kc", "question_id": "target-q"},
    }


def test_sample_is_causal_sequence_with_target_last() -> None:
    sample = MODULE.sample_from_row(row(), max_seq_len=13)
    assert len(sample.responses) == 7
    assert sample.responses[-1] == 1
    assert sample.skills[-1] == "source_a::target-kc"
    assert sample.questions[-1] == "source_a::target-q"


def test_source_prefix_prevents_identifier_collisions() -> None:
    first = MODULE.sample_from_row(row(source="source_a"), max_seq_len=13)
    second = MODULE.sample_from_row(row(source="source_b"), max_seq_len=13)
    mapping = MODULE.vocabulary([first, second], "skills")
    assert mapping["source_a::target-kc"] != mapping["source_b::target-kc"]


def test_eval_unknowns_use_zero_and_padding_uses_minus_one() -> None:
    train = MODULE.sample_from_row(row(source="train"), max_seq_len=13)
    unseen = MODULE.sample_from_row(row(history_length=1, source="unseen"), max_seq_len=13)
    dataset = MODULE.SequenceDataset(
        [unseen],
        MODULE.vocabulary([train], "skills"),
        MODULE.vocabulary([train], "questions"),
        max_seq_len=13,
    )
    item = dataset[0]
    assert item["skills"][0].item() == 0
    assert item["skills"][2].item() == -1
    assert item["target_position"].item() == 1


def test_contrastive_sampler_never_mixes_short_and_long_rows() -> None:
    samples = [
        MODULE.sample_from_row(row(history_length=length), max_seq_len=13)
        for length in [1, 2, 4, 5, 6, 12]
    ]
    sampler = MODULE.ContrastiveBatchSampler(samples, batch_size=2, seed=42)
    for batch in sampler:
        eligibility = {len(samples[index].responses) >= 5 for index in batch}
        assert len(eligibility) == 1


class FakeModel:
    dropout_rate = 0.2

    def predict(self, skills, responses, questions):
        # Make leakage observable: the final logit would equal the target
        # response if the adapter failed to replace it with a constant zero.
        logits = responses.float()
        z = torch.nn.functional.one_hot(skills.clamp_min(0) % 2, num_classes=2).float()
        reg = logits.new_zeros(())
        return logits, z, z, reg, (None, None)


def test_target_only_loss_never_reads_target_response() -> None:
    model = FakeModel()
    skills = torch.tensor([[1, 2, 3], [2, 3, 4]])
    questions = torch.tensor([[1, 2, 3], [2, 3, 4]])
    positions = torch.tensor([2, 2])
    labels = torch.tensor([1.0, 0.0])
    first = torch.tensor([[1, 0, 1], [0, 1, 0]])
    second = first.clone()
    second[:, 2] = 1 - second[:, 2]
    loss_first, _, _ = MODULE.target_only_contrastive_loss(
        model, skills, first, questions, positions, labels, lambda_cl=0.0
    )
    loss_second, _, _ = MODULE.target_only_contrastive_loss(
        model, skills, second, questions, positions, labels, lambda_cl=0.0
    )
    assert torch.equal(loss_first, loss_second)
