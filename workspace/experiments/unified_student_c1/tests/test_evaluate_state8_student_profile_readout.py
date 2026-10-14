from __future__ import annotations

from unittest.mock import patch

import torch

from scripts import evaluate_state8_student_profile_readout as profile_eval


class DummyWrapper:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        yield self.weight


def test_profile_rollout_is_inference_only() -> None:
    grad_enabled = []

    def rollout(*_args, **_kwargs):
        grad_enabled.append(torch.is_grad_enabled())
        return torch.ones(1, 1, 1)

    row = {
        "update_texts": ["update"],
        "decode_text": "decode",
        "target": "<weakness>a</weakness><strength>b</strength>",
        "profile": {"weaknesses": ["a"], "strengths": ["b"]},
        "observed_profile": {"weaknesses": ["a"], "strengths": ["b"]},
        "future_weaknesses": ["a"],
    }
    with patch.object(profile_eval, "rollout_state", side_effect=rollout), patch.object(
        profile_eval,
        "generate_one",
        return_value=(
            "<weakness>a</weakness><strength>b</strength>"
            "<profile>p</profile><evidence_confidence>high</evidence_confidence>"
        ),
    ):
        records = profile_eval.evaluate_profiles(
            DummyWrapper(), object(), [row, row], "alias", 32, 8
        )

    assert grad_enabled == [False, False]
    assert len(records) == 2
    assert all(record["schema_valid"] for record in records)


def test_profile_evaluation_releases_inference_memory_per_student() -> None:
    row = {
        "update_texts": ["update"],
        "decode_text": "decode",
        "target": "<weakness>a</weakness><strength>b</strength>",
        "profile": {"weaknesses": ["a"], "strengths": ["b"]},
        "observed_profile": {"weaknesses": ["a"], "strengths": ["b"]},
        "future_weaknesses": ["a"],
    }
    generated = (
        "<weakness>a</weakness><strength>b</strength>"
        "<profile>p</profile><evidence_confidence>high</evidence_confidence>"
    )
    with patch.object(profile_eval, "rollout_state", return_value=torch.ones(1, 1, 1)), patch.object(
        profile_eval, "generate_one", return_value=generated
    ), patch.object(profile_eval, "release_inference_memory") as release:
        profile_eval.evaluate_profiles(DummyWrapper(), object(), [row, row], "alias", 32, 8)

    assert release.call_count == 2
