from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from latent_recurrent_student_model import (  # noqa: E402
    LatentRecurrentStudentModel,
    LayerwiseLatentCodec,
)
from test_layerwise_recurrent_student_model import ToyLayerwiseLM  # noqa: E402


class LatentRecurrentStudentModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.backbone = ToyLayerwiseLM()
        self.wrapper = LatentRecurrentStudentModel(
            self.backbone, state_token_id=7, codec_rank=2
        )

    def test_codec_is_single_vector_and_identity_initialized(self) -> None:
        codec = LayerwiseLatentCodec(8, 3, 2)
        latent = torch.randn(2, 8)
        torch.testing.assert_close(codec.inject(latent, 0), codec.norm(latent))
        trajectory = torch.stack([latent, latent, latent], dim=1)
        torch.testing.assert_close(codec.collect(trajectory), codec.norm(latent))

    def test_update_and_decode_keep_single_vector_capacity(self) -> None:
        state = self.wrapper.initial_state(1)
        step = self.wrapper.update_state(
            state,
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        self.assertEqual(tuple(step.state.shape), (1, 8))
        self.assertEqual(tuple(step.trajectory.shape), (1, 3, 8))
        output = self.wrapper.decode_from_state(
            step.state,
            torch.tensor([[7, 4, 5, 6]]),
            torch.ones(1, 4, dtype=torch.long),
            labels=torch.tensor([[-100, -100, 5, 6]]),
            prompt_ends=torch.tensor([2]),
        )
        self.assertIsNotNone(output.loss)
        self.assertIs(self.backbone.last_use_cache, False)

    def test_gradient_reaches_codec_and_prior_state(self) -> None:
        prior = self.wrapper.initial_state(1).detach().requires_grad_()
        step = self.wrapper.update_state(
            prior,
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        output = self.wrapper.decode_from_state(
            step.state,
            torch.tensor([[7, 4, 5, 6]]),
            torch.ones(1, 4, dtype=torch.long),
            labels=torch.tensor([[-100, -100, 5, 6]]),
            prompt_ends=torch.tensor([2]),
        )
        output.loss.backward()
        self.assertGreater(float(prior.grad.abs().sum()), 0.0)
        gradients = [
            parameter.grad
            for parameter in self.wrapper.codec.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0 for g in gradients))

    def test_wrong_state_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Latent state"):
            self.wrapper.decode_from_state(
                torch.zeros(1, 3, 8),
                torch.tensor([[7, 4, 5]]),
                torch.ones(1, 3, dtype=torch.long),
                prompt_ends=torch.tensor([2]),
            )


if __name__ == "__main__":
    unittest.main()
