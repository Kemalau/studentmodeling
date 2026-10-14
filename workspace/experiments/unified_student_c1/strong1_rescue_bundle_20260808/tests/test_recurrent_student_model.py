from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from recurrent_student_model import (  # noqa: E402
    RecurrentStudentModel,
    assert_decode_text_is_closed,
    build_closed_decode_attention_mask,
    build_closed_update_attention_mask,
    build_observation_matched_direct_text,
    build_update_text,
    expand_state_token_alias,
    extract_direct_decode_text,
    locate_token_positions,
    serialize_observed_interaction,
    truncate_direct_history,
)
from train_recurrent_student_diagnostic import (  # noqa: E402
    deterministic_full_history_state,
    state_statistics,
    validate_resume_checkpoint,
)


class ToyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.mix = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.last_use_cache = None

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        inputs_embeds=None,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
        labels=None,
    ):
        self.last_use_cache = use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        # Simple causal mixing makes the collector depend on the injected state.
        hidden = self.mix(inputs_embeds.cumsum(dim=1))
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), loss=loss)


class RecurrentStudentModelTest(unittest.TestCase):
    def test_state_statistics_preserves_layerwise_example_axis(self) -> None:
        states = [torch.zeros(1, 3, 2), torch.ones(1, 3, 2)]
        result = state_statistics(states)
        self.assertAlmostEqual(result["state_variance"], 0.25)
        self.assertEqual(result["mean_pairwise_cosine"], 0.0)

    def test_layerwise_validation_casts_direct_candidate_attention(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_diagnostic.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'direct_candidate["attention_mask"] = direct_candidate[',
            source,
        )
        self.assertIn(
            "wrapper.backbone.get_input_embeddings().weight.dtype",
            source,
        )

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.backbone = ToyCausalLM()
        self.wrapper = RecurrentStudentModel(self.backbone, state_token_id=7)

    def test_text_builders_exclude_raw_history_from_decoder(self) -> None:
        interaction = {
            "question_text": "2+2?",
            "student_response": "5",
            "correct_response": "4",
            "correctness": 0,
            "private_field": "must not serialize",
        }
        observed = serialize_observed_interaction(interaction)
        self.assertIn("student_response", observed)
        self.assertNotIn("private_field", observed)
        update = build_update_text(interaction)
        self.assertEqual(update.count("<STUDENT_STATE>"), 2)
        direct = (
            "instruction\n<student_history>secret history</student_history>\n"
            "<STUDENT_STATE>\n<query>future question</query>"
        )
        decoded = extract_direct_decode_text(direct)
        self.assertNotIn("secret history", decoded)
        self.assertNotIn("<student_history>", decoded)
        self.assertIn("future question", decoded)
        with self.assertRaisesRegex(ValueError, "raw student history"):
            assert_decode_text_is_closed(direct)
        matched = build_observation_matched_direct_text(direct, [interaction])
        self.assertIn("student_response", matched)
        self.assertNotIn("secret history", matched)
        self.assertEqual(matched.count("<STUDENT_STATE>"), 1)

        natural = (
            "instruction\n<student_history>\nTrial 1:\nold\n\n"
            "Trial 2:\nnew\n</student_history>\n<STUDENT_STATE>\nquery"
        )
        truncated = truncate_direct_history(natural, 1)
        self.assertNotIn("old", truncated)
        self.assertIn("Trial 1:\nnew", truncated)
        self.assertNotIn("Trial 2:", truncated)

    def test_low_rank_projection_is_identity_initialized_and_bounded(self) -> None:
        wrapper = RecurrentStudentModel(
            ToyCausalLM(hidden=8),
            state_token_id=7,
            trainable_projections=True,
            state_projection_rank=2,
        )
        value = torch.randn(3, 8)
        torch.testing.assert_close(wrapper.state_in_proj(value), value)
        parameter_count = sum(
            parameter.numel()
            for name, parameter in wrapper.named_parameters()
            if "state_in_proj" in name or "state_out_proj" in name
        )
        self.assertEqual(parameter_count, 2 * 2 * 8 * 2)

    def test_low_rank_projection_accepts_bf16_state_with_fp32_weights(self) -> None:
        wrapper = RecurrentStudentModel(
            ToyCausalLM(hidden=8),
            state_token_id=7,
            trainable_projections=True,
            state_projection_rank=2,
        )
        value = torch.randn(3, 8, dtype=torch.bfloat16)
        output = wrapper.state_in_proj(value)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, value)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_sharded_final_hidden_returns_state_to_embedding_device(self) -> None:
        class ShardedToyCausalLM(ToyCausalLM):
            def forward(self, *args, **kwargs):
                output = super().forward(*args, **kwargs)
                hidden = output.hidden_states[-1].to("cuda:1")
                logits = output.logits.to("cuda:1")
                return SimpleNamespace(
                    logits=logits,
                    hidden_states=(hidden,),
                    loss=output.loss,
                )

        backbone = ShardedToyCausalLM().to("cuda:0")
        wrapper = RecurrentStudentModel(
            backbone,
            state_token_id=7,
            trainable_projections=True,
            state_projection_rank=2,
        ).to("cuda:0")
        state = wrapper.initial_state(1, device=torch.device("cuda:0"))
        step = wrapper.update_state(
            state,
            torch.tensor([[7, 2, 3, 7]], device="cuda:0"),
            torch.ones(1, 4, dtype=torch.long, device="cuda:0"),
        )
        self.assertEqual(step.state.device, torch.device("cuda:0"))

    def test_deterministic_teacher_restores_training_mode(self) -> None:
        self.backbone.train()
        ids = torch.tensor([[1, 7, 3]])
        attention = torch.ones_like(ids)
        first = deterministic_full_history_state(self.backbone, ids, attention, 7)
        second = deterministic_full_history_state(self.backbone, ids, attention, 7)
        torch.testing.assert_close(first, second)
        self.assertTrue(self.backbone.training)

    def test_resume_checkpoint_requires_matching_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter/adapter_model.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"adapter")
            state = root / "recurrent_state.pt"
            state.write_bytes(b"state")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {
                "status": "complete",
                "step": 7,
                "sha256": {
                    "adapter_model": digest(adapter),
                    "recurrent_state": digest(state),
                },
            }
            (root / "checkpoint_completion.json").write_text(json.dumps(receipt))
            self.assertEqual(validate_resume_checkpoint(state)["step"], 7)
            state.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_resume_checkpoint(state)

    def test_closed_decode_mask_uses_state_only_at_answer(self) -> None:
        attention = torch.ones(1, 7, dtype=torch.long)
        mask = build_closed_decode_attention_mask(
            attention, torch.tensor([2]), torch.tensor([5])
        )
        self.assertLess(float(mask[0, 0, 3, 2]), -1e20)
        # Position 4 is the last prompt token; its logits predict target[0].
        self.assertEqual(float(mask[0, 0, 4, 2]), 0.0)
        self.assertEqual(float(mask[0, 0, 5, 2]), 0.0)

    def test_multislot_closed_masks_keep_all_slots_late_fused(self) -> None:
        attention = torch.ones(1, 10, dtype=torch.long)
        state = torch.tensor([[2, 3, 4]])
        decode = build_closed_decode_attention_mask(
            attention, state, torch.tensor([8])
        )
        for position in state[0].tolist():
            self.assertLess(float(decode[0, 0, 5, position]), -1e20)
            self.assertEqual(float(decode[0, 0, 7, position]), 0.0)
        update = build_closed_update_attention_mask(
            attention,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[7, 8, 9]]),
        )
        self.assertEqual(float(update[0, 0, 9, 1]), 0.0)
        self.assertEqual(expand_state_token_alias("V", 3), "VVV")

    def test_closed_update_mask_is_causal_and_padding_safe(self) -> None:
        attention = torch.tensor([[1, 1, 1, 1, 1, 0]])
        mask = build_closed_update_attention_mask(
            attention, torch.tensor([1]), torch.tensor([4])
        )
        self.assertLess(float(mask[0, 0, 2, 3]), -1e20)
        self.assertEqual(float(mask[0, 0, 4, 1]), 0.0)
        self.assertEqual(float(mask[0, 0, 4, 2]), 0.0)
        self.assertLess(float(mask[0, 0, 4, 5]), -1e20)

    def test_sdpa_closed_masks_are_cast_to_backbone_dtype(self) -> None:
        self.backbone.config._attn_implementation = "sdpa"
        observed = []
        original = self.backbone.forward

        def capture(*args, **kwargs):
            observed.append(kwargs["attention_mask"].dtype)
            return original(*args, **kwargs)

        self.backbone.forward = capture
        state = self.wrapper.initial_state(1)
        step = self.wrapper.update_state(
            state,
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        self.wrapper.decode_from_state(
            step.state,
            torch.tensor([[7, 4, 5]]),
            torch.ones(1, 3, dtype=torch.long),
            prompt_ends=torch.tensor([1]),
        )
        self.assertEqual(observed, [self.backbone.embed.weight.dtype] * 2)

    def test_exact_carrier_count_is_enforced(self) -> None:
        ids = torch.tensor([[1, 7, 2, 7], [7, 3, 7, 4]])
        positions = locate_token_positions(ids, 7, 2)
        torch.testing.assert_close(positions, torch.tensor([[1, 3], [0, 2]]))
        with self.assertRaisesRegex(ValueError, "expected 1"):
            locate_token_positions(ids, 7, 1)

    def test_update_decode_has_gradient_and_no_cache(self) -> None:
        previous = self.wrapper.initial_state(1).detach().requires_grad_()
        update_ids = torch.tensor([[7, 2, 3, 7, 4]])
        update_mask = torch.ones_like(update_ids)
        step = self.wrapper.update_state(previous, update_ids, update_mask)
        self.assertEqual(tuple(step.state.shape), (1, 8))
        decode_ids = torch.tensor([[1, 7, 5, 6]])
        decode_mask = torch.ones_like(decode_ids)
        labels = torch.tensor([[-100, -100, 5, 6]])
        output = self.wrapper.decode_from_state(
            step.state, decode_ids, decode_mask, labels, prompt_ends=torch.tensor([2])
        )
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertIs(self.backbone.last_use_cache, False)

    def test_identity_projection_with_frozen_embeddings_keeps_model_gradient(self) -> None:
        self.backbone.embed.weight.requires_grad_(False)
        state = self.wrapper.initial_state(1).detach()
        step = self.wrapper.update_state(
            state, torch.tensor([[7, 2, 3, 7]]), torch.ones(1, 4, dtype=torch.long)
        )
        output = self.wrapper.decode_from_state(
            step.state,
            torch.tensor([[7, 4, 5]]),
            torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[-100, 4, 5]]),
            prompt_ends=torch.tensor([1]),
        )
        output.loss.backward()
        self.assertIsNotNone(self.backbone.mix.weight.grad)
        self.assertGreater(float(self.backbone.mix.weight.grad.abs().sum()), 0.0)

    def test_rollout_detach_boundary(self) -> None:
        state = self.wrapper.initial_state(1).detach().requires_grad_()
        update = (torch.tensor([[7, 2, 7]]), torch.ones(1, 3, dtype=torch.long))
        final_state, _ = self.wrapper.rollout(
            state,
            [update, update],
            torch.tensor([[7, 3]]),
            torch.ones(1, 2, dtype=torch.long),
            decode_prompt_ends=torch.tensor([2]),
            detach_every=1,
        )
        self.assertTrue(final_state.requires_grad)  # second step remains differentiable
        final_state.sum().backward()
        self.assertIsNone(state.grad)


if __name__ == "__main__":
    unittest.main()
