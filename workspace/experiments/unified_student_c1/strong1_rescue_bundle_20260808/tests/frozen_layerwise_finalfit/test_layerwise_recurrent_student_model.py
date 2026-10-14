from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import copy
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from layerwise_recurrent_student_model import (  # noqa: E402
    LayerwiseRecurrentStudentModel,
    patch_address,
    save_on_cpu_preserve_sdpa_mask_stride,
    validate_state,
)


class ToyBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mix = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Every position reads the causal prefix in this small structural test.
        return self.mix(hidden.cumsum(dim=1))


class ToyLayerwiseLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, _attn_implementation="eager")
        self.embed = nn.Embedding(vocab, hidden)
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([ToyBlock(hidden) for _ in range(layers)])
        # Register the ModuleList through a normal Module attribute as well.
        self.blocks = self.model.layers
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.last_use_cache = None
        self.last_attention_dtype = None
        self.last_attention_shape = None
        self.last_attention_stride = None
        self.is_gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        labels=None,
        output_hidden_states=False,
    ):
        self.last_use_cache = use_cache
        self.last_attention_dtype = attention_mask.dtype
        self.last_attention_shape = tuple(attention_mask.shape)
        self.last_attention_stride = tuple(attention_mask.stride())
        hidden = self.embed(input_ids)
        for layer in self.blocks:
            hidden = layer(hidden)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class LayerwiseRecurrentStudentModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.backbone = ToyLayerwiseLM()
        self.wrapper = LayerwiseRecurrentStudentModel(
            self.backbone, state_token_id=7
        )

    def test_update_returns_one_address_trajectory(self) -> None:
        block_ids = {id(block) for block in self.backbone.blocks}
        registered_outside_backbone = {
            id(module)
            for name, module in self.wrapper.named_modules()
            if name.startswith("_decoder_layers")
        }
        self.assertFalse(block_ids & registered_outside_backbone)
        self.assertIs(
            self.wrapper.initial_state(1, device=torch.device("cpu")), None
        )
        step = self.wrapper.update_state(
            self.wrapper.initial_state(1, device=torch.device("cpu")),
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        self.assertEqual(tuple(step.state.shape), (1, 3, 8))
        self.assertIs(self.backbone.last_use_cache, False)

    def test_decode_accepts_formal_evaluator_arguments(self) -> None:
        state = self.wrapper.update_state(
            None,
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        ).state
        output = self.wrapper.decode_from_state(
            state,
            torch.tensor([[7, 4, 5, 6]]),
            torch.ones(1, 4, dtype=torch.long),
            prompt_ends=torch.tensor([2]),
            output_hidden_states=False,
        )
        self.assertEqual(tuple(output.logits.shape), (1, 4, 32))

    def test_sdpa_additive_attention_bias_matches_embedding_dtype(self) -> None:
        self.backbone.config._attn_implementation = "sdpa"
        state = self.wrapper.update_state(
            None,
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        ).state
        self.assertEqual(
            self.backbone.last_attention_dtype, self.backbone.embed.weight.dtype
        )
        self.wrapper.decode_from_state(
            state,
            torch.tensor([[7, 4, 5, 6]]),
            torch.ones(1, 4, dtype=torch.long),
            prompt_ends=torch.tensor([2]),
        )
        self.assertEqual(
            self.backbone.last_attention_dtype, self.backbone.embed.weight.dtype
        )

    def test_sdpa_bias_stride_is_aligned_without_changing_549_token_view(self) -> None:
        self.backbone.config._attn_implementation = "sdpa"
        original = torch.ones(2, 549, dtype=torch.float32)
        original[1, -7:] = 0
        aligned = self.wrapper._attention_for_backbone(original)
        self.assertEqual(tuple(aligned.shape), (2, 1, 549, 549))
        self.assertEqual(aligned.dtype, self.backbone.embed.weight.dtype)
        self.assertEqual(aligned.stride(1) % 4, 0)
        positions = torch.arange(549)
        expected = positions.view(549, 1) >= positions.view(1, 549)
        allowed = expected.view(1, 1, 549, 549) & original.ne(0).view(
            2, 1, 1, 549
        )
        expected = torch.full_like(aligned, torch.finfo(aligned.dtype).min)
        expected.masked_fill_(allowed, 0.0)
        torch.testing.assert_close(aligned, expected)

    def test_checkpointed_549_token_direct_capture_uses_aligned_mask(self) -> None:
        self.backbone.config._attn_implementation = "sdpa"
        wrapper = LayerwiseRecurrentStudentModel(
            self.backbone, state_token_id=7, operation_checkpointing=True
        )
        ids = torch.full((1, 549), 2, dtype=torch.long)
        ids[0, 1] = 7
        labels = torch.full_like(ids, -100)
        labels[0, -2:] = torch.tensor([2, 3])
        output = wrapper.capture_direct_state(ids, torch.ones_like(ids), labels)
        output.model_output.loss.backward()
        self.assertEqual(self.backbone.last_attention_shape, (1, 1, 549, 549))
        self.assertEqual(self.backbone.last_attention_stride[1] % 4, 0)

    def test_offloaded_checkpoint_saves_2d_mask_and_rebuilds_aligned_view(self) -> None:
        self.backbone.config._attn_implementation = "sdpa"
        wrapper = LayerwiseRecurrentStudentModel(
            self.backbone, state_token_id=7, operation_checkpointing=True
        )
        ids = torch.full((1, 549), 2, dtype=torch.long)
        ids[0, 1] = 7
        labels = torch.full_like(ids, -100)
        labels[0, -2:] = torch.tensor([2, 3])
        wrapper.offload_next_operation = True
        with patch(
            "layerwise_recurrent_student_model.checkpoint",
            wraps=torch_checkpoint,
        ) as checkpoint_call:
            output = wrapper.capture_direct_state(ids, torch.ones_like(ids), labels)
            output.model_output.loss.backward()
        checkpoint_mask = checkpoint_call.call_args.args[2]
        self.assertEqual(checkpoint_mask.ndim, 2)
        self.assertEqual(tuple(checkpoint_mask.shape), (1, 549))
        self.assertEqual(self.backbone.last_attention_shape, (1, 1, 549, 549))
        self.assertEqual(self.backbone.last_attention_stride[1], 549 * 552)

    def test_saved_779_token_mask_restores_padded_head_stride(self) -> None:
        length = 779
        storage = torch.zeros(1, 1, length, 780, dtype=torch.float32)
        mask = storage[..., :length]
        observed = []

        class SaveMask(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, attention):
                ctx.save_for_backward(attention)
                return value.square()

            @staticmethod
            def backward(ctx, gradient):
                (attention,) = ctx.saved_tensors
                observed.append(tuple(attention.stride()))
                return 2.0 * gradient, None

        value = torch.ones((), requires_grad=True)
        with save_on_cpu_preserve_sdpa_mask_stride():
            SaveMask.apply(value, mask).backward()
        self.assertEqual(observed[0][1] % 4, 0)
        self.assertEqual(observed[0][1], length * 780)

    def test_saved_head_expanded_653_token_mask_restores_zero_head_stride(self) -> None:
        length = 653
        storage = torch.zeros(1, 1, length, 656, dtype=torch.float32)
        mask = storage[..., :length].expand(1, 8, length, length)
        observed = []

        class SaveMask(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, attention):
                ctx.save_for_backward(attention)
                return value.square()

            @staticmethod
            def backward(ctx, gradient):
                (attention,) = ctx.saved_tensors
                observed.append((tuple(attention.stride()), attention[:, :1].clone()))
                return 2.0 * gradient, None

        value = torch.ones((), requires_grad=True)
        expected = mask[:, :1].clone()
        with save_on_cpu_preserve_sdpa_mask_stride():
            SaveMask.apply(value, mask).backward()
        self.assertEqual(observed[0][0][1], 0)
        self.assertEqual(observed[0][0][2], 656)
        torch.testing.assert_close(observed[0][1], expected)

    def test_direct_state_and_supervised_output_share_one_forward(self) -> None:
        result = self.wrapper.capture_direct_state(
            torch.tensor([[2, 7, 4, 5]]),
            torch.ones(1, 4, dtype=torch.long),
            labels=torch.tensor([[-100, -100, 4, 5]]),
        )
        self.assertEqual(tuple(result.state.shape), (1, 3, 8))
        self.assertIsNotNone(result.model_output.loss)
        result.model_output.loss.backward()
        self.assertGreater(float(self.backbone.head.weight.grad.abs().sum()), 0.0)

    def test_previous_trajectory_changes_next_state_and_decode(self) -> None:
        ids = torch.tensor([[7, 2, 3, 7]])
        mask = torch.ones_like(ids)
        first = self.wrapper.update_state(None, ids, mask).state
        perturbed = first.clone()
        perturbed[:, :, 0] += 4.0
        second_a = self.wrapper.update_state(first, ids, mask).state
        second_b = self.wrapper.update_state(perturbed, ids, mask).state
        self.assertGreater(float((second_a - second_b).abs().sum()), 0.0)

        decode_ids = torch.tensor([[7, 4, 5, 6]])
        decode_mask = torch.ones_like(decode_ids)
        out_a = self.wrapper.decode_from_state(
            first, decode_ids, decode_mask, prompt_ends=torch.tensor([2])
        )
        out_b = self.wrapper.decode_from_state(
            perturbed, decode_ids, decode_mask, prompt_ends=torch.tensor([2])
        )
        self.assertGreater(float((out_a.logits - out_b.logits).abs().sum()), 0.0)

    def test_gradient_reaches_previous_trajectory_without_cache(self) -> None:
        prior = torch.randn(1, 3, 8, requires_grad=True)
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
        self.assertIs(self.backbone.last_use_cache, False)

    def test_whole_operation_checkpoint_matches_state_logits_and_gradients(self) -> None:
        baseline_backbone = ToyLayerwiseLM()
        baseline = LayerwiseRecurrentStudentModel(
            baseline_backbone, state_token_id=7, operation_checkpointing=False
        )
        checkpoint_backbone = copy.deepcopy(baseline_backbone)
        checkpointed = LayerwiseRecurrentStudentModel(
            checkpoint_backbone, state_token_id=7, operation_checkpointing=True
        )
        ids = torch.tensor([[7, 2, 3, 7]])
        mask = torch.ones_like(ids)
        decode_ids = torch.tensor([[7, 4, 5, 6]])
        decode_mask = torch.ones_like(decode_ids)
        labels = torch.tensor([[-100, -100, 5, 6]])

        baseline_state = baseline.update_state(None, ids, mask).state
        baseline_output = baseline.decode_from_state(
            baseline_state,
            decode_ids,
            decode_mask,
            labels=labels,
            prompt_ends=torch.tensor([2]),
        )
        baseline_output.loss.backward()

        checkpoint_state = checkpointed.update_state(None, ids, mask).state
        checkpoint_output = checkpointed.decode_from_state(
            checkpoint_state,
            decode_ids,
            decode_mask,
            labels=labels,
            prompt_ends=torch.tensor([2]),
        )
        checkpoint_output.loss.backward()

        torch.testing.assert_close(checkpoint_state, baseline_state)
        torch.testing.assert_close(checkpoint_output.logits, baseline_output.logits)
        for baseline_parameter, checkpoint_parameter in zip(
            baseline.parameters(), checkpointed.parameters(), strict=True
        ):
            self.assertIsNotNone(baseline_parameter.grad)
            self.assertIsNotNone(checkpoint_parameter.grad)
            torch.testing.assert_close(
                checkpoint_parameter.grad, baseline_parameter.grad
            )

    def test_checkpointed_direct_capture_matches_output_and_gradient(self) -> None:
        baseline_backbone = ToyLayerwiseLM()
        baseline = LayerwiseRecurrentStudentModel(
            baseline_backbone, state_token_id=7, operation_checkpointing=False
        )
        checkpoint_backbone = copy.deepcopy(baseline_backbone)
        checkpointed = LayerwiseRecurrentStudentModel(
            checkpoint_backbone, state_token_id=7, operation_checkpointing=True
        )
        ids = torch.tensor([[2, 7, 4, 5]])
        mask = torch.ones_like(ids)
        labels = torch.tensor([[-100, -100, 4, 5]])
        expected = baseline.capture_direct_state(ids, mask, labels)
        expected.model_output.loss.backward()
        actual = checkpointed.capture_direct_state(ids, mask, labels)
        actual.model_output.loss.backward()
        torch.testing.assert_close(actual.state, expected.state)
        torch.testing.assert_close(actual.model_output.logits, expected.model_output.logits)
        for baseline_parameter, checkpoint_parameter in zip(
            baseline.parameters(), checkpointed.parameters(), strict=True
        ):
            self.assertIsNotNone(baseline_parameter.grad)
            self.assertIsNotNone(checkpoint_parameter.grad)
            torch.testing.assert_close(
                checkpoint_parameter.grad, baseline_parameter.grad
            )

    def test_one_shot_checkpoint_offload_preserves_gradient_and_is_consumed(self) -> None:
        baseline_backbone = ToyLayerwiseLM()
        baseline = LayerwiseRecurrentStudentModel(
            baseline_backbone, state_token_id=7, operation_checkpointing=True
        )
        offload_backbone = copy.deepcopy(baseline_backbone)
        offloaded = LayerwiseRecurrentStudentModel(
            offload_backbone, state_token_id=7, operation_checkpointing=True
        )
        ids = torch.tensor([[2, 7, 4, 5]])
        mask = torch.ones_like(ids)
        labels = torch.tensor([[-100, -100, 4, 5]])

        expected = baseline.capture_direct_state(ids, mask, labels)
        expected.model_output.loss.backward()
        offloaded.offload_next_operation = True
        with patch(
            "layerwise_recurrent_student_model.save_on_cpu_preserve_sdpa_mask_stride",
            wraps=save_on_cpu_preserve_sdpa_mask_stride,
        ) as offload_context:
            actual = offloaded.capture_direct_state(ids, mask, labels)
            actual.model_output.loss.backward()
        self.assertGreaterEqual(offload_context.call_count, 1)
        self.assertFalse(offloaded.offload_next_operation)
        torch.testing.assert_close(actual.state, expected.state)
        torch.testing.assert_close(actual.model_output.logits, expected.model_output.logits)
        for expected_parameter, actual_parameter in zip(
            baseline.parameters(), offloaded.parameters(), strict=True
        ):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)

    def test_transition_adapter_is_identity_initialized_and_trainable(self) -> None:
        wrapper = LayerwiseRecurrentStudentModel(
            ToyLayerwiseLM(), state_token_id=7, transition_rank=2
        )
        value = torch.randn(1, 8)
        torch.testing.assert_close(wrapper.transition_in[0](value), value)
        torch.testing.assert_close(wrapper.transition_out[0](value), value)
        prior = torch.randn(1, 3, 8, requires_grad=True)
        step = wrapper.update_state(
            prior,
            torch.tensor([[7, 2, 3, 7]]),
            torch.ones(1, 4, dtype=torch.long),
        )
        step.state.square().mean().backward()
        gradients = [
            parameter.grad
            for name, parameter in wrapper.named_parameters()
            if "transition_" in name and parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None and float(gradient.abs().sum()) > 0
                for gradient in gradients
            )
        )

    def test_shape_and_patch_checks_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_state(torch.zeros(1, 2, 8), 1, 3, 8)
        hidden = torch.zeros(2, 4, 8)
        values = torch.stack([torch.ones(8), torch.full((8,), 2.0)])
        patched = patch_address(hidden, torch.tensor([1, 3]), values)
        torch.testing.assert_close(patched[0, 1], values[0])
        torch.testing.assert_close(patched[1, 3], values[1])

    def test_gradient_checkpointing_is_rejected_in_train_mode(self) -> None:
        self.backbone.is_gradient_checkpointing = True
        with self.assertRaisesRegex(RuntimeError, "gradient checkpointing"):
            self.wrapper.update_state(
                None,
                torch.tensor([[7, 2, 3, 7]]),
                torch.ones(1, 4, dtype=torch.long),
            )
        with torch.inference_mode():
            step = self.wrapper.update_state(
                None,
                torch.tensor([[7, 2, 3, 7]]),
                torch.ones(1, 4, dtype=torch.long),
            )
        self.assertEqual(tuple(step.state.shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
