#!/usr/bin/env python3
"""Verify value/loss/gradient equivalence of aligned SDPA masks on Qwen3."""

from __future__ import annotations

import copy
import json

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel


def verify(tokens: int = 549, *, checkpoint_offload: bool = False) -> dict[str, object]:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=1024,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    native_model = Qwen3ForCausalLM(config).eval()
    aligned_model = copy.deepcopy(native_model).eval()
    input_ids = torch.randint(0, 64, (1, tokens))
    input_ids[input_ids == 7] = 8
    input_ids[0, 1] = 7
    attention = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : max(0, tokens - 9)] = -100

    native = native_model(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        use_cache=False,
    )
    wrapper = LayerwiseRecurrentStudentModel(
        aligned_model,
        state_token_id=7,
        operation_checkpointing=checkpoint_offload,
    )
    aligned_attention = wrapper._attention_for_backbone(attention)
    if checkpoint_offload:
        wrapper.offload_next_operation = True
        captured = wrapper.capture_direct_state(input_ids, attention, labels)
        aligned = captured.model_output
    else:
        aligned = aligned_model(
            input_ids=input_ids,
            attention_mask=aligned_attention,
            labels=labels,
            use_cache=False,
        )
    torch.testing.assert_close(aligned.logits, native.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(aligned.loss, native.loss, rtol=1e-6, atol=1e-7)
    native.loss.backward()
    aligned.loss.backward()
    max_gradient_abs_diff = 0.0
    for (native_name, native_parameter), (aligned_name, aligned_parameter) in zip(
        native_model.named_parameters(), aligned_model.named_parameters(), strict=True
    ):
        if native_name != aligned_name:
            raise AssertionError((native_name, aligned_name))
        if native_parameter.grad is None or aligned_parameter.grad is None:
            raise AssertionError(f"missing gradient: {native_name}")
        max_gradient_abs_diff = max(
            max_gradient_abs_diff,
            float((native_parameter.grad - aligned_parameter.grad).abs().max()),
        )
        torch.testing.assert_close(
            native_parameter.grad,
            aligned_parameter.grad,
            rtol=2e-5,
            atol=2e-7,
        )
    if aligned_attention.stride(1) % 4:
        raise AssertionError("aligned mask still violates the SDPA head-stride contract")
    return {
        "status": "passed",
        "tokens": tokens,
        "checkpoint_offload": checkpoint_offload,
        "mask_shape": list(aligned_attention.shape),
        "mask_stride": list(aligned_attention.stride()),
        "loss_native": float(native.loss.detach()),
        "loss_aligned": float(aligned.loss.detach()),
        "max_gradient_abs_diff": max_gradient_abs_diff,
    }


if __name__ == "__main__":
    print(json.dumps({
        "plain_549": verify(),
        "checkpoint_offload_779": verify(779, checkpoint_offload=True),
    }, indent=2, sort_keys=True))
