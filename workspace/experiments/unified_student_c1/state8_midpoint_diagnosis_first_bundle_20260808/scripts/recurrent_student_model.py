#!/usr/bin/env python3
"""Minimal closed-state wrapper for a recurrent generative student model.

The updater receives one prior state and one observed interaction.  The decoder
receives only an instruction, the continuous state, and the current query.  No
raw history or persistent KV cache is accepted by either interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
from torch import nn


STATE_TOKEN = "<STUDENT_STATE>"
HISTORY_MARKER = "<student_history>"


OBSERVED_INTERACTION_FIELDS = (
    "timestamp",
    "interaction_id",
    "question_id",
    "kc_ids",
    "kc_names",
    "question_text",
    "problem_text",
    "options",
    "student_response",
    "student_response_id",
    "student_answer",
    "correct_response",
    "correct_response_id",
    "correct_answer",
    "correctness",
    "is_correct",
    "first_response_correctness",
    "unsupported_first_attempt_success",
    "hint_used",
    "hint_count",
    "saw_answer",
    "answer_type",
    "question_type",
    "problem_part",
    "problem_set_id",
)


def serialize_observed_interaction(interaction: dict[str, Any]) -> str:
    """Serialize only information observable after one student interaction."""
    payload = {
        key: interaction[key]
        for key in OBSERVED_INTERACTION_FIELDS
        if key in interaction and interaction[key] is not None
    }
    if not payload:
        raise ValueError("Interaction contains none of the supported observed fields")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_direct_decode_text(prompt: str) -> str:
    """Extract the state-and-query suffix from an existing direct-state row."""
    if prompt.count(STATE_TOKEN) != 1:
        raise ValueError(f"Expected exactly one {STATE_TOKEN} in direct prompt")
    prefix, suffix = prompt.split(STATE_TOKEN, 1)
    history_start = prefix.find(HISTORY_MARKER)
    if history_start < 0:
        raise ValueError(f"Direct prompt lacks {HISTORY_MARKER}")
    instruction = prefix[:history_start]
    decode_text = f"{instruction}{STATE_TOKEN}{suffix}"
    assert_decode_text_is_closed(decode_text)
    return decode_text


def build_observation_matched_direct_text(
    prompt: str, interactions: Sequence[dict[str, Any]]
) -> str:
    """Replace raw history with exactly the observations used by recurrence.

    This view is used as the one-shot direct-state comparator.  It preserves
    the original instruction, state address, and query, while ensuring the
    direct and recurrent states receive the same interactions.
    """
    history_open = "<student_history>"
    history_close = "</student_history>"
    start = prompt.find(history_open)
    end = prompt.find(history_close)
    if start < 0 or end < 0 or end < start:
        raise ValueError("Prompt lacks a well-formed student-history segment")
    if not interactions:
        raise ValueError("Observation-matched direct view requires interactions")
    observations = "\n".join(
        f'<observed_interaction index="{index}">'
        f"{serialize_observed_interaction(interaction)}"
        "</observed_interaction>"
        for index, interaction in enumerate(interactions, start=1)
    )
    replacement = f"{history_open}\n{observations}\n{history_close}"
    matched = prompt[:start] + replacement + prompt[end + len(history_close) :]
    if matched.count(STATE_TOKEN) != 1:
        raise ValueError(f"Matched direct prompt must contain exactly one {STATE_TOKEN}")
    return matched


def truncate_direct_history(prompt: str, unroll: int) -> str:
    """Keep the last ``unroll`` natural-language trials in a direct prompt."""
    if unroll <= 0:
        raise ValueError("unroll must be positive")
    history_open = "<student_history>"
    history_close = "</student_history>"
    start = prompt.find(history_open)
    end = prompt.find(history_close)
    if start < 0 or end < 0 or end < start:
        raise ValueError("Prompt lacks a well-formed student-history segment")
    body_start = start + len(history_open)
    body = prompt[body_start:end].strip()
    matches = list(re.finditer(r"(?m)^Trial\s+\d+:\s*$", body))
    if len(matches) < unroll:
        raise ValueError(f"Natural history has {len(matches)} trials, requested {unroll}")
    blocks = []
    for index, match in enumerate(matches):
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        blocks.append(body[match.start() : block_end].strip())
    selected = blocks[-unroll:]
    selected = [
        re.sub(r"^Trial\s+\d+:", f"Trial {index}:", block, count=1)
        for index, block in enumerate(selected, start=1)
    ]
    replacement = f"{history_open}\n" + "\n\n".join(selected) + f"\n{history_close}"
    matched = prompt[:start] + replacement + prompt[end + len(history_close) :]
    if matched.count(STATE_TOKEN) != 1:
        raise ValueError(f"Truncated direct prompt must contain exactly one {STATE_TOKEN}")
    return matched


def build_update_text(interaction: dict[str, Any]) -> str:
    """Use one token address twice: injected previous state, then collector."""
    observed = serialize_observed_interaction(interaction)
    return (
        "<student_state_update>\n"
        f"{STATE_TOKEN}\n"
        f"<observed_interaction>{observed}</observed_interaction>\n"
        f"{STATE_TOKEN}\n"
        "</student_state_update>"
    )


def assert_decode_text_is_closed(text: str) -> None:
    if HISTORY_MARKER.lower() in text.lower():
        raise ValueError("Decoder text contains raw student history")
    if text.count(STATE_TOKEN) != 1:
        raise ValueError(f"Decoder text must contain exactly one {STATE_TOKEN}")


def locate_token_positions(input_ids: torch.Tensor, token_id: int, count: int) -> torch.Tensor:
    """Return [batch, count] exact positions, rejecting missing/extra carriers."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}")
    rows = []
    for batch_index in range(input_ids.shape[0]):
        positions = torch.nonzero(input_ids[batch_index].eq(token_id), as_tuple=False).flatten()
        if positions.numel() != count:
            raise ValueError(
                f"Batch row {batch_index} has {positions.numel()} state carriers, expected {count}"
            )
        rows.append(positions)
    return torch.stack(rows, dim=0)


def expand_state_token_alias(alias: str, count: int) -> str:
    """Repeat one tokenizer-native carrier without introducing separator tokens."""
    if count <= 0:
        raise ValueError("State token count must be positive")
    return str(alias) * int(count)


def inject_state_embedding(
    embeddings: torch.Tensor, positions: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    """Replace exactly one carrier per row while preserving differentiability."""
    if embeddings.ndim != 3 or state.ndim != 2 or positions.ndim != 1:
        raise ValueError("Expected embeddings [B,L,D], positions [B], and state [B,D]")
    if embeddings.shape[0] != state.shape[0] or embeddings.shape[0] != positions.shape[0]:
        raise ValueError("Batch dimensions do not match")
    if embeddings.shape[2] != state.shape[1]:
        raise ValueError("State and embedding hidden dimensions do not match")
    injected = embeddings.clone()
    batch = torch.arange(embeddings.shape[0], device=embeddings.device)
    injected[batch, positions.to(embeddings.device)] = state.to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    return injected


def build_closed_decode_attention_mask(
    attention_2d: torch.Tensor,
    state_positions: torch.Tensor,
    prompt_ends: torch.Tensor,
) -> torch.Tensor:
    """Causal late-fusion mask for state-only decoding.

    Query tokens cannot read the state carrier.  The final prompt token is the
    answer anchor whose logits predict the first supervised target token, so
    that row and all supervised answer rows can read both query and state.
    There is no raw-history segment in this interface.
    """
    if attention_2d.ndim != 2:
        raise ValueError("attention_2d must have shape [batch, sequence]")
    batch_size, seq_len = attention_2d.shape
    min_value = torch.finfo(torch.float32).min
    device = attention_2d.device
    causal_block = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(causal_block.view(1, 1, seq_len, seq_len), min_value)
    mask.masked_fill_(attention_2d.eq(0).view(batch_size, 1, 1, seq_len), min_value)
    if state_positions.ndim == 1:
        state_positions = state_positions.unsqueeze(1)
    if state_positions.ndim != 2 or state_positions.shape[0] != batch_size:
        raise ValueError("State positions must have shape [batch] or [batch, slots]")
    for batch_index in range(batch_size):
        positions = state_positions[batch_index]
        if not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError("State carrier positions must be strictly increasing")
        state_start = int(positions[0].item())
        state_end = int(positions[-1].item())
        prompt_end = int(prompt_ends[batch_index].item())
        valid_len = int(attention_2d[batch_index].sum().item())
        answer_anchor = prompt_end - 1
        if not (0 <= state_start <= state_end <= answer_anchor < valid_len):
            raise ValueError(
                f"Invalid closed-decode positions: state={state_start}:{state_end}, "
                f"prompt_end={prompt_end}, valid_len={valid_len}"
            )
        # Rows before the answer anchor remain query-only.  The answer anchor's
        # logits predict target[0], hence it must be the first row that can read
        # the state.  Masking through prompt_end would silently remove state
        # conditioning from the first answer token.
        mask[
            batch_index,
            0,
            state_end + 1 : answer_anchor,
            positions.to(device),
        ] = min_value
    return mask


def build_closed_update_attention_mask(
    attention_2d: torch.Tensor,
    input_state_positions: torch.Tensor,
    output_state_positions: torch.Tensor,
) -> torch.Tensor:
    """Causal, padding-safe mask for a recurrent state update.

    The observed interaction can read the injected previous state and the
    output carrier can read both.  No raw-history segment exists in this view.
    """
    if attention_2d.ndim != 2:
        raise ValueError("attention_2d must have shape [batch, sequence]")
    batch_size, seq_len = attention_2d.shape
    if input_state_positions.ndim == 1:
        input_state_positions = input_state_positions.unsqueeze(1)
    if output_state_positions.ndim == 1:
        output_state_positions = output_state_positions.unsqueeze(1)
    if (
        input_state_positions.ndim != 2
        or output_state_positions.ndim != 2
        or input_state_positions.shape != output_state_positions.shape
        or input_state_positions.shape[0] != batch_size
    ):
        raise ValueError("State-position tensors must have shape [batch, slots]")
    min_value = torch.finfo(torch.float32).min
    device = attention_2d.device
    causal_block = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(causal_block.view(1, 1, seq_len, seq_len), min_value)
    mask.masked_fill_(attention_2d.eq(0).view(batch_size, 1, 1, seq_len), min_value)
    for batch_index in range(batch_size):
        input_positions = input_state_positions[batch_index]
        output_positions = output_state_positions[batch_index]
        if not bool((input_positions[1:] > input_positions[:-1]).all()):
            raise ValueError("Input state positions must be strictly increasing")
        if not bool((output_positions[1:] > output_positions[:-1]).all()):
            raise ValueError("Output state positions must be strictly increasing")
        input_pos = int(input_positions[0].item())
        input_end = int(input_positions[-1].item())
        output_pos = int(output_positions[0].item())
        output_end = int(output_positions[-1].item())
        valid_len = int(attention_2d[batch_index].sum().item())
        if not (0 <= input_pos <= input_end < output_pos <= output_end < valid_len):
            raise ValueError(
                f"Invalid update positions: input={input_pos}:{input_end}, "
                f"output={output_pos}:{output_end}, "
                f"valid_len={valid_len}"
            )
    return mask


@dataclass
class RecurrentStepOutput:
    state: torch.Tensor
    model_output: Any


class ResidualLowRankProjection(nn.Module):
    """Identity-initialized low-rank state map with an explicit capacity bound."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        if rank <= 0 or rank > hidden_size:
            raise ValueError(f"rank must be in [1, {hidden_size}], got {rank}")
        self.rank = int(rank)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / hidden_size**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Keep the small recurrent map in FP32 for optimizer stability while
        # accepting BF16 states from the quantized backbone.  Relying on an
        # ambient autocast context made training and standalone evaluation use
        # different dtype behavior.
        parameter_dtype = self.down.weight.dtype
        residual = self.up(self.down(value.to(dtype=parameter_dtype)))
        return value + residual.to(dtype=value.dtype)


class RecurrentStudentModel(nn.Module):
    """Thin recurrent state adapter over a causal language model."""

    def __init__(
        self,
        backbone: nn.Module,
        state_token_id: int,
        hidden_size: int | None = None,
        trainable_projections: bool = False,
        state_projection_rank: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_token_id = int(state_token_id)
        if hidden_size is None:
            config = getattr(backbone, "config", None)
            hidden_size = int(
                getattr(config, "hidden_size", 0)
                or getattr(config, "n_embd", 0)
                or backbone.get_input_embeddings().weight.shape[1]
            )
        self.hidden_size = int(hidden_size)
        self.state_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        if trainable_projections:
            self.state_in_proj = ResidualLowRankProjection(
                self.hidden_size, state_projection_rank
            )
            self.state_out_proj = ResidualLowRankProjection(
                self.hidden_size, state_projection_rank
            )
        else:
            self.state_in_proj = nn.Identity()
            self.state_out_proj = nn.Identity()

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        row = self.backbone.get_input_embeddings().weight[self.state_token_id]
        if device is not None:
            row = row.to(device)
        return row.unsqueeze(0).expand(batch_size, -1).clone()

    def _final_decoder_norm(self) -> nn.Module | None:
        """Locate the production backbone's final norm for one-state capture.

        Qwen causal-LM outputs retain every layer when ``output_hidden_states``
        is enabled.  The recurrent updater only needs the final normalized
        representation at one carrier position, so a forward hook avoids that
        multi-layer activation retention.  Small test backbones without this
        conventional module fall back to their reported hidden states.
        """
        base = (
            self.backbone.get_base_model()
            if hasattr(self.backbone, "get_base_model")
            else self.backbone
        )
        decoder = getattr(base, "model", None)
        norm = getattr(decoder, "norm", None)
        return norm if isinstance(norm, nn.Module) else None

    def update_state(
        self,
        state_prev: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> RecurrentStepOutput:
        carriers = locate_token_positions(input_ids, self.state_token_id, count=2)
        input_position = carriers[:, 0]
        output_position = carriers[:, 1]
        embeddings = self.backbone.get_input_embeddings()(input_ids)
        # A sharded backbone commonly keeps token embeddings on cuda:0 and the
        # final decoder norm on the last CUDA device.  Keep the recurrent state
        # on the embedding device, where it is injected on the next step, and
        # cross the device boundary explicitly for the small output projection.
        state_device = embeddings.device
        state_prev = state_prev.to(device=state_device)
        injected_state = self.state_in_proj(self.state_norm(state_prev))
        embeddings = inject_state_embedding(embeddings, input_position, injected_state)
        # Re-entrant gradient checkpointing requires at least one differentiable
        # input.  Quantized frozen embeddings otherwise make the first recurrent
        # update silently skip LoRA gradients in the identity-projection arm.
        if torch.is_grad_enabled() and not embeddings.requires_grad:
            embeddings.requires_grad_(True)
        model_attention = attention_mask
        if attention_mask.ndim == 2:
            model_attention = build_closed_update_attention_mask(
                attention_mask, input_position, output_position
            )
        if (
            model_attention.ndim == 4
            and getattr(self.backbone.config, "_attn_implementation", "eager")
            == "sdpa"
        ):
            model_attention = model_attention.to(dtype=embeddings.dtype)
        final_hidden: list[torch.Tensor] = []
        norm = self._final_decoder_norm()
        capture = (
            norm.register_forward_hook(
                lambda _module, _inputs, output: final_hidden.append(output)
            )
            if norm is not None
            else None
        )
        try:
            outputs = self.backbone(
                inputs_embeds=embeddings,
                attention_mask=model_attention,
                output_hidden_states=norm is None,
                use_cache=False,
                return_dict=True,
            )
        finally:
            if capture is not None:
                capture.remove()
        if norm is not None:
            if len(final_hidden) != 1:
                raise ValueError(
                    f"Expected one final hidden-state capture, got {len(final_hidden)}"
                )
            hidden = final_hidden[0]
        else:
            hidden = outputs.hidden_states[-1]
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        collected = hidden[batch, output_position.to(hidden.device)]
        output_parameter = next(self.state_out_proj.parameters(), None)
        output_device = (
            output_parameter.device if output_parameter is not None else collected.device
        )
        state_next = self.state_out_proj(collected.to(device=output_device))
        state_next = state_next.to(device=state_device)
        return RecurrentStepOutput(state=state_next, model_output=outputs)

    def decode_from_state(
        self,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        prompt_ends: torch.Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Any:
        carrier = locate_token_positions(input_ids, self.state_token_id, count=1)[:, 0]
        embeddings = self.backbone.get_input_embeddings()(input_ids)
        state = state.to(device=embeddings.device)
        embeddings = inject_state_embedding(
            embeddings, carrier, self.state_in_proj(self.state_norm(state))
        )
        if torch.is_grad_enabled() and not embeddings.requires_grad:
            embeddings.requires_grad_(True)
        model_attention = attention_mask
        if prompt_ends is not None:
            model_attention = build_closed_decode_attention_mask(
                attention_mask, carrier, prompt_ends
            )
        if (
            model_attention.ndim == 4
            and getattr(self.backbone.config, "_attn_implementation", "eager")
            == "sdpa"
        ):
            model_attention = model_attention.to(dtype=embeddings.dtype)
        kwargs = {
            "inputs_embeds": embeddings,
            "attention_mask": model_attention,
            "output_hidden_states": output_hidden_states,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if labels is not None:
            kwargs["labels"] = labels
        return self.backbone(**kwargs)

    def rollout(
        self,
        initial_state: torch.Tensor,
        update_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        decode_input_ids: torch.Tensor,
        decode_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        decode_prompt_ends: torch.Tensor | None = None,
        detach_every: int | None = None,
    ) -> tuple[torch.Tensor, Any]:
        state = initial_state
        total_steps = len(update_batches)
        for step, (input_ids, attention_mask) in enumerate(update_batches, start=1):
            state = self.update_state(state, input_ids, attention_mask).state
            # Truncate only between unroll blocks. Detaching the final state
            # would prevent the decoder objective from training the last update.
            if detach_every and step % detach_every == 0 and step < total_steps:
                state = state.detach()
        decoded = self.decode_from_state(
            state,
            decode_input_ids,
            decode_attention_mask,
            labels=labels,
            prompt_ends=decode_prompt_ends,
        )
        return state, decoded
