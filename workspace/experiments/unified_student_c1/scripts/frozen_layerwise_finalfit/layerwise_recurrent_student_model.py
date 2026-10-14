#!/usr/bin/env python3
"""Closed recurrent state carried by one address across decoder layers.

The scalar recurrent wrapper stores the final hidden state and feeds it back as
an input embedding.  This wrapper keeps the representation in its native
coordinate system: after block ``l`` it stores the output collector at block
``l`` and, on the next update/decode, patches that value at the input state
address after the same block.  Downstream block ``l + 1`` therefore consumes a
same-space state.  Only one sequence address is used and no history or KV cache
is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from recurrent_student_model import (
    ResidualLowRankProjection,
    build_closed_decode_attention_mask,
    build_closed_update_attention_mask,
    locate_token_positions,
)


def decoder_layers(backbone: nn.Module):
    candidates = (
        lambda model: model.model.layers,
        lambda model: model.model.model.layers,
        lambda model: model.base_model.model.model.layers,
        lambda model: model.get_base_model().model.layers,
        lambda model: model.get_base_model().model.model.layers,
    )
    for getter in candidates:
        try:
            layers = getter(backbone)
        except (AttributeError, TypeError):
            continue
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers")


def patch_address(
    hidden: torch.Tensor, positions: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    if hidden.ndim != 3 or positions.ndim != 1 or values.ndim != 2:
        raise ValueError("Expected hidden [B,L,D], positions [B], values [B,D]")
    if hidden.shape[0] != positions.shape[0] or hidden.shape[0] != values.shape[0]:
        raise ValueError("Layerwise state batch mismatch")
    if hidden.shape[-1] != values.shape[-1]:
        raise ValueError("Layerwise state hidden-size mismatch")
    patched = hidden.clone()
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    patched[batch, positions.to(hidden.device)] = values.to(
        device=hidden.device, dtype=hidden.dtype
    )
    return patched


def validate_state(
    state: torch.Tensor | None,
    batch_size: int,
    layer_count: int,
    hidden_size: int,
) -> None:
    if state is None:
        return
    expected = (batch_size, layer_count, hidden_size)
    if tuple(state.shape) != expected:
        raise ValueError(
            f"Layerwise state must have shape {expected}, got {tuple(state.shape)}"
        )


def _is_aligned_sdpa_mask(tensor: torch.Tensor) -> bool:
    """Identify a padded additive SDPA bias, including head-expanded views.

    Qwen expands the ``[B, 1, L, L]`` bias to ``[B, H, L, L]`` inside
    attention.  That expanded tensor has head stride zero.  Treating it as an
    ordinary saved tensor makes ``Tensor.cpu()`` materialize a contiguous
    ``H x L x L`` copy and changes the head stride to ``L**2``.  The CUDA SDPA
    backward kernel rejects that layout whenever ``L`` is not divisible by
    four, so both the one-head carrier and its zero-stride expanded view must
    be recognized here.
    """

    return (
        tensor.layout == torch.strided
        and tensor.ndim == 4
        and tensor.is_floating_point()
        and tensor.shape[-2] == tensor.shape[-1]
        and tensor.stride(-1) == 1
        and tensor.stride(-2) >= tensor.shape[-1]
        and tensor.stride(-2) % 4 == 0
        and (
            tensor.stride(1) == 0
            or (
                tensor.shape[1] == 1
                and tensor.stride(1) % 4 == 0
                and tensor.stride(1)
                != tensor.shape[-2] * tensor.shape[-1]
            )
        )
    )


@contextmanager
def save_on_cpu_preserve_sdpa_mask_stride():
    """Offload saved tensors while preserving the aligned SDPA-mask layout.

    ``torch.autograd.graph.save_on_cpu`` copies a non-contiguous CUDA view to a
    contiguous CPU tensor.  For the padded-key attention view below, that
    silently changes the head stride from ``L * ceil4(L)`` back to ``L**2``.
    The memory-efficient SDPA backward kernel then rejects any long sequence
    whose length is not divisible by four.  Preserve size/stride only for this
    uniquely shaped mask; all other saved tensors use the ordinary compact CPU
    copy so activation-offload memory remains unchanged.
    """

    def pack(tensor: torch.Tensor):
        device = tensor.device
        if _is_aligned_sdpa_mask(tensor):
            if tensor.shape[1] > 1 and tensor.stride(1) == 0:
                # Preserve one physical head with the original padded row
                # stride.  Re-expanding it during unpack restores the exact
                # zero head stride without storing H duplicate masks on CPU.
                base_size = (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])
                base_stride = (
                    tensor.stride(0),
                    tensor.stride(0),
                    tensor.stride(2),
                    tensor.stride(3),
                )
                cpu = torch.empty_strided(
                    base_size, base_stride, dtype=tensor.dtype, device="cpu"
                )
                cpu.copy_(tensor[:, :1])
                return (
                    "expanded_aligned_sdpa_mask",
                    device,
                    cpu,
                    tensor.size(),
                    tensor.stride(),
                )
            cpu = torch.empty_strided(
                tensor.size(), tensor.stride(), dtype=tensor.dtype, device="cpu"
            )
            cpu.copy_(tensor)
            return "aligned_sdpa_mask", device, cpu, tensor.size(), tensor.stride()
        return "plain", device, tensor.cpu(), None, None

    def unpack(payload):
        kind, device, cpu, size, stride = payload
        if kind == "expanded_aligned_sdpa_mask":
            base = torch.empty_strided(
                cpu.size(), cpu.stride(), dtype=cpu.dtype, device=device
            )
            base.copy_(cpu)
            restored = base.expand(size)
            if restored.stride(1) != 0 or not _is_aligned_sdpa_mask(restored):
                raise RuntimeError(
                    "Saved head-expanded SDPA mask lost its aligned layout"
                )
            return restored
        if kind == "aligned_sdpa_mask":
            restored = torch.empty_strided(
                size, stride, dtype=cpu.dtype, device=device
            )
            restored.copy_(cpu)
            if not _is_aligned_sdpa_mask(restored):
                raise RuntimeError("Saved SDPA mask lost its aligned head stride")
            return restored
        return cpu.to(device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


@dataclass
class LayerwiseRecurrentStepOutput:
    state: torch.Tensor
    model_output: Any


@dataclass
class LayerwiseDirectStateOutput:
    state: torch.Tensor
    model_output: Any


class LayerwiseRecurrentStudentModel(nn.Module):
    """One-address residual-trajectory recurrent wrapper."""

    def __init__(
        self,
        backbone: nn.Module,
        state_token_id: int,
        transition_rank: int = 0,
        operation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_token_id = int(state_token_id)
        # A plain list keeps non-owning references. Assigning the backbone's
        # ModuleList here would register every block a second time under the
        # wrapper and make state-dict/parameter ownership ambiguous.
        self._decoder_layers = list(decoder_layers(backbone))
        self.layer_count = len(self._decoder_layers)
        config = getattr(backbone, "config", None)
        self.hidden_size = int(
            getattr(config, "hidden_size", 0)
            or getattr(config, "n_embd", 0)
            or backbone.get_input_embeddings().weight.shape[1]
        )
        if transition_rank < 0 or transition_rank > self.hidden_size:
            raise ValueError(
                f"transition_rank must be in [0, {self.hidden_size}]"
            )
        self.transition_rank = int(transition_rank)
        self.operation_checkpointing = bool(operation_checkpointing)
        self.offload_next_operation = False
        projection = (
            lambda: ResidualLowRankProjection(
                self.hidden_size, self.transition_rank
            )
            if self.transition_rank
            else nn.Identity()
        )
        # These maps belong only to the recurrent update.  Direct state capture
        # and state-only decoding stay in the original per-layer coordinates,
        # preserving the passed zero-training bridge at initialization.
        self.transition_in = nn.ModuleList(
            [projection() for _ in range(self.layer_count)]
        )
        self.transition_out = nn.ModuleList(
            [projection() for _ in range(self.layer_count)]
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> None:
        """The empty history uses the pretrained carrier trajectory naturally."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # ``device`` is intentionally accepted for interface compatibility with
        # the scalar recurrent wrapper.  There is no materialized empty state to
        # move: the first update follows the pretrained carrier trajectory.
        del device
        return None

    def _assert_hook_safe_gradient_mode(self) -> None:
        """Reject checkpoint recomputation that would bypass removed hooks."""
        if torch.is_grad_enabled() and bool(
            getattr(self.backbone, "is_gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "Layerwise recurrent hooks require gradient checkpointing to be "
                "disabled; checkpoint recomputation after temporary hook removal "
                "is not graph-equivalent"
            )

    @staticmethod
    def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def _attention_for_backbone(self, attention: torch.Tensor) -> torch.Tensor:
        """Match the SDPA additive-bias dtype and kernel stride contract.

        Transformers materializes a contiguous boolean ``[B, 1, L, L]``
        causal mask from a 2-D mask.  PyTorch's memory-efficient SDPA kernel
        requires the per-head bias stride to be a multiple of four, so that
        layout fails whenever ``L`` is not divisible by four (the observed
        549-token row had ``stride(1) == 549 ** 2``).  Materialize the same
        allowed-token mask here with a key-storage dimension padded to four.
        The returned view retains shape ``[B, 1, L, L]``; token positions,
        mask values, logits, and the supervised objective are unchanged.
        """
        implementation = getattr(
            getattr(self.backbone, "config", None),
            "_attn_implementation",
            "eager",
        )
        if implementation == "sdpa" and attention.ndim == 2:
            batch_size, seq_len = attention.shape
            padded_keys = (int(seq_len) + 3) // 4 * 4
            dtype = self.backbone.get_input_embeddings().weight.dtype
            minimum = torch.finfo(dtype).min
            storage = torch.full(
                (batch_size, 1, seq_len, padded_keys),
                minimum,
                dtype=dtype,
                device=attention.device,
            )
            positions = torch.arange(seq_len, device=attention.device)
            causal = positions.view(seq_len, 1) >= positions.view(1, seq_len)
            allowed = causal.view(1, 1, seq_len, seq_len) & attention.ne(0).view(
                batch_size, 1, 1, seq_len
            )
            # Pass the final additive bias to SDPA.  A boolean mask is
            # value-equivalent in forward, but the memory-efficient kernel
            # converts it to a new contiguous additive bias during backward,
            # losing this padded head stride.  Constructing the additive bias
            # here avoids that hidden boolean-to-bias materialization.
            storage[..., :seq_len].masked_fill_(allowed, 0.0)
            aligned = storage[..., :seq_len]
            if aligned.stride(1) % 4:
                raise RuntimeError("Could not align SDPA causal-mask head stride")
            return aligned
        if attention.ndim == 4 and implementation == "sdpa":
            dtype = self.backbone.get_input_embeddings().weight.dtype
            attention = attention.to(dtype=dtype)
            if attention.stride(1) % 4:
                padded_keys = (int(attention.shape[-1]) + 3) // 4 * 4
                storage = attention.new_empty((*attention.shape[:-1], padded_keys))
                storage[..., : attention.shape[-1]].copy_(attention)
                attention = storage[..., : attention.shape[-1]]
                if attention.stride(1) % 4:
                    raise RuntimeError(
                        "Could not align SDPA additive-bias head stride"
                    )
            return attention
        return attention

    def capture_direct_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> LayerwiseDirectStateOutput:
        """Capture the native residual trajectory at the direct state address.

        The returned model output comes from the same forward pass, so a
        supervised direct-replay loss and a detached layerwise teacher state do
        not require two 32B forwards.
        """
        self._assert_hook_safe_gradient_mode()
        carrier = locate_token_positions(
            input_ids, self.state_token_id, count=1
        )[:, 0]
        kwargs = {
            "input_ids": input_ids,
            # Keep the compact 2-D mask as the checkpoint input.  Reentrant
            # checkpointing saves its tensor arguments outside the operation's
            # saved-tensor hooks; passing the padded 4-D view here lets that
            # save compact the view back to an L-by-L stride.  Rebuild the
            # value-equivalent aligned view inside ``operation`` on both the
            # no-grad forward and the grad-enabled recomputation instead.
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if labels is not None:
            kwargs["labels"] = labels
        checkpointed = self.operation_checkpointing and torch.is_grad_enabled()
        operation_output: Any = None

        offload = bool(self.offload_next_operation)
        self.offload_next_operation = False

        def operation(
            ids: torch.Tensor, mask: torch.Tensor, _gradient_anchor: torch.Tensor | None = None
        ):
            nonlocal operation_output
            captured: list[torch.Tensor] = []

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                batch = torch.arange(hidden.shape[0], device=hidden.device)
                state_device = self.backbone.get_input_embeddings().weight.device
                captured.append(
                    hidden[batch, carrier.to(hidden.device)].to(state_device)
                )

            handles = [
                layer.register_forward_hook(hook) for layer in self._decoder_layers
            ]
            local_kwargs = dict(kwargs)
            local_kwargs["input_ids"] = ids
            local_kwargs["attention_mask"] = self._attention_for_backbone(mask)
            if checkpointed:
                # The memory-safe path computes CE from returned logits outside
                # the checkpoint.  Avoid building the backbone's duplicate CE.
                local_kwargs.pop("labels", None)
            # Reentrant checkpoint executes its first pass under no_grad and
            # recomputes under grad.  Apply saved-tensor CPU offload only to the
            # recomputation graph; doing so around non-reentrant checkpoint
            # hooks would hide saved tensors from the checkpoint frame.
            offload_context = (
                save_on_cpu_preserve_sdpa_mask_stride()
                if offload and torch.is_grad_enabled()
                else nullcontext()
            )
            try:
                with offload_context:
                    output = self.backbone(**local_kwargs)
                if not checkpointed:
                    operation_output = output
            finally:
                for handle in handles:
                    handle.remove()
            if len(captured) != self.layer_count:
                raise ValueError(
                    f"Expected {self.layer_count} direct states, got {len(captured)}"
                )
            return torch.stack(captured, dim=1), output.logits

        if checkpointed:
            if offload:
                gradient_anchor = torch.empty(
                    0, device=input_ids.device, requires_grad=True
                )
                state, logits = checkpoint(
                    operation,
                    input_ids,
                    kwargs["attention_mask"],
                    gradient_anchor,
                    use_reentrant=True,
                    preserve_rng_state=True,
                )
            else:
                state, logits = checkpoint(
                    operation,
                    input_ids,
                    kwargs["attention_mask"],
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            output = SimpleNamespace(logits=logits, loss=loss)
        else:
            state, logits = operation(input_ids, kwargs["attention_mask"])
            # Preserve the original full output for callers outside the
            # memory-safe training path without a second model forward.
            output = operation_output
        return LayerwiseDirectStateOutput(
            state=state, model_output=output
        )

    def update_state(
        self,
        state_prev: torch.Tensor | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> LayerwiseRecurrentStepOutput:
        self._assert_hook_safe_gradient_mode()
        carriers = locate_token_positions(
            input_ids, self.state_token_id, count=2
        )
        input_position = carriers[:, 0]
        output_position = carriers[:, 1]
        validate_state(
            state_prev,
            input_ids.shape[0],
            self.layer_count,
            self.hidden_size,
        )
        model_attention = attention_mask
        if attention_mask.ndim == 2:
            model_attention = build_closed_update_attention_mask(
                attention_mask, input_position, output_position
            )
        model_attention = self._attention_for_backbone(model_attention)
        checkpointed = self.operation_checkpointing and torch.is_grad_enabled()
        operation_output: Any = None

        def operation(ids: torch.Tensor, mask: torch.Tensor, *prior: torch.Tensor):
            nonlocal operation_output
            prior_state = prior[0] if prior else None
            captured: list[torch.Tensor] = []

            def make_hook(layer_index: int):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    batch = torch.arange(hidden.shape[0], device=hidden.device)
                    collected = hidden[
                        batch, output_position.to(hidden.device)
                    ]
                    # Move every layer state to the embedding device so callers
                    # receive one contiguous tensor even for a sharded backbone.
                    state_device = self.backbone.get_input_embeddings().weight.device
                    collected = self.transition_out[layer_index](
                        collected.to(state_device)
                    )
                    captured.append(collected)
                    if prior_state is None:
                        return output
                    injected = self.transition_in[layer_index](
                        prior_state[:, layer_index]
                    )
                    patched = patch_address(hidden, input_position, injected)
                    return self._replace_hidden(output, patched)

                return hook

            handles = [
                layer.register_forward_hook(make_hook(index))
                for index, layer in enumerate(self._decoder_layers)
            ]
            try:
                outputs = self.backbone(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
                if not checkpointed:
                    operation_output = outputs
            finally:
                for handle in handles:
                    handle.remove()
            if len(captured) != self.layer_count:
                raise ValueError(
                    f"Expected {self.layer_count} collected states, got {len(captured)}"
                )
            return torch.stack(captured, dim=1)

        inputs = (input_ids, model_attention)
        if state_prev is not None:
            inputs = (*inputs, state_prev)
        state_next = (
            checkpoint(
                operation,
                *inputs,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            if checkpointed
            else operation(*inputs)
        )
        return LayerwiseRecurrentStepOutput(
            state=state_next, model_output=operation_output
        )

    def decode_from_state(
        self,
        state: torch.Tensor | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        prompt_ends: torch.Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Any:
        self._assert_hook_safe_gradient_mode()
        carrier = locate_token_positions(
            input_ids, self.state_token_id, count=1
        )[:, 0]
        validate_state(
            state,
            input_ids.shape[0],
            self.layer_count,
            self.hidden_size,
        )
        model_attention = attention_mask
        if prompt_ends is not None:
            model_attention = build_closed_decode_attention_mask(
                attention_mask, carrier, prompt_ends
            )
        model_attention = self._attention_for_backbone(model_attention)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": model_attention,
            "use_cache": use_cache,
            "return_dict": True,
            "output_hidden_states": output_hidden_states,
        }
        if labels is not None:
            kwargs["labels"] = labels
        checkpointed = (
            self.operation_checkpointing
            and torch.is_grad_enabled()
            and not use_cache
            and not output_hidden_states
        )
        operation_output: Any = None

        def operation(ids: torch.Tensor, mask: torch.Tensor, *prior: torch.Tensor):
            nonlocal operation_output
            decode_state = prior[0] if prior else None

            def make_hook(layer_index: int):
                def hook(_module, _inputs, output):
                    if decode_state is None:
                        return output
                    hidden = output[0] if isinstance(output, tuple) else output
                    patched = patch_address(
                        hidden, carrier, decode_state[:, layer_index]
                    )
                    return self._replace_hidden(output, patched)

                return hook

            handles = [
                layer.register_forward_hook(make_hook(index))
                for index, layer in enumerate(self._decoder_layers)
            ]
            local_kwargs = dict(kwargs)
            local_kwargs["input_ids"] = ids
            local_kwargs["attention_mask"] = mask
            if checkpointed:
                local_kwargs.pop("labels", None)
            try:
                output = self.backbone(**local_kwargs)
                if not checkpointed:
                    operation_output = output
            finally:
                for handle in handles:
                    handle.remove()
            return output.logits

        inputs = (input_ids, model_attention)
        if state is not None:
            inputs = (*inputs, state)
        if checkpointed:
            logits = checkpoint(
                operation,
                *inputs,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(logits=logits, loss=loss)
        operation(*inputs)
        return operation_output
