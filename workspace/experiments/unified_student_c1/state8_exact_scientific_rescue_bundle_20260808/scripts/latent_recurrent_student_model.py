#!/usr/bin/env python3
"""Single-vector recurrent latent routed through one address at every layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from layerwise_recurrent_student_model import decoder_layers, patch_address
from recurrent_student_model import (
    ResidualLowRankProjection,
    build_closed_decode_attention_mask,
    build_closed_update_attention_mask,
    locate_token_positions,
)


class LayerwiseLatentCodec(nn.Module):
    """Map one hidden-size latent to/from native per-layer residual spaces."""

    def __init__(self, hidden_size: int, layer_count: int, rank: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_count = int(layer_count)
        self.injectors = nn.ModuleList(
            [
                ResidualLowRankProjection(self.hidden_size, rank)
                for _ in range(self.layer_count)
            ]
        )
        self.collectors = nn.ModuleList(
            [
                ResidualLowRankProjection(self.hidden_size, rank)
                for _ in range(self.layer_count)
            ]
        )
        # Softmax starts uniform; training can select the causal layer band.
        self.layer_logits = nn.Parameter(torch.zeros(self.layer_count))
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

    def inject(self, latent: torch.Tensor, layer_index: int) -> torch.Tensor:
        return self.injectors[layer_index](self.norm(latent))

    def collect(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3 or trajectory.shape[1:] != (
            self.layer_count,
            self.hidden_size,
        ):
            raise ValueError(
                "Collector trajectory must have shape "
                f"[B,{self.layer_count},{self.hidden_size}]"
            )
        projected = torch.stack(
            [
                module(self.norm(trajectory[:, index]))
                for index, module in enumerate(self.collectors)
            ],
            dim=1,
        )
        weights = torch.softmax(self.layer_logits, dim=0).to(
            device=projected.device, dtype=projected.dtype
        )
        return (projected * weights.view(1, -1, 1)).sum(dim=1)


@dataclass
class LatentRecurrentStepOutput:
    state: torch.Tensor
    trajectory: torch.Tensor
    model_output: Any


class LatentRecurrentStudentModel(nn.Module):
    """Closed, one-vector latent with layer-native injection projections."""

    def __init__(
        self,
        backbone: nn.Module,
        state_token_id: int,
        codec_rank: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.state_token_id = int(state_token_id)
        self._decoder_layers = list(decoder_layers(backbone))
        self.layer_count = len(self._decoder_layers)
        config = getattr(backbone, "config", None)
        self.hidden_size = int(
            getattr(config, "hidden_size", 0)
            or getattr(config, "n_embd", 0)
            or backbone.get_input_embeddings().weight.shape[1]
        )
        self.codec = LayerwiseLatentCodec(
            self.hidden_size, self.layer_count, codec_rank
        )

    def initial_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> torch.Tensor:
        row = self.backbone.get_input_embeddings().weight[self.state_token_id]
        if device is not None:
            row = row.to(device)
        return row.unsqueeze(0).expand(batch_size, -1).clone()

    def _assert_hook_safe_gradient_mode(self) -> None:
        if torch.is_grad_enabled() and bool(
            getattr(self.backbone, "is_gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "Latent recurrent hooks require gradient checkpointing disabled"
            )

    @staticmethod
    def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def _check_state(self, state: torch.Tensor, batch_size: int) -> None:
        if tuple(state.shape) != (batch_size, self.hidden_size):
            raise ValueError(
                f"Latent state must have shape {(batch_size, self.hidden_size)}, "
                f"got {tuple(state.shape)}"
            )

    def update_state(
        self,
        state_prev: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> LatentRecurrentStepOutput:
        self._assert_hook_safe_gradient_mode()
        carriers = locate_token_positions(input_ids, self.state_token_id, count=2)
        input_position, output_position = carriers[:, 0], carriers[:, 1]
        self._check_state(state_prev, input_ids.shape[0])
        attention = attention_mask
        if attention_mask.ndim == 2:
            attention = build_closed_update_attention_mask(
                attention_mask, input_position, output_position
            )
        captured: list[torch.Tensor] = []

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                batch = torch.arange(hidden.shape[0], device=hidden.device)
                captured.append(
                    hidden[batch, output_position.to(hidden.device)].to(
                        self.codec.layer_logits.device
                    )
                )
                injected = self.codec.inject(
                    state_prev.to(self.codec.layer_logits.device), layer_index
                )
                patched = patch_address(hidden, input_position, injected)
                return self._replace_hidden(output, patched)

            return hook

        handles = [
            layer.register_forward_hook(make_hook(index))
            for index, layer in enumerate(self._decoder_layers)
        ]
        try:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        trajectory = torch.stack(captured, dim=1)
        state_next = self.codec.collect(trajectory)
        return LatentRecurrentStepOutput(state_next, trajectory, output)

    def decode_from_state(
        self,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        prompt_ends: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> Any:
        self._assert_hook_safe_gradient_mode()
        carrier = locate_token_positions(input_ids, self.state_token_id, count=1)[:, 0]
        self._check_state(state, input_ids.shape[0])
        attention = attention_mask
        if prompt_ends is not None:
            attention = build_closed_decode_attention_mask(
                attention_mask, carrier, prompt_ends
            )

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                injected = self.codec.inject(
                    state.to(self.codec.layer_logits.device), layer_index
                )
                patched = patch_address(hidden, carrier, injected)
                return self._replace_hidden(output, patched)

            return hook

        handles = [
            layer.register_forward_hook(make_hook(index))
            for index, layer in enumerate(self._decoder_layers)
        ]
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if labels is not None:
            kwargs["labels"] = labels
        try:
            return self.backbone(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
