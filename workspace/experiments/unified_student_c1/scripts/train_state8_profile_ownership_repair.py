#!/usr/bin/env python3
"""Align State8 training with the revised ownership/readability gate.

The existing profile continuation optimizes natural-language generation and a
small swap objective.  The scientific gate, however, fits a frozen State8
readout to post-cutoff KC rates and asks whether permuting student ownership
hurts.  This entrypoint adds that missing train-time objective while preserving
the production route: the updater sees observed interactions and the decoder
sees only State8 plus the fixed query.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_student_profile_readout as profile_trainer
from build_state8_student_profile_readout import PROFILE_TASK


PROFILE_FEATURE_VIEW = "layerwise_mean_slots_rp"
PROFILE_PROJECTION_DIM = 64
PROFILE_PROJECTION_SEED = 20260818
PROFILE_HEAD_INITIALIZATION_SEED = 20260820
PROFILE_BIAS_PRIOR_STRENGTH = 8.0
PROFILE_MIN_KC_TRAIN_SUPPORT = 4


def stable_projection(
    input_dim: int, output_dim: int, seed: int
) -> torch.Tensor:
    """Match the frozen C4 evaluator's deterministic Achlioptas projection."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Projection dimensions must be positive")
    rng = np.random.default_rng(seed)
    values = rng.choice(
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        size=(input_dim, output_dim),
        p=(1 / 6, 2 / 3, 1 / 6),
    )
    return torch.from_numpy(values * math.sqrt(3.0 / output_dim))


def profile_state_features(
    state: torch.Tensor,
    projection: torch.Tensor,
    normalization: str = "none",
) -> torch.Tensor:
    """Use exactly the feature view consumed by the frozen revised-C4 readout."""
    if state.ndim != 4:
        raise ValueError(f"Profile ownership requires [B,L,S,D], got {tuple(state.shape)}")
    if projection.ndim != 2 or projection.shape[0] != state.shape[-1]:
        raise ValueError("Profile projection is not aligned to State8 hidden width")
    per_layer = state.float().mean(dim=2)
    features = torch.einsum(
        "bld,dp->blp",
        per_layer,
        projection.to(device=state.device, dtype=torch.float32),
    )
    features = features.flatten(start_dim=1)
    if normalization == "layernorm":
        features = F.layer_norm(features, (features.shape[-1],))
    elif normalization != "none":
        raise ValueError(f"Unknown profile feature normalization: {normalization}")
    return features


def profile_target_logit_bias(
    train_jsonl: Path,
    output_dim: int = 30,
    prior_strength: float = PROFILE_BIAS_PRIOR_STRENGTH,
) -> torch.Tensor:
    """Initialize from shrunk eligible-training means, never held-out labels.

    A KC can legitimately have no eligible post-cutoff target among the
    training students even though it is present in the fixed output
    vocabulary.  Such a column must not be initialized from validation or test
    labels.  Unsupported columns use the training-wide mean.  Sparse columns
    are empirical-Bayes shrunk toward that same mean so one or two students
    cannot give a fresh head an extreme KC-specific prior.
    """
    if prior_strength < 0:
        raise ValueError("Profile bias prior strength must be non-negative")
    totals = torch.zeros(output_dim, dtype=torch.float64)
    counts = torch.zeros(output_dim, dtype=torch.float64)
    with train_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            if task is not None and str(task) != PROFILE_TASK:
                continue
            if "future_eligible_mask" not in row:
                continue
            values = row.get("future_smoothed_rates") or []
            mask = (
                row.get("ownership_training_mask")
                if "ownership_training_mask" in row
                else row.get("future_eligible_mask")
            ) or []
            if len(values) != output_dim or len(mask) != output_dim:
                raise ValueError("Profile target width does not match ownership head")
            for index, (value, eligible) in enumerate(zip(values, mask, strict=True)):
                if eligible:
                    totals[index] += float(value)
                    counts[index] += 1
    supported = counts > 0
    if not bool(supported.any()):
        raise ValueError("Training profiles contain no eligible KC targets")
    global_probability = totals.sum() / counts.sum()
    probability = torch.full(
        (output_dim,), float(global_probability), dtype=torch.float64
    )
    probability[supported] = (
        totals[supported] + prior_strength * global_probability
    ) / (counts[supported] + prior_strength)
    probability.clamp_(0.01, 0.99)
    return torch.logit(probability).to(torch.float32)


def profile_kc_support(
    train_jsonl: Path, output_dim: int = 30
) -> torch.Tensor:
    """Count eligible training students for each KC using train rows only."""
    counts = torch.zeros(output_dim, dtype=torch.float32)
    with train_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            if task is not None and str(task) != PROFILE_TASK:
                continue
            if "future_eligible_mask" not in row:
                continue
            mask = (
                row.get("ownership_training_mask")
                if "ownership_training_mask" in row
                else row.get("future_eligible_mask")
            ) or []
            if len(mask) != output_dim:
                raise ValueError("Profile target width does not match KC support")
            counts += torch.tensor(mask, dtype=torch.float32)
    if not bool((counts > 0).any()):
        raise ValueError("Training profiles contain no eligible KC targets")
    return counts


def profile_optimizer_groups(
    wrapper: torch.nn.Module, base_lr: float, head_lr: float
) -> list[dict[str, Any]]:
    """Give the fresh auxiliary head a separate, explicit learning rate."""
    if base_lr <= 0 or head_lr <= 0:
        raise ValueError("Profile optimizer learning rates must be positive")
    head_ids = {id(parameter) for parameter in wrapper.profile_ownership_head.parameters()}
    head: list[torch.nn.Parameter] = []
    representation: list[torch.nn.Parameter] = []
    for parameter in wrapper.parameters():
        if not parameter.requires_grad:
            continue
        (head if id(parameter) in head_ids else representation).append(parameter)
    if not head or not representation:
        raise ValueError("Profile optimizer requires head and representation parameters")
    return [
        {"params": representation, "lr": float(base_lr), "group_name": "representation"},
        {"params": head, "lr": float(head_lr), "group_name": "profile_ownership_head"},
    ]


def freeze_profile_ownership_head(wrapper: torch.nn.Module) -> None:
    """Freeze the train-only readout so ownership gradients shape State8.

    A fresh trainable 122k-parameter head can memorize the 663 fitting students
    without making the reusable representation more informative.  V5 instead
    initializes the readout from train-only KC priors, freezes it, and sends the
    exact numeric objective gradient through features into the state updater.
    """
    head = getattr(wrapper, "profile_ownership_head", None)
    if not isinstance(head, torch.nn.Module):
        raise ValueError("Profile ownership head has not been constructed")
    for parameter in head.parameters():
        parameter.requires_grad_(False)


def build_profile_ownership_head(
    layer_count: int,
    output_dim: int = 30,
    projection_dim: int = PROFILE_PROJECTION_DIM,
    initialization_seed: int = PROFILE_HEAD_INITIALIZATION_SEED,
) -> torch.nn.Linear:
    """Create the audited KC readout with rank-identical initialization.

    The distributed trainer deliberately seeds each rank differently for
    stochastic training operations and synchronizes gradients manually.  A
    newly constructed auxiliary module must therefore not inherit the
    rank-local RNG state: averaging gradients does not make parameters agree
    when their starting values differ.  Initialize from a private CPU
    generator so every replica starts byte-for-byte aligned without changing
    any process RNG stream.
    """
    if layer_count <= 0 or output_dim <= 0 or projection_dim <= 0:
        raise ValueError("Profile ownership head dimensions must be positive")
    input_dim = layer_count * projection_dim
    head = torch.nn.Linear(input_dim, output_dim, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(initialization_seed))
    bound = 1.0 / math.sqrt(input_dim)
    with torch.no_grad():
        head.weight.uniform_(-bound, bound, generator=generator)
        if head.bias is not None:
            head.bias.uniform_(-bound, bound, generator=generator)
    return head


def masked_profile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    negative_target: torch.Tensor,
    overlap_mask: torch.Tensor,
    *,
    ownership_margin: float,
    kc_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Future-rate regression plus a support-safe ownership hinge.

    ``kc_weight`` is fixed from training support only.  Setting unsupported or
    ultra-sparse columns to zero prevents a handful of noisy KC labels from
    dominating rows that happen to contain little else.  The remaining loss is
    normalized by effective weight, preserving its scale across students.
    """
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Prediction and profile targets must be aligned [B, KC]")
    if negative_target.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Profile target tensors must have identical shapes")
    if overlap_mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("Profile masks must be aligned boolean tensors")
    if overlap_mask.dtype != torch.bool or not bool(mask.any(dim=1).all()):
        raise ValueError("Every profile needs supported future KCs")
    if not bool(overlap_mask.any(dim=1).all()):
        raise ValueError("Every ownership pair needs jointly supported KCs")
    if ownership_margin < 0:
        raise ValueError("Ownership margin must be non-negative")
    if kc_weight is None:
        weight = prediction.new_ones((1, prediction.shape[1]))
    else:
        if kc_weight.ndim != 1 or kc_weight.shape[0] != prediction.shape[1]:
            raise ValueError("KC weights must have shape [KC]")
        if not bool(torch.isfinite(kc_weight).all()) or bool((kc_weight < 0).any()):
            raise ValueError("KC weights must be finite and non-negative")
        weight = kc_weight.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(0)
    native_weight = mask.to(prediction.dtype) * weight
    overlap_weight = overlap_mask.to(prediction.dtype) * weight
    if not bool((native_weight.sum(dim=1) > 0).all()):
        raise ValueError("Every profile needs a supported, train-reliable KC")
    if not bool((overlap_weight.sum(dim=1) > 0).all()):
        raise ValueError("Every ownership pair needs a jointly train-reliable KC")
    squared_native = (prediction - target).square()
    native = (squared_native * native_weight).sum(dim=1) / native_weight.sum(dim=1)
    squared_negative = (prediction - negative_target).square()
    matched_overlap = (squared_native * overlap_weight).sum(dim=1) / overlap_weight.sum(dim=1)
    mismatched = (squared_negative * overlap_weight).sum(dim=1) / overlap_weight.sum(dim=1)
    ownership = F.relu(
        prediction.new_tensor(float(ownership_margin)) + matched_overlap - mismatched
    )
    total = native.mean() + ownership.mean()
    return total, {
        "profile_future_rate_mse": native.mean(),
        "profile_matched_overlap_mse": matched_overlap.mean(),
        "profile_mismatched_overlap_mse": mismatched.mean(),
        "profile_ownership_margin_loss": ownership.mean(),
        "profile_ownership_preference": (mismatched - matched_overlap).mean(),
    }


def masked_profile_panel_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    negative_targets: torch.Tensor,
    overlap_masks: torch.Tensor,
    *,
    ownership_margin: float,
    kc_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Future-rate regression plus a mean margin over several students.

    The v4 objective always paired a student with one deterministic hardest
    negative.  A high-capacity head can memorize that mapping.  Averaging the
    hinge over a deterministic panel makes ownership a neighborhood property
    while retaining the exact masked-MSE semantics of the frozen evaluator.
    """
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Prediction and profile targets must be aligned [B, KC]")
    if negative_targets.ndim != 3 or negative_targets.shape[0] != prediction.shape[0]:
        raise ValueError("Negative profile panel must have shape [B, N, KC]")
    if negative_targets.shape[2] != prediction.shape[1]:
        raise ValueError("Negative profile panel KC width mismatch")
    if overlap_masks.shape != negative_targets.shape or overlap_masks.dtype != torch.bool:
        raise ValueError("Panel overlap masks must be aligned boolean tensors")
    if negative_targets.shape[1] <= 0:
        raise ValueError("Negative profile panel must be non-empty")
    if kc_weight is None:
        weight = prediction.new_ones((1, 1, prediction.shape[1]))
    else:
        if kc_weight.ndim != 1 or kc_weight.shape[0] != prediction.shape[1]:
            raise ValueError("KC weights must have shape [KC]")
        weight = kc_weight.to(prediction).reshape(1, 1, -1)
    native_weight = mask.to(prediction.dtype) * weight[:, 0]
    if not bool((native_weight.sum(dim=1) > 0).all()):
        raise ValueError("Every profile needs a supported, train-reliable KC")
    native = (
        (prediction - target).square() * native_weight
    ).sum(dim=1) / native_weight.sum(dim=1)
    native = native.mean()
    panel_weight = overlap_masks.to(prediction.dtype) * weight
    if not bool((panel_weight.sum(dim=2) > 0).all()):
        raise ValueError("Every panel member needs a jointly train-reliable KC")
    prediction_panel = prediction.unsqueeze(1)
    target_panel = target.unsqueeze(1)
    matched = (
        (prediction_panel - target_panel).square() * panel_weight
    ).sum(dim=2) / panel_weight.sum(dim=2)
    mismatched = (
        (prediction_panel - negative_targets).square() * panel_weight
    ).sum(dim=2) / panel_weight.sum(dim=2)
    preference = mismatched - matched
    ownership = F.relu(
        prediction.new_tensor(float(ownership_margin)) - preference
    )
    return native + ownership.mean(), {
        "profile_future_rate_mse": native,
        "profile_matched_overlap_mse": matched.mean(),
        "profile_mismatched_overlap_mse": mismatched.mean(),
        "profile_ownership_margin_loss": ownership.mean(),
        "profile_ownership_preference": preference.mean(),
        "profile_ownership_panel_min_preference": preference.min(dim=1).values.mean(),
        "profile_ownership_panel_size": prediction.new_tensor(
            float(negative_targets.shape[1])
        ),
    }


def row_targets(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    required = (
        "future_smoothed_rates",
        "future_eligible_mask",
        "ownership_negative_future_smoothed_rates",
        "ownership_overlap_mask",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Profile ownership row lacks {missing}")
    mask = torch.tensor(
        row.get("ownership_training_mask", row["future_eligible_mask"]),
        dtype=torch.bool,
        device=device,
    )
    overlap = torch.tensor(row["ownership_overlap_mask"], dtype=torch.bool, device=device)
    target = torch.tensor(
        [0.5 if value is None else float(value) for value in row["future_smoothed_rates"]],
        dtype=torch.float32,
        device=device,
    )
    negative = torch.tensor(
        [
            0.5 if value is None else float(value)
            for value in row["ownership_negative_future_smoothed_rates"]
        ],
        dtype=torch.float32,
        device=device,
    )
    return target.unsqueeze(0), mask.unsqueeze(0), negative.unsqueeze(0), overlap.unsqueeze(0)


def row_target_panel(
    row: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Return an ownership panel, falling back to the legacy one-negative row."""
    required = (
        "future_smoothed_rates",
        "future_eligible_mask",
        "ownership_negative_future_smoothed_rates",
        "ownership_overlap_mask",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Profile ownership row lacks {missing}")
    mask = torch.tensor(
        row.get("ownership_training_mask", row["future_eligible_mask"]),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    target = torch.tensor(
        [
            0.5 if value is None else float(value)
            for value in row["future_smoothed_rates"]
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    legacy_negative = torch.tensor(
        [
            0.5 if value is None else float(value)
            for value in row["ownership_negative_future_smoothed_rates"]
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    legacy_overlap = torch.tensor(
        row["ownership_overlap_mask"], dtype=torch.bool, device=device
    ).unsqueeze(0)
    if not (
        target.shape == mask.shape == legacy_negative.shape == legacy_overlap.shape
    ):
        raise ValueError("Profile target tensors must have identical shapes")
    raw_targets = row.get("ownership_negative_future_smoothed_rates_panel")
    raw_overlaps = row.get("ownership_overlap_mask_panel")
    if raw_targets is None and raw_overlaps is None:
        return target, mask, legacy_negative.unsqueeze(1), legacy_overlap.unsqueeze(1)
    if not isinstance(raw_targets, list) or not isinstance(raw_overlaps, list):
        raise ValueError("Ownership panel targets and masks must both be present")
    if not raw_targets or len(raw_targets) != len(raw_overlaps):
        raise ValueError("Ownership panel targets and masks must be non-empty and aligned")
    width = target.shape[1]
    if any(len(values) != width for values in raw_targets) or any(
        len(values) != width for values in raw_overlaps
    ):
        raise ValueError("Ownership panel KC width mismatch")
    negatives = torch.tensor(
        [
            [0.5 if value is None else float(value) for value in values]
            for values in raw_targets
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    overlaps = torch.tensor(raw_overlaps, dtype=torch.bool, device=device).unsqueeze(0)
    return target, mask, negatives, overlaps


def main() -> None:
    ownership_weight = float(os.environ.get("STATE8_PROFILE_OWNERSHIP_WEIGHT", "1"))
    ownership_margin = float(os.environ.get("STATE8_PROFILE_OWNERSHIP_MARGIN", "0.02"))
    head_lr_text = os.environ.get("STATE8_PROFILE_OWNERSHIP_HEAD_LEARNING_RATE")
    head_learning_rate = float(head_lr_text) if head_lr_text else None
    reset_head = os.environ.get("STATE8_PROFILE_OWNERSHIP_RESET_HEAD", "0") == "1"
    freeze_head = os.environ.get("STATE8_PROFILE_OWNERSHIP_FREEZE_HEAD", "0") == "1"
    feature_normalization = os.environ.get(
        "STATE8_PROFILE_OWNERSHIP_FEATURE_NORMALIZATION", "none"
    )
    minimum_kc_train_support = int(os.environ.get(
        "STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT",
        str(PROFILE_MIN_KC_TRAIN_SUPPORT),
    ))
    bias_prior_strength = float(os.environ.get(
        "STATE8_PROFILE_OWNERSHIP_BIAS_PRIOR_STRENGTH",
        str(PROFILE_BIAS_PRIOR_STRENGTH),
    ))
    if ownership_weight <= 0 or ownership_margin < 0:
        raise ValueError("Profile ownership weight must be positive and margin non-negative")
    if head_learning_rate is not None and head_learning_rate <= 0:
        raise ValueError("Profile ownership head learning rate must be positive")
    if feature_normalization not in {"none", "layernorm"}:
        raise ValueError("Profile ownership feature normalization is invalid")
    if minimum_kc_train_support <= 0 or bias_prior_strength < 0:
        raise ValueError("KC support must be positive and bias prior non-negative")

    original_parse = argparse.ArgumentParser.parse_args
    original_objective = trainer.backward_training_objective
    original_payload = trainer.recurrent_state_payload
    original_load = trainer.load_recurrent_state_payload
    output_dir: Path | None = None
    target_bias: torch.Tensor | None = None
    kc_weight: torch.Tensor | None = None

    def parse(parser, *args, **kwargs):
        nonlocal output_dir, target_bias, kc_weight
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        namespace.profile_ownership_weight = ownership_weight
        namespace.profile_ownership_margin = ownership_margin
        namespace.profile_ownership_head_input = PROFILE_FEATURE_VIEW
        namespace.profile_ownership_projection_dim = PROFILE_PROJECTION_DIM
        namespace.profile_ownership_projection_seed = PROFILE_PROJECTION_SEED
        namespace.profile_ownership_head_initialization_seed = (
            PROFILE_HEAD_INITIALIZATION_SEED
        )
        namespace.profile_ownership_supervision_visible_at_inference = False
        namespace.profile_ownership_min_kc_train_support = minimum_kc_train_support
        namespace.profile_ownership_bias_prior_strength = bias_prior_strength
        support = profile_kc_support(Path(namespace.train_jsonl))
        kc_weight = (support >= minimum_kc_train_support).to(torch.float32)
        if not bool(kc_weight.any()):
            raise ValueError("No KC meets the minimum training-support threshold")
        if head_learning_rate is not None:
            namespace.profile_ownership_head_learning_rate = head_learning_rate
        namespace.profile_ownership_head_frozen = freeze_head
        if reset_head:
            namespace.profile_ownership_reset_head = True
            namespace.profile_ownership_bias_initialization = "train_profile_kc_shrunk_mean_logit"
            target_bias = profile_target_logit_bias(
                Path(namespace.train_jsonl), prior_strength=bias_prior_strength
            )
        if feature_normalization != "none":
            namespace.profile_ownership_feature_normalization = feature_normalization
        return namespace

    def payload(wrapper, config):
        value = original_payload(wrapper, config)
        head = getattr(wrapper, "profile_ownership_head", None)
        if head is not None:
            value["profile_ownership_head"] = head.state_dict()
        return value

    def load(wrapper, value):
        original_load(wrapper, value)
        state = value.get("profile_ownership_head")
        if reset_head:
            state = None
        if state is not None:
            if not hasattr(wrapper, "profile_ownership_head"):
                output_dim, input_dim = state["weight"].shape
                if input_dim % PROFILE_PROJECTION_DIM:
                    raise ValueError("Saved ownership head has an invalid feature width")
                wrapper.profile_ownership_head = build_profile_ownership_head(
                    input_dim // PROFILE_PROJECTION_DIM,
                    output_dim,
                    PROFILE_PROJECTION_DIM,
                ).to(next(wrapper.parameters()).device)
            wrapper.profile_ownership_head.load_state_dict(state)

    def objective(
        wrapper,
        tokenizer,
        row,
        causal_row,
        cache,
        online_kd_cache,
        namespace,
        device,
        state_token_id,
        loss_scale,
        training_progress,
    ):
        empty = dict(
            profile_future_rate_mse=0.0,
            profile_matched_overlap_mse=0.0,
            profile_mismatched_overlap_mse=0.0,
            profile_ownership_margin_loss=0.0,
            profile_ownership_preference=0.0,
            profile_ownership_effective_weight=0.0,
        )
        def auxiliary(*, wrapper, state, row, device):
            if str(row.get("task")) != PROFILE_TASK:
                return state.new_zeros((), dtype=torch.float32), empty
            if row.get("ownership_training_eligible") is False:
                output = dict(empty)
                output["profile_ownership_skipped_no_seen_support"] = True
                return state.new_zeros((), dtype=torch.float32), output
            target, mask, negative, overlap = row_targets(row, device)
            features = profile_state_features(
                state,
                wrapper.profile_ownership_projection,
                normalization=feature_normalization,
            )
            head = wrapper.profile_ownership_head
            prediction = torch.sigmoid(head(features))
            loss, diagnostics = masked_profile_loss(
                prediction, target, mask, negative, overlap,
                ownership_margin=ownership_margin,
                kc_weight=kc_weight,
            )
            value = float(loss.detach())
            output = {
                key: float(metric.detach()) for key, metric in diagnostics.items()
            }
            output["profile_ownership_effective_weight"] = ownership_weight
            output["profile_ownership_weighted_loss"] = ownership_weight * value
            weighted_loss = loss * ownership_weight
            del features, prediction
            return weighted_loss, output

        record = original_objective(
            wrapper, tokenizer, row, causal_row, cache, online_kd_cache,
            namespace, device, state_token_id, loss_scale, training_progress,
            state_auxiliary_objective=auxiliary,
        )
        for key, value in empty.items():
            record.setdefault(key, value)
        return record

    # The base trainer creates its optimizer after building the wrapper, so the
    # auxiliary module must exist at wrapper construction time.  Patch the
    # layerwise class locally with an initializer that installs a 64->30 head.
    original_layerwise = trainer.LayerwiseRecurrentStudentModel

    class OwnershipLayerwise(original_layerwise):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            projection = stable_projection(
                self.hidden_size,
                PROFILE_PROJECTION_DIM,
                PROFILE_PROJECTION_SEED,
            ).to(self.backbone.get_input_embeddings().weight.device)
            self.register_buffer(
                "profile_ownership_projection", projection, persistent=False
            )
            self.profile_ownership_head = build_profile_ownership_head(
                self.layer_count, 30, PROFILE_PROJECTION_DIM
            ).to(self.backbone.get_input_embeddings().weight.device)
            if reset_head:
                if target_bias is None:
                    raise RuntimeError("Profile target bias was not initialized")
                with torch.no_grad():
                    # A frozen zero-weight head would make the numeric loss
                    # independent of State8 and send exactly zero gradient to
                    # the updater.  Keep the deterministic random coding matrix
                    # in the frozen-head recipe; v4's trainable-head reset still
                    # starts at the train-only population prior as before.
                    if not freeze_head:
                        self.profile_ownership_head.weight.zero_()
                    self.profile_ownership_head.bias.copy_(
                        target_bias.to(self.profile_ownership_head.bias.device)
                    )
            if freeze_head:
                freeze_profile_ownership_head(self)

        def optimizer_parameter_groups(self, base_lr: float):
            if freeze_head:
                return None
            if head_learning_rate is None:
                return None
            return profile_optimizer_groups(self, base_lr, head_learning_rate)

    argparse.ArgumentParser.parse_args = parse
    trainer.LayerwiseRecurrentStudentModel = OwnershipLayerwise
    trainer.backward_training_objective = objective
    trainer.recurrent_state_payload = payload
    trainer.load_recurrent_state_payload = load
    try:
        profile_trainer.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Ownership trainer did not observe output directory")
            history = json.loads((output_dir / "trainer_history.json").read_text())
            run_config = json.loads((output_dir / "run_config.json").read_text())
            diagnostic = bool(
                run_config.get("diagnostic_max_micro_steps") is not None
                or run_config.get("diagnostic_stop_after_round") is not None
                or run_config.get("diagnostic_weight_only_warm_start") is True
            )
            applied = [
                row for row in history
                if float(row.get("profile_ownership_effective_weight", 0)) > 0
            ]
            if not applied:
                raise RuntimeError("Profile ownership objective was never exercised")
            receipt = {
                "stage": "state8_profile_ownership_repair_v1",
                "status": "diagnostic_complete" if diagnostic else "complete",
                "diagnostic": diagnostic,
                "applied_rows": len(applied),
                "weight": ownership_weight,
                "margin": ownership_margin,
                "head_input": PROFILE_FEATURE_VIEW,
                "projection_dim": PROFILE_PROJECTION_DIM,
                "projection_seed": PROFILE_PROJECTION_SEED,
                "head_initialization_seed": PROFILE_HEAD_INITIALIZATION_SEED,
                "feature_normalization": feature_normalization,
                "head_learning_rate": head_learning_rate,
                "head_frozen": freeze_head,
                "head_reset_from_training_targets": reset_head,
                "head_output": "30_KC_future_rates",
                "minimum_kc_train_support": minimum_kc_train_support,
                "bias_prior_strength": bias_prior_strength,
                "sparse_kc_policy": "exclude_below_train_support_then_weight_equally",
                "negative": "same_split_hard_mismatched_student",
                "raw_history_visible_to_decoder": False,
                "student_id_visible_to_decoder": False,
                "ownership_supervision_visible_at_inference": False,
                "final_inference_route": "strict_state_only",
            }
            (output_dir / "profile_ownership_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.LayerwiseRecurrentStudentModel = original_layerwise
        trainer.backward_training_objective = original_objective
        trainer.recurrent_state_payload = original_payload
        trainer.load_recurrent_state_payload = original_load


if __name__ == "__main__":
    main()
