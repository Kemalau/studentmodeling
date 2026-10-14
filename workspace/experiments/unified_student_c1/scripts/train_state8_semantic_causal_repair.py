#!/usr/bin/env python3
"""Repair State8 semantic causality without exposing labels at inference.

The previous State8 repair weighted canonical labels only on ordinary real-row
generation.  C4 diagnosis twins use the trainer's per-sequence loss and thus
silently bypassed that weighting.  Its random-state branch also trained the
recipient diagnosis as the random state's gold output, contrary to the
semantic C4 requirement that an unstructured state express insufficient
evidence rather than a stable misconception.

This entrypoint keeps the frozen production trainer intact and adds three
independently switchable, train-only objectives:

* canonical-label weighting on C4 diagnosis sequences;
* an explicit ``insufficient_evidence`` target for norm-matched random state;
* same-rule attraction and different-rule separation directly on recurrent
  states.  This metric loss has no inference-time classifier or label input.

The final decoder route remains strictly state plus current query.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_state8_stateonly_teacher_repair as repair
from t3_judge_utils import natural_schema_valid


NEUTRAL_TARGET = (
    "<label>insufficient_evidence</label>\n"
    "<diagnosis>The supplied latent state does not contain coherent evidence "
    "for a stable student misconception.</diagnosis>\n"
    "<evidence>An unstructured or random state cannot support a specific "
    "diagnostic attribution.</evidence>"
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def cli_value(name: str) -> str:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise ValueError(f"Missing required argument {name}")
    return sys.argv[sys.argv.index(name) + 1]


def schema_weighted_per_sequence_factory(
    tokenizer, label_weight: float
) -> Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]:
    """Return per-sequence CE with extra weight only on ``<label>`` values."""
    if label_weight < 1.0:
        raise ValueError("Causal label weight must be at least one")
    opener = tokenizer("<label>", add_special_tokens=False)["input_ids"]
    closer = tokenizer("</label>", add_special_tokens=False)["input_ids"]
    if not opener or not closer:
        raise ValueError("Tokenizer produced an empty label delimiter")

    def loss(
        logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        shifted = labels[:, 1:].contiguous()
        valid = shifted.ne(-100)
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("Every sequence must have a supervised target token")
        outputs: list[torch.Tensor] = []
        for index in range(logits.shape[0]):
            mask = valid[index]
            values = shifted[index][mask]
            token_losses = F.cross_entropy(
                logits[index, :-1][mask].float(), values, reduction="none"
            )
            weights = token_losses.new_ones(token_losses.shape)
            ids = values.tolist()
            begin = repair.find_subsequence(ids, opener)
            end = (
                repair.find_subsequence(ids, closer, begin + len(opener))
                if begin >= 0
                else -1
            )
            if begin >= 0 and end >= 0:
                weights[begin + len(opener) : end] = float(label_weight)
            total = (token_losses * weights).sum()
            outputs.append(total if reduction == "sum" else total / weights.sum())
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"Unknown per-sequence reduction: {reduction}")
        return torch.stack(outputs)

    loss.label_weight = float(label_weight)
    loss.label_opener_ids = opener
    loss.label_closer_ids = closer
    return loss


def state_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim not in {3, 4}:
        raise ValueError(
            "Semantic state tensors must be aligned [B,L,D] or [B,L,S,D]"
        )
    return F.cosine_similarity(left.float(), right.float(), dim=-1).mean()


def semantic_state_geometry_loss(
    recipient: torch.Tensor,
    same_rule: torch.Tensor,
    donor: torch.Tensor,
    *,
    different_max_cosine: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Attract same-rule states and separate both from a different rule."""
    if not -1.0 <= different_max_cosine <= 1.0:
        raise ValueError("different_max_cosine must be in [-1, 1]")
    positive = state_cosine(recipient, same_rule)
    recipient_negative = state_cosine(recipient, donor)
    control_negative = state_cosine(same_rule, donor)
    threshold = positive.new_tensor(float(different_max_cosine))
    attraction = 1.0 - positive
    separation = 0.5 * (
        F.relu(recipient_negative - threshold)
        + F.relu(control_negative - threshold)
    )
    return attraction + separation, {
        "semantic_same_rule_cosine": positive,
        "semantic_recipient_donor_cosine": recipient_negative,
        "semantic_control_donor_cosine": control_negative,
        "semantic_attraction_loss": attraction,
        "semantic_separation_loss": separation,
    }


def backward_semantic_state_geometry(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    *,
    different_max_cosine: float,
    weight: float,
    loss_scale: float,
) -> tuple[float, dict[str, float]]:
    """Backpropagate exact pairwise geometry without retaining three graphs."""
    update_keys = (
        "recipient_native_update_texts",
        "same_rule_control_update_texts",
        "donor_update_texts",
    )
    with torch.no_grad():
        probes = [
            trainer.rollout_state(
                wrapper,
                tokenizer,
                list(row[key]),
                alias,
                max_length,
                device,
            )
            for key in update_keys
        ]
        probe_loss, probe_metrics = semantic_state_geometry_loss(
            probes[0],
            probes[1],
            probes[2],
            different_max_cosine=different_max_cosine,
        )
        recipient_active = float(
            probe_metrics["semantic_recipient_donor_cosine"]
            > different_max_cosine
        )
        control_active = float(
            probe_metrics["semantic_control_donor_cosine"]
            > different_max_cosine
        )
        metrics = {key: float(value) for key, value in probe_metrics.items()}
        loss_value = float(probe_loss)
    recipient_probe, control_probe, donor_probe = probes

    # Split every symmetric cosine term equally across its two endpoints.
    # The summed branch gradients are exactly those of
    # ``semantic_state_geometry_loss`` while only one updater graph is live.
    recipient = trainer.rollout_state(
        wrapper,
        tokenizer,
        list(row[update_keys[0]]),
        alias,
        max_length,
        device,
    )
    recipient_loss = 1.0 - state_cosine(recipient, control_probe)
    recipient_loss = recipient_loss + 0.5 * recipient_active * (
        state_cosine(recipient, donor_probe) - different_max_cosine
    )
    trainer.backward_loss_branch(recipient_loss, weight, loss_scale)
    del recipient, recipient_loss

    control = trainer.rollout_state(
        wrapper,
        tokenizer,
        list(row[update_keys[1]]),
        alias,
        max_length,
        device,
    )
    control_loss = 1.0 - state_cosine(control, recipient_probe)
    control_loss = control_loss + 0.5 * control_active * (
        state_cosine(control, donor_probe) - different_max_cosine
    )
    trainer.backward_loss_branch(control_loss, weight, loss_scale)
    del control, control_loss

    donor = trainer.rollout_state(
        wrapper,
        tokenizer,
        list(row[update_keys[2]]),
        alias,
        max_length,
        device,
    )
    donor_loss = 0.5 * recipient_active * (
        state_cosine(donor, recipient_probe) - different_max_cosine
    ) + 0.5 * control_active * (
        state_cosine(donor, control_probe) - different_max_cosine
    )
    if recipient_active or control_active:
        trainer.backward_loss_branch(donor_loss, weight, loss_scale)
    del donor, donor_loss, probes
    return loss_value, metrics


def is_diagnosis_pair(row: dict[str, Any]) -> bool:
    return str(row.get("causal_target_kind")) == "diagnosis_natural_schema"


def early_label_query_mode() -> str:
    value = os.environ.get("STATE8_EARLY_LABEL_QUERY_MODE", "with_query")
    aliases = {
        "with_query": "with_query",
        "query": "with_query",
        "state_only": "state_only",
        "no_query": "state_only",
    }
    if value not in aliases:
        raise ValueError(
            "STATE8_EARLY_LABEL_QUERY_MODE must be with_query or state_only"
        )
    return aliases[value]


def early_label_decode_text(
    decode_text: str, query_mode: str | None = None
) -> str:
    """Build a train-only label readout without exposing history or candidates.

    The natural diagnosis objective teacher-forces a long target.  Candidate
    NLL can then be reduced by continuing from candidate-specific gold words,
    even when the state does not determine the first free-generation branch.
    This auxiliary view keeps the identical closed state/current-query suffix
    but asks for the canonical identifier directly, so the first supervised
    label token must be selected from the state rather than a gold diagnosis
    prefix.  It is never used by the production inference route.
    """
    text = str(decode_text)
    marker = "<STUDENT_STATE>"
    if text.count(marker) != 1:
        raise ValueError("Early-label readout requires exactly one state marker")
    mode = query_mode or early_label_query_mode()
    if mode == "state_only":
        suffix = marker
    elif mode == "with_query":
        suffix = marker + text.split(marker, 1)[1]
        suffix = suffix.rstrip()
        if suffix.endswith("Answer:"):
            suffix = suffix[: -len("Answer:")].rstrip()
    else:
        raise ValueError(f"Unknown early-label query mode: {mode}")
    if "<student_history>" in suffix.lower():
        raise ValueError("Early-label readout leaked raw history")
    return (
        "You are performing an auxiliary training-only state readout.\n"
        f"{suffix}\n"
        "Return only the exact canonical misconception label identifier. "
        "Do not emit XML tags, a diagnosis, evidence, or alternatives.\n"
        "Canonical misconception label:"
    )


def early_label_target(value: Any) -> str:
    """Return a non-empty identifier whose first token has no gold prefix."""
    label = str(value).strip()
    if not label or any(character in label for character in "\r\n<>"):
        raise ValueError(f"Invalid early-label target: {value!r}")
    return label


def is_early_label_pair(row: dict[str, Any] | None) -> bool:
    if not is_diagnosis_pair(row or {}):
        return False
    native = str((row or {}).get("native_target_label") or "").strip()
    donor = str((row or {}).get("donor_target_label") or "").strip()
    return bool(native and donor and native != donor)


def early_binding_labels(row: dict[str, Any]) -> tuple[str, str, str]:
    """Prefer readable labels while preserving numeric-ID compatibility."""
    native = str(
        row.get("native_semantic_label") or row.get("native_target_label") or ""
    ).strip()
    donor = str(
        row.get("donor_semantic_label") or row.get("donor_target_label") or ""
    ).strip()
    mode = (
        "semantic_identifier_only_no_gold_prefix"
        if row.get("native_semantic_label") and row.get("donor_semantic_label")
        else "identifier_only_no_gold_prefix"
    )
    if not native or not donor or native == donor:
        raise ValueError("Early-label binding requires distinct non-empty labels")
    return native, donor, mode


def backward_early_label_binding(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    weight: float,
    loss_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    """Bind native/donor states before natural-language teacher forcing helps.

    The same decoder and recurrent state are used; this is not a classifier
    head.  Only the train-time prompt/target are shortened.  The production
    diagnosis remains natural language and is judged semantically.
    """
    if weight <= 0:
        raise ValueError("Early-label binding weight must be positive")
    if not is_early_label_pair(row):
        raise ValueError("Early-label binding requires distinct diagnosis labels")
    auxiliary = dict(row)
    auxiliary["decode_text"] = early_label_decode_text(str(row["decode_text"]))
    native_label, donor_label, target_mode = early_binding_labels(row)
    auxiliary["native_target"] = early_label_target(native_label)
    auxiliary["donor_target"] = early_label_target(donor_label)
    value, margin_value, diagnostics = trainer.backward_paired_causal_objective(
        wrapper,
        tokenizer,
        auxiliary,
        alias,
        max_length,
        device,
        margin,
        "sum",
        "directional",
        1.0,
        1.0,
        weight,
        loss_scale,
    )
    return value, margin_value, {
        "early_label_native_advantage": diagnostics["causal_native_advantage"],
        "early_label_donor_advantage": diagnostics["causal_donor_advantage"],
        "early_label_preference": diagnostics["causal_preference"],
        "early_label_target_mode": target_mode,
        "early_label_preference_reduction": "sum",
    }


def backward_early_label_controls(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    random_seed: int,
    weight: float,
    loss_scale: float,
    *,
    random_weight: float,
    same_rule_weight: float,
) -> tuple[float, dict[str, float | bool | str]]:
    """Make identifier-only readout reject random state and preserve same rule.

    The ordinary paired label objective binds native and donor trajectories.  It
    does not by itself say what an unstructured state should emit, nor does it
    require another trajectory with the recipient rule to keep the recipient
    identifier.  These two train-only branches close those controls using the
    same query-free identifier prompt as the primary binding objective.
    """
    if random_weight < 0 or same_rule_weight < 0:
        raise ValueError("Early-label control weights must be non-negative")
    if weight <= 0 or margin < 0:
        raise ValueError("Early-label controls require positive weight and margin")
    if random_weight == 0 and same_rule_weight == 0:
        return 0.0, {
            "early_label_control_random_applied": False,
            "early_label_control_same_rule_applied": False,
            "early_label_control_neutral_target": "NONE",
        }

    auxiliary = dict(row)
    auxiliary["decode_text"] = early_label_decode_text(
        str(row["decode_text"]), "state_only"
    )
    native_label, donor_label, _ = early_binding_labels(row)
    native_target = early_label_target(native_label)
    donor_target = early_label_target(donor_label)
    neutral_target = early_label_target("NONE")
    was_training = wrapper.training
    wrapper.eval()

    total = 0.0
    metrics: dict[str, float | bool | str] = {
        "early_label_control_random_applied": False,
        "early_label_control_same_rule_applied": False,
        "early_label_control_neutral_target": neutral_target,
    }

    if same_rule_weight > 0 and row.get("same_rule_control_update_texts"):
        same_state = trainer.rollout_state(
            wrapper,
            tokenizer,
            list(row["same_rule_control_update_texts"]),
            alias,
            max_length,
            device,
        )
        losses, preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [same_state, same_state],
            auxiliary["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            "sum",
        )
        advantage = preferences[1] - preferences[0]
        active = float(float(advantage.detach()) < float(margin))
        branch = recipient_control_two_view_loss(
            losses, preferences, margin=float(margin), active=active
        )
        value = float(branch.detach())
        trainer.backward_loss_branch(
            branch, float(weight) * float(same_rule_weight), loss_scale
        )
        total += float(same_rule_weight) * value
        metrics.update(
            early_label_control_same_rule_applied=True,
            early_label_control_same_rule_loss=value,
            early_label_control_same_rule_advantage=float(advantage.detach()),
        )
        del same_state, losses, preferences, advantage, branch

    if random_weight > 0:
        with torch.no_grad():
            native_reference = trainer.rollout_state(
                wrapper,
                tokenizer,
                list(row["native_update_texts"]),
                alias,
                max_length,
                device,
            )
            random_state = trainer.norm_matched_random_state(
                native_reference, random_seed
            )
        del native_reference
        random_values: list[float] = []
        random_advantages: list[float] = []
        for wrong_target in (native_target, donor_target):
            losses, preferences = trainer.decode_candidate_losses(
                wrapper,
                tokenizer,
                [random_state, random_state],
                auxiliary["decode_text"],
                [neutral_target, wrong_target],
                alias,
                max_length,
                device,
                "sum",
            )
            advantage = preferences[1] - preferences[0]
            active = float(float(advantage.detach()) < float(margin))
            branch = recipient_control_two_view_loss(
                losses, preferences, margin=float(margin), active=active
            )
            value = float(branch.detach())
            trainer.backward_loss_branch(
                branch,
                float(weight) * float(random_weight) / 2.0,
                loss_scale,
            )
            random_values.append(value)
            random_advantages.append(float(advantage.detach()))
            del losses, preferences, advantage, branch
        del random_state
        mean_value = sum(random_values) / len(random_values)
        total += float(random_weight) * mean_value
        metrics.update(
            early_label_control_random_applied=True,
            early_label_control_random_loss=mean_value,
            early_label_control_random_min_advantage=min(random_advantages),
        )

    wrapper.train(was_training)
    return total, metrics


def aggregate_early_label_counts(
    history: list[dict[str, Any]], configured_weight: float
) -> dict[str, Any]:
    applied = [
        row
        for row in history
        if float(row.get("early_label_binding_effective_weight", 0.0)) > 0.0
    ]
    if configured_weight > 0 and not applied:
        raise RuntimeError(
            "Positive early-label binding weight produced no history records"
        )
    return {
        "eligible": len(applied),
        "applied": len(applied),
        "by_source": dict(
            sorted(Counter(str(row.get("early_label_binding_source") or "unknown") for row in applied).items())
        ),
        "by_rank": dict(
            sorted(Counter(str(row.get("rank", "single")) for row in applied).items())
        ),
        "audit_source": "global_trainer_history",
    }


def is_eedi_semantic_pair(row: dict[str, Any]) -> bool:
    return bool(row.get("eedi_semantic_pair")) and str(
        row.get("eedi_pair_supervision_visibility")
    ) == "train_only_not_decoder_input"


def aggregate_geometry_counts(
    history: list[dict[str, Any]],
    *,
    geometry_weight: float,
    eedi_geometry_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit geometry execution from the globally gathered trainer history.

    A resumed distributed job may enter rank-zero finalization without running
    another training micro-step.  Process-local counters are therefore not an
    authoritative receipt: they are reset on the finalizer even though the
    checkpoint history contains the completed geometry branches.  The final
    trainer history is gathered across every rank and is the durable source of
    truth for both fresh and resumed runs.
    """

    def summarize(marker: str, weight: float) -> dict[str, Any]:
        applied = [row for row in history if marker in row]
        by_rank = Counter(str(row.get("rank", "single")) for row in applied)
        summary: dict[str, Any] = {
            "eligible": len(applied),
            "applied": len(applied),
            "by_rank": dict(sorted(by_rank.items())),
            "audit_source": "global_trainer_history",
        }
        if weight > 0 and not applied:
            raise RuntimeError(
                f"Positive semantic geometry weight produced no {marker} records"
            )
        return summary

    return (
        summarize("semantic_geometry_effective_weight", geometry_weight),
        summarize("eedi_semantic_geometry_effective_weight", eedi_geometry_weight),
    )


def random_neutral_two_view_branch_loss(
    losses: torch.Tensor,
    preferences: torch.Tensor,
    *,
    margin: float,
    active: float,
) -> torch.Tensor:
    """Return one half of the exact split random-neutral objective.

    The original implementation decoded ``[neutral, native, donor]`` in one
    gradient-carrying batch.  On Qwen3-32B that three-view graph can exceed an
    A100 80GB even though native and donor are conditionally independent once
    the detached hinge activations are fixed.  The neutral gold CE is divided
    equally between the two deterministic forwards, while each preference
    hinge remains on its corresponding pair.  Their scalar sum and gradient
    are therefore identical to the original three-view objective away from
    the usual hinge boundary.
    """
    if losses.shape[0] != 2 or preferences.shape[0] != 2:
        raise ValueError("Random-neutral branch requires two views")
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    if active not in (0.0, 1.0):
        raise ValueError("Random-neutral hinge activations must be binary")

    advantage = preferences[1] - preferences[0]
    branch_margin = float(active) * (
        advantage.new_tensor(float(margin)) - advantage
    ) / 4.0
    # Each deterministic two-view forward contributes half of the neutral CE.
    return losses[0] / 6.0 + branch_margin


def recipient_control_two_view_loss(
    losses: torch.Tensor,
    preferences: torch.Tensor,
    *,
    margin: float,
    active: float,
) -> torch.Tensor:
    """Keep an answer/control state on the recipient target.

    This is the direct output-space control used for non-diagnosis random
    states and same-rule states.  Unlike the legacy single-random triplet, the
    loss can be applied to several independent isotropic directions without
    changing the native/donor swap objective.
    """
    if losses.shape[0] != 2 or preferences.shape[0] != 2:
        raise ValueError("Recipient control requires two views")
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    if active not in (0.0, 1.0):
        raise ValueError("Recipient-control hinge activation must be binary")
    advantage = preferences[1] - preferences[0]
    return losses[0] + float(active) * (
        advantage.new_tensor(float(margin)) - advantage
    )


def control_random_seeds(base_seed: int, count: int) -> list[int]:
    """Return independent deterministic seeds using the frozen-C4 stride."""
    if count <= 0:
        raise ValueError("Random-control sample count must be positive")
    return [int(base_seed) + 1_000_003 * index for index in range(count)]


def backward_control_robust_extras(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    preference_reduction: str,
    random_seed: int,
    weight: float,
    loss_scale: float,
    *,
    random_samples: int,
    random_weight: float,
    same_rule_weight: float,
    diagnosis_random_neutral: bool,
) -> tuple[float, float, dict[str, Any]]:
    """Backpropagate held-out-control-style random and same-rule extras.

    The base specificity objective already sees one deterministic random
    direction.  Here we add ``random_samples - 1`` directions using the exact
    isotropic, per-layer/per-slot norm-matched construction and seed stride of
    the frozen evaluator.  Each extra graph is at most two views.  Same-rule
    states receive an explicit recipient-output loss rather than only a cosine
    attraction loss.
    """
    if random_samples <= 0:
        raise ValueError("random_samples must be positive")
    if random_weight < 0 or same_rule_weight < 0:
        raise ValueError("Control-robust weights must be non-negative")
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    native_target = str(row["native_target"])
    donor_target = str(row["donor_target"])
    was_training = wrapper.training
    wrapper.eval()
    random_losses: list[float] = []
    random_margins: list[float] = []

    extra_seeds = control_random_seeds(random_seed, random_samples)[1:]
    if extra_seeds and random_weight > 0:
        with torch.no_grad():
            reference = trainer.rollout_state(
                wrapper,
                tokenizer,
                row["native_update_texts"],
                alias,
                max_length,
                device,
            )
        per_random_weight = float(weight) * float(random_weight) / len(extra_seeds)
        for seed in extra_seeds:
            random_state = trainer.norm_matched_random_state(reference, seed)
            if diagnosis_random_neutral and is_diagnosis_pair(row):
                targets = (native_target, donor_target)
                branch_values: list[float] = []
                branch_margins: list[float] = []
                for target in targets:
                    with torch.no_grad():
                        _, probe_preferences = trainer.decode_candidate_losses(
                            wrapper,
                            tokenizer,
                            [random_state, random_state],
                            row["decode_text"],
                            [NEUTRAL_TARGET, target],
                            alias,
                            max_length,
                            device,
                            preference_reduction,
                        )
                        advantage_value = float(
                            probe_preferences[1] - probe_preferences[0]
                        )
                    active = float(advantage_value < float(margin))
                    losses, preferences = trainer.decode_candidate_losses(
                        wrapper,
                        tokenizer,
                        [random_state, random_state],
                        row["decode_text"],
                        [NEUTRAL_TARGET, target],
                        alias,
                        max_length,
                        device,
                        preference_reduction,
                    )
                    branch = random_neutral_two_view_branch_loss(
                        losses,
                        preferences,
                        margin=float(margin),
                        active=active,
                    )
                    branch_values.append(float(branch.detach()))
                    branch_margins.append(max(0.0, float(margin) - advantage_value))
                    trainer.backward_loss_branch(
                        branch, per_random_weight, loss_scale
                    )
                    del losses, preferences, branch
                random_losses.append(sum(branch_values))
                random_margins.append(sum(branch_margins) / len(branch_margins))
            else:
                with torch.no_grad():
                    _, probe_preferences = trainer.decode_candidate_losses(
                        wrapper,
                        tokenizer,
                        [random_state, random_state],
                        row["decode_text"],
                        [native_target, donor_target],
                        alias,
                        max_length,
                        device,
                        preference_reduction,
                    )
                    advantage_value = float(
                        probe_preferences[1] - probe_preferences[0]
                    )
                active = float(advantage_value < float(margin))
                losses, preferences = trainer.decode_candidate_losses(
                    wrapper,
                    tokenizer,
                    [random_state, random_state],
                    row["decode_text"],
                    [native_target, donor_target],
                    alias,
                    max_length,
                    device,
                    preference_reduction,
                )
                branch = recipient_control_two_view_loss(
                    losses,
                    preferences,
                    margin=float(margin),
                    active=active,
                )
                random_losses.append(float(branch.detach()))
                random_margins.append(max(0.0, float(margin) - advantage_value))
                trainer.backward_loss_branch(branch, per_random_weight, loss_scale)
                del losses, preferences, branch
            del random_state
        del reference

    same_rule_loss_value = 0.0
    same_rule_margin_value = 0.0
    same_rule_updates = row.get("same_rule_control_update_texts")
    if same_rule_weight > 0 and same_rule_updates:
        same_rule_state = trainer.rollout_state(
            wrapper,
            tokenizer,
            list(same_rule_updates),
            alias,
            max_length,
            device,
        )
        with torch.no_grad():
            _, probe_preferences = trainer.decode_candidate_losses(
                wrapper,
                tokenizer,
                [same_rule_state.detach(), same_rule_state.detach()],
                row["decode_text"],
                [native_target, donor_target],
                alias,
                max_length,
                device,
                preference_reduction,
            )
            same_rule_advantage = float(
                probe_preferences[1] - probe_preferences[0]
            )
        active = float(same_rule_advantage < float(margin))
        losses, preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [same_rule_state, same_rule_state],
            row["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        branch = recipient_control_two_view_loss(
            losses,
            preferences,
            margin=float(margin),
            active=active,
        )
        same_rule_loss_value = float(branch.detach())
        same_rule_margin_value = max(0.0, float(margin) - same_rule_advantage)
        trainer.backward_loss_branch(
            branch, float(weight) * float(same_rule_weight), loss_scale
        )
        del same_rule_state, losses, preferences, branch

    wrapper.train(was_training)
    random_loss_value = sum(random_losses) / max(1, len(random_losses))
    random_margin_value = sum(random_margins) / max(1, len(random_margins))
    return (
        float(random_weight) * random_loss_value
        + float(same_rule_weight) * same_rule_loss_value,
        float(random_weight) * random_margin_value
        + float(same_rule_weight) * same_rule_margin_value,
        {
            "causal_control_random_samples": random_samples,
            "causal_control_extra_random_samples": len(extra_seeds),
            "causal_control_random_weight": float(random_weight),
            "causal_control_random_loss": random_loss_value,
            "causal_control_random_margin": random_margin_value,
            "causal_control_same_rule_weight": float(same_rule_weight),
            "causal_control_same_rule_applied": bool(
                same_rule_weight > 0 and same_rule_updates
            ),
            "causal_control_same_rule_loss": same_rule_loss_value,
            "causal_control_same_rule_margin": same_rule_margin_value,
            "causal_control_seed_stride": 1_000_003,
            "causal_control_random_distribution": (
                "cpu_isotropic_per_layer_slot_norm_matched"
            ),
        },
    )


def backward_random_neutral_specificity(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    preference_reduction: str,
    random_seed: int,
    weight: float,
    loss_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    """Specificity triplet whose random state predicts no stable diagnosis."""
    if not is_diagnosis_pair(row):
        return ORIGINAL_SPECIFICITY(
            wrapper,
            tokenizer,
            row,
            alias,
            max_length,
            device,
            margin,
            preference_reduction,
            random_seed,
            weight,
            loss_scale,
        )
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    native_target = str(row["native_target"])
    donor_target = str(row["donor_target"])
    margin_value = float(margin)
    was_training = wrapper.training
    wrapper.eval()

    with torch.no_grad():
        native_probe, donor_probe = trainer.rollout_state_pair(
            wrapper,
            tokenizer,
            row["native_update_texts"],
            row["donor_update_texts"],
            alias,
            max_length,
            device,
        )
        native_losses, native_preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [native_probe, native_probe],
            row["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        donor_losses, donor_preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [donor_probe, donor_probe],
            row["decode_text"],
            [donor_target, native_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        random_state = trainer.norm_matched_random_state(native_probe, random_seed)
        random_losses, random_preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [random_state, random_state, random_state],
            row["decode_text"],
            [NEUTRAL_TARGET, native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        native_advantage_value = float(native_preferences[1] - native_preferences[0])
        donor_advantage_value = float(donor_preferences[1] - donor_preferences[0])
        random_native_advantage_value = float(
            random_preferences[1] - random_preferences[0]
        )
        random_donor_advantage_value = float(
            random_preferences[2] - random_preferences[0]
        )
        state_cosine_value = float(state_cosine(native_probe, donor_probe))
        state_delta = torch.linalg.vector_norm(
            donor_probe.float() - native_probe.float(), dim=-1
        ).mean()
        state_scale = 0.5 * (
            torch.linalg.vector_norm(native_probe.float(), dim=-1).mean()
            + torch.linalg.vector_norm(donor_probe.float(), dim=-1).mean()
        )
        state_delta_value = float(state_delta)
        state_relative_delta_value = float(
            state_delta / state_scale.clamp_min(1e-12)
        )
    del native_losses, donor_losses, random_losses
    del native_preferences, donor_preferences, random_preferences
    del native_probe, donor_probe, state_delta, state_scale

    native_active = float(native_advantage_value < margin_value)
    donor_active = float(donor_advantage_value < margin_value)
    random_native_active = float(random_native_advantage_value < margin_value)
    random_donor_active = float(random_donor_advantage_value < margin_value)

    native_state = trainer.rollout_state(
        wrapper, tokenizer, row["native_update_texts"], alias, max_length, device
    )
    native_losses, native_preferences = trainer.decode_candidate_losses(
        wrapper,
        tokenizer,
        [native_state, native_state],
        row["decode_text"],
        [native_target, donor_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    native_advantage = native_preferences[1] - native_preferences[0]
    native_margin = native_active * (native_advantage.new_tensor(margin_value) - native_advantage) / 4.0
    native_loss = native_losses[0] / 3.0 + native_margin
    native_loss_value = float(native_loss.detach())
    trainer.backward_loss_branch(native_loss, weight, loss_scale)
    del native_state, native_losses, native_preferences, native_advantage, native_margin, native_loss

    donor_state = trainer.rollout_state(
        wrapper, tokenizer, row["donor_update_texts"], alias, max_length, device
    )
    donor_losses, donor_preferences = trainer.decode_candidate_losses(
        wrapper,
        tokenizer,
        [donor_state, donor_state],
        row["decode_text"],
        [donor_target, native_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    donor_advantage = donor_preferences[1] - donor_preferences[0]
    donor_margin = donor_active * (donor_advantage.new_tensor(margin_value) - donor_advantage) / 4.0
    donor_loss = donor_losses[0] / 3.0 + donor_margin
    donor_loss_value = float(donor_loss.detach())
    trainer.backward_loss_branch(donor_loss, weight, loss_scale)
    del donor_state, donor_losses, donor_preferences, donor_advantage, donor_margin, donor_loss

    random_native_losses, random_native_preferences = trainer.decode_candidate_losses(
        wrapper,
        tokenizer,
        [random_state, random_state],
        row["decode_text"],
        [NEUTRAL_TARGET, native_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    random_native_loss = random_neutral_two_view_branch_loss(
        random_native_losses,
        random_native_preferences,
        margin=margin_value,
        active=random_native_active,
    )
    random_native_loss_value = float(random_native_loss.detach())
    trainer.backward_loss_branch(random_native_loss, weight, loss_scale)
    del random_native_losses, random_native_preferences, random_native_loss

    random_donor_losses, random_donor_preferences = trainer.decode_candidate_losses(
        wrapper,
        tokenizer,
        [random_state, random_state],
        row["decode_text"],
        [NEUTRAL_TARGET, donor_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    random_donor_loss = random_neutral_two_view_branch_loss(
        random_donor_losses,
        random_donor_preferences,
        margin=margin_value,
        active=random_donor_active,
    )
    random_loss_value = random_native_loss_value + float(random_donor_loss.detach())
    trainer.backward_loss_branch(random_donor_loss, weight, loss_scale)
    del random_state, random_donor_losses, random_donor_preferences, random_donor_loss

    full_margin = 0.25 * sum(
        max(0.0, margin_value - value)
        for value in (
            native_advantage_value,
            donor_advantage_value,
            random_native_advantage_value,
            random_donor_advantage_value,
        )
    )
    wrapper.train(was_training)
    return native_loss_value + donor_loss_value + random_loss_value, full_margin, {
        "causal_objective_phase": "semantic_random_neutral_specificity",
        "causal_native_advantage": native_advantage_value,
        "causal_donor_advantage": donor_advantage_value,
        "causal_preference": 0.5 * (native_advantage_value + donor_advantage_value),
        "causal_random_neutral_vs_native_advantage": random_native_advantage_value,
        "causal_random_neutral_vs_donor_advantage": random_donor_advantage_value,
        "causal_random_target": "insufficient_evidence",
        "causal_state_cosine": state_cosine_value,
        "causal_state_delta_norm": state_delta_value,
        "causal_state_relative_delta": state_relative_delta_value,
        "causal_backward_graph_max_views": 2,
        "causal_random_neutral_split": "two_exact_two_view_graphs",
        "causal_dropout_mode": "eval_deterministic",
    }


def backward_updater_only_specificity(
    wrapper,
    tokenizer,
    row: dict[str, Any],
    alias: str,
    max_length: int,
    device: torch.device,
    margin: float,
    preference_reduction: str,
    random_seed: int,
    weight: float,
    loss_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    """Train native/donor state transport while the answer decoder is frozen.

    A norm-matched random state is detached from the updater by construction,
    so its neutral-output branch can only train decoder parameters.  The
    updater-only arm deliberately omits that impossible branch and relies on
    native/donor preference plus same-rule geometry; random/no-op remain
    held-out evaluation controls.
    """
    del random_seed
    if margin < 0:
        raise ValueError("Causal margin must be non-negative")
    native_target = str(row["native_target"])
    donor_target = str(row["donor_target"])
    margin_value = float(margin)
    was_training = wrapper.training
    wrapper.eval()
    with torch.no_grad():
        native_probe, donor_probe = trainer.rollout_state_pair(
            wrapper,
            tokenizer,
            row["native_update_texts"],
            row["donor_update_texts"],
            alias,
            max_length,
            device,
        )
        _, native_preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [native_probe, native_probe],
            row["decode_text"],
            [native_target, donor_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        _, donor_preferences = trainer.decode_candidate_losses(
            wrapper,
            tokenizer,
            [donor_probe, donor_probe],
            row["decode_text"],
            [donor_target, native_target],
            alias,
            max_length,
            device,
            preference_reduction,
        )
        native_advantage_value = float(native_preferences[1] - native_preferences[0])
        donor_advantage_value = float(donor_preferences[1] - donor_preferences[0])
        state_cosine_value = float(state_cosine(native_probe, donor_probe))
        state_delta = torch.linalg.vector_norm(
            donor_probe.float() - native_probe.float(), dim=-1
        ).mean()
        state_scale = 0.5 * (
            torch.linalg.vector_norm(native_probe.float(), dim=-1).mean()
            + torch.linalg.vector_norm(donor_probe.float(), dim=-1).mean()
        )
        state_delta_value = float(state_delta)
        state_relative_delta_value = float(state_delta / state_scale.clamp_min(1e-12))
    del native_probe, donor_probe, native_preferences, donor_preferences
    del state_delta, state_scale

    native_active = float(native_advantage_value < margin_value)
    donor_active = float(donor_advantage_value < margin_value)
    native_state = trainer.rollout_state(
        wrapper, tokenizer, row["native_update_texts"], alias, max_length, device
    )
    native_losses, native_preferences = trainer.decode_candidate_losses(
        wrapper,
        tokenizer,
        [native_state, native_state],
        row["decode_text"],
        [native_target, donor_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    native_advantage = native_preferences[1] - native_preferences[0]
    native_margin = native_active * (
        native_advantage.new_tensor(margin_value) - native_advantage
    ) / 2.0
    native_loss = native_losses[0] + native_margin
    native_loss_value = float(native_loss.detach())
    trainer.backward_loss_branch(native_loss, 0.5 * weight, loss_scale)
    del native_state, native_losses, native_preferences, native_advantage, native_margin, native_loss

    donor_state = trainer.rollout_state(
        wrapper, tokenizer, row["donor_update_texts"], alias, max_length, device
    )
    donor_losses, donor_preferences = trainer.decode_candidate_losses(
        wrapper,
        tokenizer,
        [donor_state, donor_state],
        row["decode_text"],
        [donor_target, native_target],
        alias,
        max_length,
        device,
        preference_reduction,
    )
    donor_advantage = donor_preferences[1] - donor_preferences[0]
    donor_margin = donor_active * (
        donor_advantage.new_tensor(margin_value) - donor_advantage
    ) / 2.0
    donor_loss = donor_losses[0] + donor_margin
    donor_loss_value = float(donor_loss.detach())
    trainer.backward_loss_branch(donor_loss, 0.5 * weight, loss_scale)
    del donor_state, donor_losses, donor_preferences, donor_advantage, donor_margin, donor_loss
    wrapper.train(was_training)
    margin_loss = 0.5 * (
        max(0.0, margin_value - native_advantage_value)
        + max(0.0, margin_value - donor_advantage_value)
    )
    return native_loss_value + donor_loss_value, margin_loss, {
        "causal_objective_phase": "updater_only_native_donor",
        "causal_native_advantage": native_advantage_value,
        "causal_donor_advantage": donor_advantage_value,
        "causal_preference": 0.5 * (native_advantage_value + donor_advantage_value),
        "causal_random_branch_trainable": False,
        "causal_random_branch_scope": "held_out_evaluation_only",
        "causal_state_cosine": state_cosine_value,
        "causal_state_delta_norm": state_delta_value,
        "causal_state_relative_delta": state_relative_delta_value,
        "causal_dropout_mode": "eval_deterministic",
    }


ORIGINAL_SPECIFICITY = trainer.backward_specificity_triplet_objective


def main() -> None:
    label_weight = float(os.environ.get("STATE8_CAUSAL_LABEL_WEIGHT", "1"))
    random_neutral = env_flag("STATE8_RANDOM_NEUTRAL", False)
    updater_only = env_flag("STATE8_UPDATER_ONLY", False)
    geometry_weight = float(os.environ.get("STATE8_GEOMETRY_WEIGHT", "0"))
    eedi_geometry_weight = float(
        os.environ.get("STATE8_EEDI_GEOMETRY_WEIGHT", "0")
    )
    different_max_cosine = float(
        os.environ.get("STATE8_DIFFERENT_MAX_COSINE", "0.90")
    )
    control_random_samples = int(
        os.environ.get("STATE8_CONTROL_RANDOM_SAMPLES", "1")
    )
    control_random_weight = float(
        os.environ.get("STATE8_CONTROL_RANDOM_WEIGHT", "0")
    )
    control_same_rule_weight = float(
        os.environ.get("STATE8_CONTROL_SAME_RULE_WEIGHT", "0")
    )
    early_label_binding_weight = float(
        os.environ.get("STATE8_EARLY_LABEL_BINDING_WEIGHT", "0")
    )
    early_label_random_weight = float(
        os.environ.get("STATE8_EARLY_LABEL_RANDOM_WEIGHT", "0")
    )
    early_label_same_rule_weight = float(
        os.environ.get("STATE8_EARLY_LABEL_SAME_RULE_WEIGHT", "0")
    )
    label_query_mode = early_label_query_mode()
    if geometry_weight < 0 or eedi_geometry_weight < 0:
        raise ValueError("State geometry weights must be non-negative")
    if control_random_samples <= 0:
        raise ValueError("STATE8_CONTROL_RANDOM_SAMPLES must be positive")
    if control_random_weight < 0 or control_same_rule_weight < 0:
        raise ValueError("Control-robust weights must be non-negative")
    if early_label_binding_weight < 0:
        raise ValueError("STATE8_EARLY_LABEL_BINDING_WEIGHT must be non-negative")
    if early_label_random_weight < 0 or early_label_same_rule_weight < 0:
        raise ValueError("Early-label control weights must be non-negative")
    if (
        early_label_random_weight > 0 or early_label_same_rule_weight > 0
    ) and early_label_binding_weight <= 0:
        raise ValueError("Early-label controls require positive label binding")
    control_robust = (
        control_random_samples > 1
        or control_random_weight > 0
        or control_same_rule_weight > 0
    )
    if control_robust and updater_only:
        raise ValueError("Control-robust decoding requires the full training arm")
    if control_random_weight > 0 and control_random_samples < 2:
        raise ValueError(
            "Positive control-random weight requires at least two random samples"
        )
    if not natural_schema_valid(NEUTRAL_TARGET):
        raise AssertionError("Random-neutral target is not valid natural schema")

    tokenizer = AutoTokenizer.from_pretrained(
        cli_value("--model"), trust_remote_code=True
    )
    weighted_per_sequence = schema_weighted_per_sequence_factory(
        tokenizer, label_weight
    )
    output_dir = Path(cli_value("--output-dir"))

    original_parse = argparse.ArgumentParser.parse_args
    original_per_sequence = trainer.causal_lm_loss_per_sequence
    original_specificity = trainer.backward_specificity_triplet_objective
    original_objective = trainer.backward_training_objective
    geometry_counts = {"eligible": 0, "applied": 0}
    eedi_geometry_counts = {"eligible": 0, "applied": 0}

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.semantic_causal_label_weight = label_weight
        namespace.semantic_random_neutral = random_neutral
        namespace.semantic_updater_only = updater_only
        namespace.semantic_geometry_weight = geometry_weight
        namespace.eedi_semantic_geometry_weight = eedi_geometry_weight
        namespace.semantic_different_max_cosine = different_max_cosine
        namespace.semantic_auxiliary_visible_at_inference = False
        namespace.semantic_random_target = (
            "held_out_evaluation_only"
            if updater_only
            else ("insufficient_evidence" if random_neutral else "legacy_recipient")
        )
        namespace.control_robust_objective = control_robust
        namespace.control_random_samples = control_random_samples
        namespace.control_random_weight = control_random_weight
        namespace.control_same_rule_weight = control_same_rule_weight
        namespace.control_random_seed_stride = 1_000_003
        namespace.early_label_binding_weight = early_label_binding_weight
        namespace.early_label_binding_scope = (
            "diagnosis_causal_pairs_train_only"
            if early_label_binding_weight > 0
            else "disabled"
        )
        namespace.early_label_binding_visible_at_inference = False
        namespace.early_label_query_mode = label_query_mode
        namespace.early_label_random_weight = early_label_random_weight
        namespace.early_label_same_rule_weight = early_label_same_rule_weight
        namespace.early_label_binding_metric_role = (
            "auxiliary_supervision_not_authoritative_evaluation"
        )
        return namespace

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
        record = original_objective(
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
        )
        record["semantic_geometry_loss"] = 0.0
        record["eedi_semantic_geometry_loss"] = 0.0
        record["early_label_binding_loss"] = 0.0
        record["early_label_control_loss"] = 0.0
        rendered_alias = getattr(
            namespace, "state_token_rendered_alias", namespace.state_token_alias
        )
        if (
            geometry_weight > 0
            and causal_row is not None
            and is_diagnosis_pair(causal_row)
            and causal_row.get("native_control_role") == "recipient_native"
        ):
            geometry_counts["eligible"] += 1
            required = (
                "recipient_native_update_texts",
                "same_rule_control_update_texts",
                "donor_update_texts",
            )
            missing = [key for key in required if not causal_row.get(key)]
            if missing:
                raise ValueError(f"Semantic geometry row lacks {missing}")
            effective_weight = geometry_weight * float(
                namespace.causal_exposure_multiplier
            )
            value, diagnostics = backward_semantic_state_geometry(
                wrapper,
                tokenizer,
                causal_row,
                rendered_alias,
                namespace.max_length,
                device,
                different_max_cosine=different_max_cosine,
                weight=effective_weight,
                loss_scale=loss_scale,
            )
            geometry_counts["applied"] += 1
            record["semantic_geometry_loss"] = value
            record["semantic_geometry_effective_weight"] = effective_weight
            record["loss"] += geometry_weight * value
            record.update(diagnostics)
        if eedi_geometry_weight > 0 and is_eedi_semantic_pair(row):
            eedi_geometry_counts["eligible"] += 1
            required = (
                "eedi_anchor_update_texts",
                "eedi_positive_update_texts",
                "eedi_negative_update_texts",
            )
            missing = [key for key in required if not row.get(key)]
            if missing:
                raise ValueError(f"EEDI semantic geometry row lacks {missing}")
            if str(row["eedi_anchor_label"]) != str(row["eedi_positive_label"]):
                raise ValueError("EEDI positive-label drift")
            if str(row["eedi_anchor_label"]) == str(row["eedi_negative_label"]):
                raise ValueError("EEDI negative-label drift")
            geometry_row = {
                "recipient_native_update_texts": row["eedi_anchor_update_texts"],
                "same_rule_control_update_texts": row["eedi_positive_update_texts"],
                "donor_update_texts": row["eedi_negative_update_texts"],
            }
            value, diagnostics = backward_semantic_state_geometry(
                wrapper,
                tokenizer,
                geometry_row,
                rendered_alias,
                namespace.max_length,
                device,
                different_max_cosine=different_max_cosine,
                weight=eedi_geometry_weight,
                loss_scale=loss_scale,
            )
            eedi_geometry_counts["applied"] += 1
            record["eedi_semantic_geometry_loss"] = value
            record["eedi_semantic_geometry_effective_weight"] = eedi_geometry_weight
            record["loss"] += eedi_geometry_weight * value
            record.update(
                {f"eedi_{key}": metric for key, metric in diagnostics.items()}
            )
        if early_label_binding_weight > 0 and is_early_label_pair(causal_row):
            effective_weight = early_label_binding_weight * float(
                namespace.causal_exposure_multiplier
            )
            value, margin_value, diagnostics = backward_early_label_binding(
                wrapper,
                tokenizer,
                causal_row,
                rendered_alias,
                namespace.max_length,
                device,
                namespace.causal_margin,
                effective_weight,
                loss_scale,
            )
            record["early_label_binding_loss"] = value
            record["early_label_binding_margin_loss"] = margin_value
            record["early_label_binding_effective_weight"] = effective_weight
            record["early_label_binding_source"] = str(
                causal_row.get("source") or "unknown"
            )
            record["loss"] += effective_weight * value
            record.update(diagnostics)
            control_value, control_diagnostics = backward_early_label_controls(
                wrapper,
                tokenizer,
                causal_row,
                rendered_alias,
                namespace.max_length,
                device,
                namespace.causal_margin,
                int(namespace.seed) + int(record.get("global_position", 0)),
                effective_weight,
                loss_scale,
                random_weight=early_label_random_weight,
                same_rule_weight=early_label_same_rule_weight,
            )
            record["early_label_control_loss"] = control_value
            record["loss"] += effective_weight * control_value
            record.update(control_diagnostics)
        return record

    argparse.ArgumentParser.parse_args = parse
    trainer.causal_lm_loss_per_sequence = weighted_per_sequence
    base_specificity = original_specificity
    if updater_only:
        base_specificity = backward_updater_only_specificity
    elif random_neutral:
        base_specificity = backward_random_neutral_specificity

    if control_robust:
        def control_robust_specificity(
            wrapper,
            tokenizer,
            row,
            alias,
            max_length,
            device,
            margin,
            preference_reduction,
            random_seed,
            weight,
            loss_scale,
        ):
            base_loss, base_margin, diagnostics = base_specificity(
                wrapper,
                tokenizer,
                row,
                alias,
                max_length,
                device,
                margin,
                preference_reduction,
                random_seed,
                weight,
                loss_scale,
            )
            extra_loss, extra_margin, extra_diagnostics = (
                backward_control_robust_extras(
                    wrapper,
                    tokenizer,
                    row,
                    alias,
                    max_length,
                    device,
                    margin,
                    preference_reduction,
                    random_seed,
                    weight,
                    loss_scale,
                    random_samples=control_random_samples,
                    random_weight=control_random_weight,
                    same_rule_weight=control_same_rule_weight,
                    diagnosis_random_neutral=random_neutral,
                )
            )
            diagnostics.update(extra_diagnostics)
            diagnostics["causal_objective_phase"] = "control_robust_specificity"
            return (
                float(base_loss) + float(extra_loss),
                float(base_margin) + float(extra_margin),
                diagnostics,
            )

        trainer.backward_specificity_triplet_objective = control_robust_specificity
    else:
        trainer.backward_specificity_triplet_objective = base_specificity
    trainer.backward_training_objective = objective
    try:
        repair.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            history_path = output_dir / "trainer_history.json"
            if not history_path.is_file():
                raise FileNotFoundError(
                    f"Global trainer history is required for semantic audit: {history_path}"
                )
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                raise ValueError("Global trainer history must be a JSON list")
            geometry_audit, eedi_geometry_audit = aggregate_geometry_counts(
                history,
                geometry_weight=geometry_weight,
                eedi_geometry_weight=eedi_geometry_weight,
            )
            early_label_audit = aggregate_early_label_counts(
                history, early_label_binding_weight
            )
            receipt = {
                "stage": "state8_semantic_causal_repair_v1",
                "causal_label_weight": label_weight,
                "random_neutral": random_neutral,
                "updater_only": updater_only,
                "random_target": (
                    "held_out_evaluation_only"
                    if updater_only
                    else ("insufficient_evidence" if random_neutral else "legacy_recipient")
                ),
                "geometry_weight": geometry_weight,
                "eedi_geometry_weight": eedi_geometry_weight,
                "different_max_cosine": different_max_cosine,
                "control_robust_objective": control_robust,
                "control_random_samples": control_random_samples,
                "control_random_weight": control_random_weight,
                "control_same_rule_weight": control_same_rule_weight,
                "control_random_seed_stride": 1_000_003,
                "control_random_distribution": (
                    "cpu_isotropic_per_layer_slot_norm_matched"
                ),
                "early_label_binding_weight": early_label_binding_weight,
                "early_label_binding_scope": (
                    "diagnosis_causal_pairs_train_only"
                    if early_label_binding_weight > 0
                    else "disabled"
                ),
                "early_label_binding_target": "identifier_only_no_gold_prefix",
                "early_label_query_mode": label_query_mode,
                "early_label_current_query_visible": label_query_mode == "with_query",
                "early_label_random_weight": early_label_random_weight,
                "early_label_same_rule_weight": early_label_same_rule_weight,
                "early_label_binding": early_label_audit,
                "early_label_binding_visible_at_inference": False,
                "early_label_binding_metric_role": (
                    "auxiliary_supervision_not_authoritative_evaluation"
                ),
                "same_rule_geometry": geometry_weight > 0,
                "auxiliary_visible_at_inference": False,
                "final_inference_route": "strict_state_only",
                "geometry_counts": geometry_audit,
                "eedi_geometry_counts": eedi_geometry_audit,
                "process_local_geometry_counts": geometry_counts,
                "process_local_eedi_geometry_counts": eedi_geometry_counts,
                "eedi_pair_supervision_visible_at_inference": False,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "semantic_causal_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        trainer.causal_lm_loss_per_sequence = original_per_sequence
        trainer.backward_specificity_triplet_objective = original_specificity
        trainer.backward_training_objective = original_objective


if __name__ == "__main__":
    main()
