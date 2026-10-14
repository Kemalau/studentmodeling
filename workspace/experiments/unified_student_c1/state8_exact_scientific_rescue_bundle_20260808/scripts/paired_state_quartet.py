#!/usr/bin/env python3
"""Pure construction helpers for causally identified state-swap quartets."""

from __future__ import annotations

import copy
from typing import Any

import torch


STATE_TOKEN = "<STUDENT_STATE>"


def patch_state_residual(
    hidden: torch.Tensor,
    state_positions: torch.Tensor,
    source_indices: torch.Tensor,
    destination_indices: torch.Tensor,
) -> torch.Tensor:
    """Copy matched source state rows without touching any other residual row."""
    if source_indices.numel() == 0:
        return hidden
    positions = state_positions.to(hidden.device)
    sources = source_indices.to(hidden.device)
    destinations = destination_indices.to(hidden.device)
    source_vectors = hidden[sources, positions[sources], :]
    patched = hidden.clone()
    patched[destinations, positions[destinations], :] = source_vectors
    return patched


def make_counterfactual_source(
    donor: dict[str, Any], recipient: dict[str, Any], donor_target: str, pair_id: int
) -> dict[str, Any]:
    row = copy.deepcopy(donor)
    row["example_id"] = f"paired-state-swap:source:{pair_id:07d}"
    row["query"] = copy.deepcopy(recipient["query"])
    row["candidate_labels"] = copy.deepcopy(recipient["candidate_labels"])
    row["candidate_targets"] = copy.deepcopy(recipient["candidate_targets"])
    row["target_label"] = donor_target
    row["target"] = row["candidate_targets"][donor_target]
    row["seg_question"] = recipient["seg_question"]
    row["prompt"] = f"{row['seg_instruction']}{row['seg_history']}{STATE_TOKEN}\n{row['seg_question']}"
    return row


def make_swap_destination(
    recipient: dict[str, Any], donor_target: str, pair_id: int
) -> dict[str, Any]:
    row = copy.deepcopy(recipient)
    row["example_id"] = f"paired-state-swap:destination:{pair_id:07d}"
    row["target_label"] = donor_target
    row["target"] = row["candidate_targets"][donor_target]
    return row


def make_native_recipient(recipient: dict[str, Any], pair_id: int) -> dict[str, Any]:
    row = copy.deepcopy(recipient)
    row["example_id"] = f"paired-state-swap:native-recipient:{pair_id:07d}"
    return row


def build_quartet(
    donor: dict[str, Any],
    recipient: dict[str, Any],
    normal: dict[str, Any],
    donor_target: str,
    pair_id: int,
    arm: str,
) -> tuple[tuple[dict[str, Any], ...], list[tuple[int, int]], dict[str, str]]:
    """Return matched rows, patch pairs, and explicit A/B targets.

    Index 2 and index 3 have identical prompts in the treatment arm.  Their
    different targets are therefore identifiable only through the patched
    state at index 3.
    """
    source = make_counterfactual_source(donor, recipient, donor_target, pair_id)
    native = make_native_recipient(recipient, pair_id)
    native_target = str(native["target_label"])
    if donor_target == native_target:
        raise ValueError(f"Pair {pair_id} does not define an A/B contrast")
    if source["seg_question"] != native["seg_question"]:
        raise ValueError(f"Pair {pair_id} does not share a recipient question")

    if arm == "paired_swap":
        counterfactual = make_swap_destination(recipient, donor_target, pair_id)
        swap_pairs = [(1, 3)]
        if native["prompt"] != counterfactual["prompt"]:
            raise AssertionError(f"Treatment prompts differ for pair {pair_id}")
    elif arm == "control":
        counterfactual = copy.deepcopy(source)
        counterfactual["example_id"] = f"paired-state-swap:control-duplicate:{pair_id:07d}"
        swap_pairs = []
    else:
        raise ValueError(f"Unknown arm={arm!r}")

    return (
        (copy.deepcopy(normal), source, native, counterfactual),
        swap_pairs,
        {"donor_on_recipient": donor_target, "native_recipient": native_target},
    )
