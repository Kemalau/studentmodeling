"""Checkpoint cadence helpers for preemptible promoted exact-epoch fits."""

from __future__ import annotations


def should_save_distributed_checkpoint(
    completed_round: int,
    *,
    start_round: int,
    total_rounds: int,
    save_steps: int,
    recovery_first_offset: int = 0,
    recovery_interval: int = 0,
) -> bool:
    """Return whether an exact-resume checkpoint is due.

    The normal cadence remains authoritative. A recovered attempt may also
    write one early checkpoint and sparse follow-ups so a short-lived cluster
    allocation does not lose the entire segment between normal checkpoints.
    These writes preserve the optimizer trajectory and change only recovery
    plumbing.
    """
    if completed_round <= start_round or completed_round > total_rounds:
        return False
    if completed_round == total_rounds or completed_round % save_steps == 0:
        return True
    if recovery_first_offset <= 0:
        return False
    first_round = start_round + recovery_first_offset
    if completed_round == first_round:
        return True
    return (
        recovery_interval > 0
        and completed_round > first_round
        and (completed_round - first_round) % recovery_interval == 0
    )
