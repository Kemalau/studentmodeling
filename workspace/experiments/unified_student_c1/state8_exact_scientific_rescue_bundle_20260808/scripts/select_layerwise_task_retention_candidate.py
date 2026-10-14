#!/usr/bin/env python3
"""Select a fully gated task-retention recipe for exact-epoch promotion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from validate_checkpoint_t3_semantic_result import (
    FORMAL_PANEL_SUPPORT,
    MIN_GROUP_RETENTION_RATIO,
    REFERENCE_GROUP_ACCURACY,
)
from validate_t3_semantic_result import MIN_RETENTION_RATIO, REFERENCE_SEMANTIC_ACCURACY


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def prevalence_corrected_t3(value: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy balanced checkpoint evidence to formal-panel weighting."""

    if value.get("weighting_scheme") == "formal_panel_group_prevalence_v1":
        if (
            int(value.get("formal_panel_support", -1)) != 1494
            or type(value.get("group_floor_pass")) is not bool
            or type(value.get("pass")) is not bool
        ):
            raise ValueError("Malformed prevalence-corrected T3 evidence")
        return value
    groups = value.get("by_group") or {}
    if set(groups) != set(FORMAL_PANEL_SUPPORT):
        raise ValueError("Legacy T3 evidence lacks the frozen three-group panel")
    weighted_correct = 0.0
    upgraded_groups: dict[str, Any] = {}
    for name, formal_support in FORMAL_PANEL_SUPPORT.items():
        source = groups[name]
        accuracy = float(source.get("semantic_accuracy", float("nan")))
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"Invalid legacy T3 group accuracy: {name}")
        group_ratio = accuracy / REFERENCE_GROUP_ACCURACY[name]
        upgraded_groups[name] = {
            **source,
            "formal_panel_support": formal_support,
            "reference_semantic_accuracy": REFERENCE_GROUP_ACCURACY[name],
            "retention_ratio": group_ratio,
            "minimum_retention_ratio": MIN_GROUP_RETENTION_RATIO,
            "retention_pass": group_ratio >= MIN_GROUP_RETENTION_RATIO,
        }
        weighted_correct += accuracy * formal_support
    semantic_accuracy = weighted_correct / sum(FORMAL_PANEL_SUPPORT.values())
    retention_ratio = semantic_accuracy / REFERENCE_SEMANTIC_ACCURACY
    group_floor_pass = all(row["retention_pass"] for row in upgraded_groups.values())
    return {
        **value,
        "pass": retention_ratio >= MIN_RETENTION_RATIO and group_floor_pass,
        "semantic_accuracy": semantic_accuracy,
        "reference_semantic_accuracy": REFERENCE_SEMANTIC_ACCURACY,
        "retention_ratio": retention_ratio,
        "minimum_retention_ratio": MIN_RETENTION_RATIO,
        "weighting_scheme": "formal_panel_group_prevalence_v1",
        "formal_panel_support": sum(FORMAL_PANEL_SUPPORT.values()),
        "minimum_group_retention_ratio": MIN_GROUP_RETENTION_RATIO,
        "group_floor_pass": group_floor_pass,
        "by_group": upgraded_groups,
    }


def normalize(
    row: dict[str, Any],
    *,
    branch: str,
    loss_profile: str,
    learning_rate: float = 5e-7,
) -> dict[str, Any]:
    row = dict(row)
    original_checks = row.get("checks") or {}
    if not original_checks or any(
        type(value) is not bool for value in original_checks.values()
    ):
        raise ValueError(f"Candidate has malformed checks: {branch}/{loss_profile}")
    if bool(row.get("all_gates_pass")) != all(original_checks.values()):
        raise ValueError(f"Candidate gate summary is inconsistent: {branch}/{loss_profile}")
    t3 = prevalence_corrected_t3(row.get("T3_semantic") or {})
    row["T3_semantic"] = t3
    checks = dict(original_checks)
    checks["T3_semantic"] = bool(t3["pass"])
    row["checks"] = checks
    row["all_gates_pass"] = all(checks.values())
    if not checks or any(type(value) is not bool for value in checks.values()):
        raise ValueError(f"Candidate has malformed checks: {branch}/{loss_profile}")
    observed_pass = bool(row.get("all_gates_pass"))
    computed_pass = all(checks.values())
    if observed_pass != computed_pass:
        raise ValueError(f"Candidate gate summary is inconsistent: {branch}/{loss_profile}")
    observed_lr = float(row.get("learning_rate", learning_rate))
    if abs(observed_lr - learning_rate) >= 1e-15:
        raise ValueError(f"Candidate learning-rate drift: {branch}/{loss_profile}")
    lr_scanned_branches = {
        "eedi_full",
        "eedi_full_open_t3",
        "equal_t3_t6_open_t3",
    }
    candidate_id = (
        f"{branch}__{loss_profile}"
        if branch not in lr_scanned_branches or learning_rate == 5e-7
        else f"{branch}__{loss_profile}__lr{learning_rate:.0e}"
    )
    if row.get("candidate_id") not in {None, candidate_id}:
        raise ValueError(f"Candidate ID drift: {branch}/{loss_profile}")
    causal_weight = float(row.get("causal_weight", -1.0))
    if causal_weight != 2.0 or checks.get("causal_recipe") is not True:
        raise ValueError(f"Candidate causal recipe drift: {branch}/{loss_profile}")
    t3 = row["T3_semantic"]
    if (
        t3.get("weighting_scheme") != "formal_panel_group_prevalence_v1"
        or int(t3.get("formal_panel_support", -1)) != 1494
        or type(t3.get("group_floor_pass")) is not bool
        or type(t3.get("pass")) is not bool
    ):
        raise ValueError(
            f"Candidate lacks prevalence-corrected T3 evidence: {branch}/{loss_profile}"
        )
    if observed_pass and (t3["pass"] is not True or t3["group_floor_pass"] is not True):
        raise ValueError(
            f"Passing candidate fails corrected T3 gate: {branch}/{loss_profile}"
        )
    return {
        **row,
        "candidate_id": candidate_id,
        "branch": branch,
        "loss_profile": loss_profile,
        "learning_rate": learning_rate,
        "causal_weight": causal_weight,
    }


def normalize_consolidation_rows(path: Path) -> list[dict[str, Any]]:
    """Load both fully audited two-stage arms behind a completed selection.

    The compact selection artifact intentionally contains only ranking fields;
    promotion needs the complete C4/T3/T6 evidence retained in the sibling arm
    summaries.  Requiring both files and the parent ``complete`` marker keeps a
    partially downloaded or half-judged consolidation run out of selection.
    """

    selection = read(path)
    if selection.get("stage") != "layerwise_open_t3_consolidation_selection_v1":
        raise ValueError("Open-T3 consolidation selection stage drift")
    rows: list[dict[str, Any]] = []
    for profile in ("state_kd", "state_path"):
        summary_path = path.parent / f"{profile}.summary.json"
        if not summary_path.is_file():
            raise ValueError(f"Open-T3 consolidation summary missing: {profile}")
        row = read(summary_path)
        stage = row.get("stage")
        if stage == "layerwise_single_task_retention_candidate_pruned_v1":
            # A full frozen C4 gate can terminate an arm before the more
            # expensive T3/T6 judgments.  That is an authoritative completed
            # diagnostic, not a malformed candidate.  Keep it in the census
            # (so completion accounting remains exact), but make it
            # structurally impossible to promote.
            expected_source_id = f"equal_t3_t6_open_t3__{profile}__lr5e-07"
            checks = row.get("checks") or {}
            if (
                row.get("loss_profile") != profile
                or row.get("candidate_id") != expected_source_id
                or float(row.get("learning_rate", -1.0)) != 5e-7
                or row.get("pruned") is not True
                or row.get("all_gates_pass") is not False
                or row.get("prune_reason")
                != "frozen_full_c4_specificity_gate_failed"
                or checks.get("C4") is not False
                or checks.get("specificity") is not False
            ):
                raise ValueError(
                    f"Open-T3 consolidation pruned candidate drift: {profile}"
                )
            rows.append(
                {
                    **row,
                    "candidate_id": (
                        f"equal_t3_t6_open_t3_two_stage__{profile}"
                    ),
                    "branch": "equal_t3_t6_open_t3_two_stage",
                    "loss_profile": profile,
                    "learning_rate": 5e-7,
                    "promotion_eligible": False,
                }
            )
            continue
        if stage != "layerwise_single_task_retention_candidate":
            raise ValueError(f"Open-T3 consolidation candidate stage drift: {profile}")
        if row.get("loss_profile") != profile:
            raise ValueError(f"Open-T3 consolidation profile drift: {profile}")
        source = dict(row)
        # The diagnostic summary uses the single-stage branch ID.  Promotion
        # gives this scientifically distinct curriculum its own immutable ID.
        source.pop("candidate_id", None)
        rows.append(
            normalize(
                source,
                branch="equal_t3_t6_open_t3_two_stage",
                loss_profile=profile,
                learning_rate=5e-7,
            )
        )
    passing = [row for row in rows if row["all_gates_pass"]]
    if bool(passing) != bool(selection.get("all_gates_pass")):
        raise ValueError("Open-T3 consolidation pass state is inconsistent")
    selected_id = selection.get("selected_candidate_id")
    if passing and selected_id not in {
        f"equal_t3_t6_open_t3__{row['loss_profile']}" for row in passing
    }:
        raise ValueError("Open-T3 consolidation selected arm drift")
    for row in rows:
        row["promotion_eligible"] = (
            f"equal_t3_t6_open_t3__{row['loss_profile']}" == selected_id
        )
    return rows


def apply_t6_calibration_overlay(
    candidates: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    """Replace exactly one hash-bound diagnostic row with calibrated evidence."""

    overlay = read(path)
    if overlay.get("stage") != "layerwise_t6_calibrated_candidate_v1":
        raise ValueError("T6 calibration overlay stage drift")
    candidate_id = str(overlay.get("candidate_id"))
    matches = [index for index, row in enumerate(candidates) if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"T6 calibration candidate match drift: {matches}")
    index = matches[0]
    source = candidates[index]
    if (
        source.get("branch") != overlay.get("branch")
        or source.get("loss_profile") != overlay.get("loss_profile")
        or float(source.get("learning_rate", -1))
        != float(overlay.get("learning_rate", -2))
        or float(source.get("causal_weight", -1)) != 2.0
    ):
        raise ValueError("T6 calibration overlay identity drift")
    # Calibration artifacts created before the formal-prevalence T3 repair
    # retain the original balanced 25/25/25 checkpoint record.  The source
    # candidate has already passed through ``normalize`` above, so compare the
    # same canonical representation on both sides.  This changes no observed
    # scores; it only prevents a stale serialization shape from bypassing or
    # blocking the corrected Eedi-heavy selection rule.
    overlay = dict(overlay)
    overlay["T3_semantic"] = prevalence_corrected_t3(
        overlay.get("T3_semantic") or {}
    )
    import hashlib

    expected = hashlib.sha256(
        (json.dumps(source, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    # The source summary hash is computed over the on-disk arm before normalize
    # injects immutable identity fields.  Require the canonical scientific
    # subrecords to remain equivalent instead of relying only on candidate ID.
    for field in ("C4", "T3_semantic", "formal_screen"):
        if source.get(field) != overlay.get(field):
            raise ValueError(f"T6 calibration source evidence drift: {field}")
    normalized = normalize(
        overlay,
        branch=str(source["branch"]),
        loss_profile=str(source["loss_profile"]),
        learning_rate=float(source["learning_rate"]),
    )
    normalized["normalized_source_candidate_sha256"] = expected
    output = list(candidates)
    output[index] = normalized
    return output


def choose(candidates: list[dict[str, Any]], *, total_candidate_count: int) -> dict[str, Any]:
    """Select only from fully materialized, internally consistent candidates."""

    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")
    eligible = [
        row
        for row in candidates
        if row["all_gates_pass"] and row.get("promotion_eligible", True)
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["T3_semantic"]["retention_ratio"]),
            float(row["T6"]["retention_ratio"]),
            float(row["C4"]["D_specific"]),
        ),
    ) if eligible else None
    return {
        "stage": "layerwise_task_retention_unified_candidate_selection",
        "diagnostic_only": True,
        "selection_scope": "completed_candidates_only",
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "all_diagnostics_complete": len(candidates) == total_candidate_count,
        "candidates": candidates,
        "any_all_gates_pass": bool(eligible),
        "selected_candidate": (
            {
                "all_gates_pass": True,
                "candidate_id": selected["candidate_id"],
                "branch": selected["branch"],
                "loss_profile": selected["loss_profile"],
                "learning_rate": selected["learning_rate"],
                "causal_weight": selected["causal_weight"],
                "T3_retention_ratio": selected["T3_semantic"]["retention_ratio"],
                "T6_retention_ratio": selected["T6"]["retention_ratio"],
                "C4_D_specific": selected["C4"]["D_specific"],
                **(
                    {"t6_calibration": selected["t6_calibration"]}
                    if selected.get("t6_calibration") is not None
                    else {}
                ),
            }
            if selected
            else None
        ),
    }


def select(
    rehearsal: dict[str, Any],
    eedi: dict[str, Any],
    state_kd: dict[str, Any],
    staged: dict[str, Any],
    staged_state_path: dict[str, Any],
    equal_state_path: dict[str, Any],
    eedi_full: dict[str, Any],
    eedi_full_state_kd: dict[str, Any],
    eedi_full_open_t3: dict[str, Any] | None = None,
    eedi_full_open_t3_selective_kd: dict[str, Any] | None = None,
    equal_state_kd: dict[str, Any] | None = None,
    equal_state_kd_open_t3: dict[str, Any] | None = None,
    equal_state_path_open_t3: dict[str, Any] | None = None,
    equal_state_kd_open_t3_lr_scan: dict[str, Any] | None = None,
    equal_state_kd_open_t3_lr1e6: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if rehearsal.get("stage") != "layerwise_task_retention_rehearsal_comparison":
        raise ValueError("Rehearsal comparison stage drift")
    if eedi.get("stage") != "layerwise_eedi_state_path_ab_comparison":
        raise ValueError("Eedi comparison stage drift")
    if state_kd.get("stage") != "layerwise_single_task_retention_candidate":
        raise ValueError("State-KD candidate stage drift")
    if staged.get("stage") != "layerwise_single_task_retention_candidate":
        raise ValueError("Staged candidate stage drift")
    if equal_state_path.get("stage") != "layerwise_single_task_retention_candidate":
        raise ValueError("Equal state-path candidate stage drift")
    if staged_state_path.get("stage") != "layerwise_single_task_retention_candidate":
        raise ValueError("Staged state-path candidate stage drift")
    if eedi_full.get("stage") != "layerwise_eedi_full_lr_scan_comparison":
        raise ValueError("Eedi full LR-scan stage drift")
    if eedi_full_state_kd.get("stage") != "layerwise_eedi_full_lr_scan_comparison":
        raise ValueError("Eedi full state-KD LR-scan stage drift")
    if (
        eedi_full_open_t3 is not None
        and eedi_full_open_t3.get("stage") != "layerwise_eedi_full_lr_scan_comparison"
    ):
        raise ValueError("Eedi full open-T3 LR-scan stage drift")
    if (
        eedi_full_open_t3_selective_kd is not None
        and eedi_full_open_t3_selective_kd.get("stage")
        != "layerwise_eedi_full_lr_scan_comparison"
    ):
        raise ValueError("Eedi full open-T3 selective-KD LR-scan stage drift")
    if (
        equal_state_kd is not None
        and equal_state_kd.get("stage") != "layerwise_single_task_retention_candidate"
    ):
        raise ValueError("Equal state-KD candidate stage drift")
    if (
        equal_state_kd_open_t3 is not None
        and equal_state_kd_open_t3.get("stage")
        != "layerwise_single_task_retention_candidate"
    ):
        raise ValueError("Equal open-T3 state-KD candidate stage drift")
    if (
        equal_state_path_open_t3 is not None
        and equal_state_path_open_t3.get("stage")
        != "layerwise_single_task_retention_candidate"
    ):
        raise ValueError("Equal open-T3 state-path candidate stage drift")
    if (
        equal_state_kd_open_t3_lr_scan is not None
        and equal_state_kd_open_t3_lr_scan.get("stage")
        != "layerwise_eedi_full_lr_scan_comparison"
    ):
        raise ValueError("Equal open-T3 state-KD LR-scan stage drift")
    if (
        equal_state_kd_open_t3_lr1e6 is not None
        and equal_state_kd_open_t3_lr1e6.get("stage")
        != "layerwise_single_task_retention_candidate"
    ):
        raise ValueError("Equal open-T3 state-KD lr1e-6 stage drift")
    candidates = []
    base_rows = {str(row.get("branch")): row for row in rehearsal.get("arms") or []}
    if set(base_rows) != {"equal_t3_t6", "t3_heavy"}:
        raise ValueError("Base rehearsal candidate set drift")
    for branch, row in sorted(base_rows.items()):
        candidates.append(normalize(row, branch=branch, loss_profile="standard"))
    eedi_rows = {
        str(row.get("loss_profile")): row for row in eedi.get("profiles") or []
    }
    if set(eedi_rows) != {"standard", "state_path"}:
        raise ValueError("Eedi objective candidate set drift")
    for profile, row in sorted(eedi_rows.items()):
        candidates.append(normalize(row, branch="eedi_dense", loss_profile=profile))
    candidates.append(normalize(state_kd, branch="eedi_dense", loss_profile="state_kd"))
    candidates.append(
        normalize(staged, branch="staged_t3_t6", loss_profile="standard")
    )
    candidates.append(
        normalize(staged_state_path, branch="staged_t3_t6", loss_profile="state_path")
    )
    candidates.append(
        normalize(equal_state_path, branch="equal_t3_t6", loss_profile="state_path")
    )
    if equal_state_kd is not None:
        candidates.append(
            normalize(equal_state_kd, branch="equal_t3_t6", loss_profile="state_kd")
        )
    if equal_state_kd_open_t3 is not None:
        candidates.append(
            normalize(
                equal_state_kd_open_t3,
                branch="equal_t3_t6_open_t3",
                loss_profile="state_kd",
            )
        )
    if equal_state_path_open_t3 is not None:
        candidates.append(
            normalize(
                equal_state_path_open_t3,
                branch="equal_t3_t6_open_t3",
                loss_profile="state_path",
            )
        )
    if equal_state_kd_open_t3_lr_scan is not None:
        if equal_state_kd_open_t3_lr_scan.get("loss_profile") != "state_kd":
            raise ValueError("Equal open-T3 state-KD LR-scan profile drift")
        equal_open_lr_rows = {
            float(row.get("learning_rate", -1)): row
            for row in equal_state_kd_open_t3_lr_scan.get("arms") or []
        }
        if set(equal_open_lr_rows) != {2e-6, 5e-6}:
            raise ValueError("Equal open-T3 state-KD LR candidate set drift")
        for learning_rate, row in sorted(equal_open_lr_rows.items()):
            candidates.append(
                normalize(
                    row,
                    branch="equal_t3_t6_open_t3",
                    loss_profile="state_kd",
                    learning_rate=learning_rate,
                )
            )
    if equal_state_kd_open_t3_lr1e6 is not None:
        candidates.append(
            normalize(
                equal_state_kd_open_t3_lr1e6,
                branch="equal_t3_t6_open_t3",
                loss_profile="state_kd",
                learning_rate=1e-6,
            )
        )
    eedi_full_rows = {
        float(row.get("learning_rate", -1)): row for row in eedi_full.get("arms") or []
    }
    if set(eedi_full_rows) != {2e-6, 5e-6}:
        raise ValueError("Eedi full LR candidate set drift")
    for learning_rate, row in sorted(eedi_full_rows.items()):
        candidates.append(
            normalize(
                row,
                branch="eedi_full",
                loss_profile="state_path",
                learning_rate=learning_rate,
            )
        )
    eedi_full_state_kd_rows = {
        float(row.get("learning_rate", -1)): row
        for row in eedi_full_state_kd.get("arms") or []
    }
    if set(eedi_full_state_kd_rows) != {2e-6, 5e-6}:
        raise ValueError("Eedi full state-KD LR candidate set drift")
    for learning_rate, row in sorted(eedi_full_state_kd_rows.items()):
        candidates.append(
            normalize(
                row,
                branch="eedi_full",
                loss_profile="state_kd",
                learning_rate=learning_rate,
            )
        )
    if eedi_full_open_t3 is not None:
        if eedi_full_open_t3.get("loss_profile") != "state_path":
            raise ValueError("Eedi full open-T3 loss-profile drift")
        eedi_full_open_rows = {
            float(row.get("learning_rate", -1)): row
            for row in eedi_full_open_t3.get("arms") or []
        }
        if set(eedi_full_open_rows) != {2e-6, 5e-6}:
            raise ValueError("Eedi full open-T3 LR candidate set drift")
        for learning_rate, row in sorted(eedi_full_open_rows.items()):
            candidates.append(
                normalize(
                    row,
                    branch="eedi_full_open_t3",
                    loss_profile="state_path",
                    learning_rate=learning_rate,
                )
            )
    if eedi_full_open_t3_selective_kd is not None:
        if eedi_full_open_t3_selective_kd.get("loss_profile") != "state_kd":
            raise ValueError("Eedi full open-T3 selective-KD loss-profile drift")
        selective_rows = {
            float(row.get("learning_rate", -1)): row
            for row in eedi_full_open_t3_selective_kd.get("arms") or []
        }
        if set(selective_rows) != {2e-6, 5e-6}:
            raise ValueError("Eedi full open-T3 selective-KD LR candidate set drift")
        for learning_rate, row in sorted(selective_rows.items()):
            candidates.append(
                normalize(
                    row,
                    branch="eedi_full_open_t3",
                    loss_profile="state_kd",
                    learning_rate=learning_rate,
                )
            )
    return choose(
        candidates,
        total_candidate_count=(
            22
            if equal_state_kd_open_t3_lr1e6 is not None
            else 21
            if equal_state_kd_open_t3_lr_scan is not None
            else 19
            if equal_state_path_open_t3 is not None
            else 18
            if equal_state_kd_open_t3 is not None
            else (17 if equal_state_kd is not None else len(candidates))
        ),
    )


def select_available(
    paths: dict[str, Path], t6_calibrated_candidate: Path | None = None
) -> dict[str, Any]:
    """Select from completed diagnostic artifacts without imputing missing arms."""

    candidates: list[dict[str, Any]] = []
    for key, path in paths.items():
        if not path.is_file() or not (path.parent / "complete").is_file():
            continue
        value = read(path)
        if key == "open_t3_consolidation":
            candidates.extend(normalize_consolidation_rows(path))
        elif key == "rehearsal":
            if value.get("stage") != "layerwise_task_retention_rehearsal_comparison":
                raise ValueError("Rehearsal comparison stage drift")
            rows = {str(row.get("branch")): row for row in value.get("arms") or []}
            if set(rows) != {"equal_t3_t6", "t3_heavy"}:
                raise ValueError("Base rehearsal candidate set drift")
            candidates.extend(normalize(row, branch=branch, loss_profile="standard") for branch, row in sorted(rows.items()))
        elif key == "eedi":
            if value.get("stage") != "layerwise_eedi_state_path_ab_comparison":
                raise ValueError("Eedi comparison stage drift")
            rows = {str(row.get("loss_profile")): row for row in value.get("profiles") or []}
            if set(rows) != {"standard", "state_path"}:
                raise ValueError("Eedi objective candidate set drift")
            candidates.extend(normalize(row, branch="eedi_dense", loss_profile=profile) for profile, row in sorted(rows.items()))
        elif key in {"state_kd", "staged", "staged_state_path", "equal_state_path", "equal_state_kd", "equal_state_kd_open_t3", "equal_state_path_open_t3", "equal_state_kd_open_t3_lr1e6"}:
            if value.get("stage") != "layerwise_single_task_retention_candidate":
                raise ValueError(f"Single-candidate stage drift: {key}")
            expected = {
                "state_kd": ("eedi_dense", "state_kd"),
                "staged": ("staged_t3_t6", "standard"),
                "staged_state_path": ("staged_t3_t6", "state_path"),
                "equal_state_path": ("equal_t3_t6", "state_path"),
                "equal_state_kd": ("equal_t3_t6", "state_kd"),
                "equal_state_kd_open_t3": ("equal_t3_t6_open_t3", "state_kd"),
                "equal_state_path_open_t3": ("equal_t3_t6_open_t3", "state_path"),
                "equal_state_kd_open_t3_lr1e6": (
                    "equal_t3_t6_open_t3",
                    "state_kd",
                ),
            }[key]
            candidates.append(
                normalize(
                    value,
                    branch=expected[0],
                    loss_profile=expected[1],
                    learning_rate=(
                        1e-6 if key == "equal_state_kd_open_t3_lr1e6" else 5e-7
                    ),
                )
            )
        elif key in {"eedi_full", "eedi_full_state_kd", "eedi_full_open_t3", "eedi_full_open_t3_selective_kd", "equal_state_kd_open_t3_lr_scan"}:
            if value.get("stage") != "layerwise_eedi_full_lr_scan_comparison":
                raise ValueError(f"LR-scan stage drift: {key}")
            branch = (
                "equal_t3_t6_open_t3"
                if key == "equal_state_kd_open_t3_lr_scan"
                else "eedi_full_open_t3"
                if "open_t3" in key
                else "eedi_full"
            )
            profile = "state_kd" if "state_kd" in key or "selective_kd" in key else "state_path"
            if ("open_t3" in key) and value.get("loss_profile") != profile:
                raise ValueError(f"Open-T3 loss-profile drift: {key}")
            rows = {float(row.get("learning_rate", -1)): row for row in value.get("arms") or []}
            if set(rows) != {2e-6, 5e-6}:
                raise ValueError(f"LR candidate set drift: {key}")
            candidates.extend(
                normalize(row, branch=branch, loss_profile=profile, learning_rate=lr)
                for lr, row in sorted(rows.items())
            )
        else:
            raise ValueError(f"Unknown candidate source: {key}")
    if t6_calibrated_candidate is not None:
        if not t6_calibrated_candidate.is_file():
            raise ValueError("T6 calibration overlay is missing")
        candidates = apply_t6_calibration_overlay(
            candidates, t6_calibrated_candidate
        )
    return choose(candidates, total_candidate_count=24)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--eedi", type=Path, required=True)
    parser.add_argument("--state-kd", type=Path, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--staged-state-path", type=Path, required=True)
    parser.add_argument("--equal-state-path", type=Path, required=True)
    parser.add_argument("--eedi-full", type=Path, required=True)
    parser.add_argument("--eedi-full-state-kd", type=Path, required=True)
    parser.add_argument("--eedi-full-open-t3", type=Path)
    parser.add_argument("--eedi-full-open-t3-selective-kd", type=Path)
    parser.add_argument("--equal-state-kd", type=Path)
    parser.add_argument("--equal-state-kd-open-t3", type=Path)
    parser.add_argument("--equal-state-path-open-t3", type=Path)
    parser.add_argument("--equal-state-kd-open-t3-lr-scan", type=Path)
    parser.add_argument("--equal-state-kd-open-t3-lr1e6", type=Path)
    parser.add_argument("--open-t3-consolidation", type=Path)
    parser.add_argument("--t6-calibrated-candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--available", action="store_true")
    args = parser.parse_args()
    available_paths = {
        "rehearsal": args.rehearsal,
        "eedi": args.eedi,
        "state_kd": args.state_kd,
        "staged": args.staged,
        "staged_state_path": args.staged_state_path,
        "equal_state_path": args.equal_state_path,
        "equal_state_kd": args.equal_state_kd,
        "equal_state_kd_open_t3": args.equal_state_kd_open_t3,
        "equal_state_path_open_t3": args.equal_state_path_open_t3,
        "equal_state_kd_open_t3_lr_scan": args.equal_state_kd_open_t3_lr_scan,
        "equal_state_kd_open_t3_lr1e6": args.equal_state_kd_open_t3_lr1e6,
        "open_t3_consolidation": args.open_t3_consolidation,
        "eedi_full": args.eedi_full,
        "eedi_full_state_kd": args.eedi_full_state_kd,
        "eedi_full_open_t3": args.eedi_full_open_t3,
        "eedi_full_open_t3_selective_kd": args.eedi_full_open_t3_selective_kd,
    }
    if args.available:
        value = select_available(
            {key: path for key, path in available_paths.items() if path is not None},
            args.t6_calibrated_candidate,
        )
    else:
        value = select(
        read(args.rehearsal),
        read(args.eedi),
        read(args.state_kd),
        read(args.staged),
        read(args.staged_state_path),
        read(args.equal_state_path),
        read(args.eedi_full),
        read(args.eedi_full_state_kd),
        read(args.eedi_full_open_t3) if args.eedi_full_open_t3 else None,
        read(args.eedi_full_open_t3_selective_kd)
        if args.eedi_full_open_t3_selective_kd
        else None,
        read(args.equal_state_kd) if args.equal_state_kd else None,
        read(args.equal_state_kd_open_t3) if args.equal_state_kd_open_t3 else None,
        read(args.equal_state_path_open_t3) if args.equal_state_path_open_t3 else None,
        read(args.equal_state_kd_open_t3_lr_scan)
        if args.equal_state_kd_open_t3_lr_scan
        else None,
        read(args.equal_state_kd_open_t3_lr1e6)
        if args.equal_state_kd_open_t3_lr1e6
        else None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["any_all_gates_pass"] else 2)


if __name__ == "__main__":
    main()
