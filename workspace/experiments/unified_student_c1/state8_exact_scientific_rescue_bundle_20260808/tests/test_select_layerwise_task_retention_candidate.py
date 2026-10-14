from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_layerwise_task_retention_candidate import (  # noqa: E402
    apply_t6_calibration_overlay,
    normalize,
    prevalence_corrected_t3,
    select,
    select_available,
)


def candidate(branch: str, *, passed: bool, t3: float, t6: float, c4: float) -> dict:
    return {
        "branch": branch,
        "checks": {"all": passed, "causal_recipe": True},
        "all_gates_pass": passed,
        "causal_weight": 2.0,
        "T3_semantic": {
            "pass": passed,
            "retention_ratio": t3,
            "weighting_scheme": "formal_panel_group_prevalence_v1",
            "formal_panel_support": 1494,
            "group_floor_pass": passed,
        },
        "T6": {"retention_ratio": t6},
        "C4": {"D_specific": c4},
    }


def fixtures():
    rehearsal = {
        "stage": "layerwise_task_retention_rehearsal_comparison",
        "arms": [
            candidate("equal_t3_t6", passed=False, t3=0.5, t6=0.9, c4=0.2),
            candidate("t3_heavy", passed=False, t3=0.6, t6=0.7, c4=0.2),
        ],
    }
    eedi = {
        "stage": "layerwise_eedi_state_path_ab_comparison",
        "profiles": [
            {**candidate("eedi_dense", passed=True, t3=0.9, t6=0.86, c4=0.16), "loss_profile": "standard"},
            {**candidate("eedi_dense", passed=True, t3=0.95, t6=0.88, c4=0.17), "loss_profile": "state_path"},
        ],
    }
    state_kd = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("eedi_dense", passed=False, t3=0.8, t6=0.9, c4=0.2),
        "loss_profile": "state_kd",
    }
    staged = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("staged_t3_t6", passed=True, t3=1.0, t6=0.87, c4=0.18),
        "loss_profile": "standard",
    }
    equal_state_path = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=False, t3=0.7, t6=0.9, c4=0.19),
        "loss_profile": "state_path",
    }
    staged_state_path = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("staged_t3_t6", passed=False, t3=0.8, t6=0.9, c4=0.2),
        "loss_profile": "state_path",
    }
    eedi_full = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "arms": [
            {
                **candidate("eedi_full", passed=False, t3=1.1, t6=0.9, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": 2e-6,
                "candidate_id": "eedi_full__state_path__lr2e-06",
            },
            {
                **candidate("eedi_full", passed=False, t3=1.2, t6=0.9, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": 5e-6,
                "candidate_id": "eedi_full__state_path__lr5e-06",
            },
        ],
    }
    eedi_full_state_kd = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "arms": [
            {
                **candidate("eedi_full", passed=False, t3=1.0, t6=0.95, c4=0.2),
                "loss_profile": "state_kd",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full__state_kd__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    return rehearsal, eedi, state_kd, staged, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd


def test_selects_highest_t3_among_only_fully_gated_candidates() -> None:
    result = select(*fixtures())
    assert result["candidate_count"] == 12
    assert result["selected_candidate"] == {
        "all_gates_pass": True,
        "candidate_id": "staged_t3_t6__standard",
        "branch": "staged_t3_t6",
        "loss_profile": "standard",
        "learning_rate": 5e-7,
        "causal_weight": 2.0,
        "T3_retention_ratio": 1.0,
        "T6_retention_ratio": 0.87,
        "C4_D_specific": 0.18,
    }


def test_legacy_balanced_t3_is_reweighted_before_selection() -> None:
    legacy = {
        "pass": True,
        "retention_ratio": 1.2,
        "by_group": {
            "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
                "support": 25,
                "semantic_accuracy": 0.0,
            },
            "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
                "support": 25,
                "semantic_accuracy": 0.8,
            },
            "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
                "support": 25,
                "semantic_accuracy": 0.8,
            },
        },
    }
    corrected = prevalence_corrected_t3(legacy)
    assert corrected["weighting_scheme"] == "formal_panel_group_prevalence_v1"
    assert corrected["formal_panel_support"] == 1494
    assert corrected["group_floor_pass"] is False
    assert corrected["pass"] is False


def test_t6_overlay_canonicalizes_legacy_t3_before_evidence_comparison(
    tmp_path: Path,
) -> None:
    legacy_t3 = {
        "pass": True,
        "retention_ratio": 1.2,
        "by_group": {
            "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
                "support": 25,
                "semantic_accuracy": 0.5,
            },
            "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {
                "support": 25,
                "semantic_accuracy": 0.5,
            },
            "diagnose_misconception/malrule/synthetic_same_template_open_generation": {
                "support": 25,
                "semantic_accuracy": 0.5,
            },
        },
    }
    raw = {
        "candidate_id": "eedi_full_open_t3__state_path__lr2e-06",
        "branch": "eedi_full_open_t3",
        "loss_profile": "state_path",
        "learning_rate": 2e-6,
        "causal_weight": 2.0,
        "checks": {"C4": True, "T3_semantic": True, "causal_recipe": True},
        "all_gates_pass": True,
        "C4": {"D_specific": 0.2},
        "T3_semantic": legacy_t3,
        "T6": {"retention_ratio": 0.8},
        "formal_screen": {"diagnostic_gate_pass": True},
    }
    source = normalize(
        raw,
        branch="eedi_full_open_t3",
        loss_profile="state_path",
        learning_rate=2e-6,
    )
    overlay = copy.deepcopy(raw)
    overlay.update(
        {
            "stage": "layerwise_t6_calibrated_candidate_v1",
            "T6": {"retention_ratio": 0.92, "calibrated": True},
            "t6_calibration": {
                "stage": "layerwise_t6_threshold_calibration_v1"
            },
        }
    )
    path = tmp_path / "overlay.json"
    path.write_text(__import__("json").dumps(overlay))

    result = apply_t6_calibration_overlay([source], path)

    assert result[0]["T3_semantic"] == source["T3_semantic"]
    assert result[0]["T6"]["retention_ratio"] == 0.92


def test_preserves_state_path_profile_in_selection() -> None:
    rehearsal, eedi, state_kd, staged, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd = fixtures()
    staged["checks"] = {"all": False, "causal_recipe": True}
    staged["all_gates_pass"] = False
    result = select(rehearsal, eedi, state_kd, staged, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd)
    assert result["selected_candidate"]["candidate_id"] == "eedi_dense__state_path"
    assert result["selected_candidate"]["loss_profile"] == "state_path"
    assert result["selected_candidate"]["all_gates_pass"] is True


def test_inconsistent_summary_fails_closed() -> None:
    rehearsal, eedi, state_kd, staged, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd = fixtures()
    broken = copy.deepcopy(staged)
    broken["all_gates_pass"] = True
    broken["checks"] = {"all": False}
    try:
        select(rehearsal, eedi, state_kd, broken, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent candidate was accepted")


def test_selects_passing_full_eedi_lr_candidate_without_losing_lr() -> None:
    rehearsal, eedi, state_kd, staged, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd = fixtures()
    staged["checks"] = {"all": False, "causal_recipe": True}
    staged["all_gates_pass"] = False
    for row in eedi["profiles"]:
        row["checks"] = {"all": False, "causal_recipe": True}
        row["all_gates_pass"] = False
    eedi_full["arms"][0]["checks"] = {"all": True, "causal_recipe": True}
    eedi_full["arms"][0]["all_gates_pass"] = True
    eedi_full["arms"][0]["T3_semantic"].update(
        {"pass": True, "group_floor_pass": True}
    )
    result = select(rehearsal, eedi, state_kd, staged, staged_state_path, equal_state_path, eedi_full, eedi_full_state_kd)
    assert result["selected_candidate"]["candidate_id"] == "eedi_full__state_path__lr2e-06"
    assert result["selected_candidate"]["learning_rate"] == 2e-6


def test_includes_and_selects_open_t3_full_eedi_candidates() -> None:
    values = fixtures()
    open_t3 = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_path",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=learning_rate == 2e-6, t3=1.3, t6=0.9, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_path__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    result = select(*values, open_t3)
    assert result["candidate_count"] == 14
    assert result["selected_candidate"]["candidate_id"] == "eedi_full_open_t3__state_path__lr2e-06"


def test_includes_selective_kd_open_t3_candidates() -> None:
    values = fixtures()
    open_t3 = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_path",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_path__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    selective = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_kd",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=learning_rate == 2e-6, t3=1.1, t6=0.95, c4=0.22),
                "loss_profile": "state_kd",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_kd__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    result = select(*values, open_t3, selective)
    assert result["candidate_count"] == 16
    assert result["selected_candidate"]["candidate_id"] == "eedi_full_open_t3__state_kd__lr2e-06"


def test_includes_equal_curriculum_state_kd_candidate() -> None:
    values = fixtures()
    open_t3 = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_path",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_path__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    selective = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_kd",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_kd",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_kd__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    equal_state_kd = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=True, t3=1.4, t6=0.95, c4=0.2),
        "loss_profile": "state_kd",
    }
    result = select(*values, open_t3, selective, equal_state_kd)
    assert result["candidate_count"] == 17
    assert result["selected_candidate"]["candidate_id"] == "equal_t3_t6__state_kd"


def test_includes_equal_curriculum_open_t3_state_kd_candidate() -> None:
    values = fixtures()
    open_t3 = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_path",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_path__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    selective = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_kd",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_kd",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_kd__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    equal_state_kd = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=False, t3=0.8, t6=0.9, c4=0.2),
        "loss_profile": "state_kd",
    }
    equal_open = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6_open_t3", passed=True, t3=1.1, t6=0.95, c4=0.21),
        "loss_profile": "state_kd",
    }
    result = select(*values, open_t3, selective, equal_state_kd, equal_open)
    assert result["candidate_count"] == 18
    assert result["selected_candidate"]["candidate_id"] == "equal_t3_t6_open_t3__state_kd"


def test_includes_equal_curriculum_open_t3_state_path_candidate() -> None:
    values = fixtures()
    open_t3 = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_path",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_path",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_path__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    selective = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_kd",
        "arms": [
            {
                **candidate("eedi_full_open_t3", passed=False, t3=0.7, t6=0.7, c4=0.2),
                "loss_profile": "state_kd",
                "learning_rate": learning_rate,
                "candidate_id": f"eedi_full_open_t3__state_kd__lr{learning_rate:.0e}",
            }
            for learning_rate in (2e-6, 5e-6)
        ],
    }
    equal_state_kd = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=False, t3=0.8, t6=0.9, c4=0.2),
        "loss_profile": "state_kd",
    }
    equal_open_kd = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6_open_t3", passed=False, t3=0.8, t6=0.9, c4=0.2),
        "loss_profile": "state_kd",
    }
    equal_open_path = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6_open_t3", passed=True, t3=1.2, t6=0.9, c4=0.2),
        "loss_profile": "state_path",
    }
    result = select(
        *values,
        open_t3,
        selective,
        equal_state_kd,
        equal_open_kd,
        equal_open_path,
    )
    assert result["candidate_count"] == 19
    assert result["total_candidate_count"] == 19
    assert result["selected_candidate"]["candidate_id"] == "equal_t3_t6_open_t3__state_path"


def test_available_selection_uses_only_completed_summaries(tmp_path: Path) -> None:
    failed = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=False, t3=0.8, t6=0.9, c4=0.2),
        "loss_profile": "state_path",
    }
    passed = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=True, t3=1.1, t6=0.95, c4=0.18),
        "loss_profile": "state_kd",
    }
    failed_path = tmp_path / "failed/summary.json"
    passed_path = tmp_path / "passed/summary.json"
    failed_path.parent.mkdir(); passed_path.parent.mkdir()
    failed_path.write_text(__import__("json").dumps(failed))
    passed_path.write_text(__import__("json").dumps(passed))
    (failed_path.parent / "complete").touch()
    # The passing summary is not authoritative until its owner writes complete.
    result = select_available({"equal_state_path": failed_path, "equal_state_kd": passed_path})
    assert result["candidate_count"] == 1
    assert result["any_all_gates_pass"] is False
    assert result["all_diagnostics_complete"] is False
    (passed_path.parent / "complete").touch()
    result = select_available({"equal_state_path": failed_path, "equal_state_kd": passed_path})
    assert result["candidate_count"] == 2
    assert result["any_all_gates_pass"] is True
    assert result["selected_candidate"]["candidate_id"] == "equal_t3_t6__state_kd"
    assert result["selection_scope"] == "completed_candidates_only"


def test_available_selection_rejects_inconsistent_completed_summary(tmp_path: Path) -> None:
    broken = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate("equal_t3_t6", passed=True, t3=1.1, t6=0.95, c4=0.18),
        "loss_profile": "state_kd",
    }
    broken["checks"] = {"all": False, "causal_recipe": True}
    path = tmp_path / "broken/summary.json"
    path.parent.mkdir(); path.write_text(__import__("json").dumps(broken))
    (path.parent / "complete").touch()
    try:
        select_available({"equal_state_kd": path})
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent completed candidate was accepted")


def test_available_selection_includes_equal_open_state_kd_lr_scan(
    tmp_path: Path,
) -> None:
    arms = []
    for learning_rate in (2e-6, 5e-6):
        row = {
            **candidate(
                "equal_t3_t6_open_t3",
                passed=learning_rate == 2e-6,
                t3=1.1,
                t6=0.92,
                c4=0.21,
            ),
            "loss_profile": "state_kd",
            "learning_rate": learning_rate,
            "candidate_id": (
                "equal_t3_t6_open_t3__state_kd__"
                f"lr{learning_rate:.0e}"
            ),
        }
        arms.append(row)
    comparison = {
        "stage": "layerwise_eedi_full_lr_scan_comparison",
        "loss_profile": "state_kd",
        "arms": arms,
    }
    path = tmp_path / "equal_open_lr/comparison.json"
    path.parent.mkdir()
    path.write_text(__import__("json").dumps(comparison))
    (path.parent / "complete").touch()
    result = select_available({"equal_state_kd_open_t3_lr_scan": path})
    assert result["candidate_count"] == 2
    assert result["total_candidate_count"] == 24
    assert result["selected_candidate"]["candidate_id"] == (
        "equal_t3_t6_open_t3__state_kd__lr2e-06"
    )


def test_available_selection_includes_equal_open_state_kd_lr1e6(
    tmp_path: Path,
) -> None:
    row = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate(
            "equal_t3_t6_open_t3",
            passed=True,
            t3=1.0,
            t6=0.9,
            c4=0.2,
        ),
        "loss_profile": "state_kd",
        "learning_rate": 1e-6,
        "candidate_id": "equal_t3_t6_open_t3__state_kd__lr1e-06",
    }
    path = tmp_path / "equal_open_lr1e6/summary.json"
    path.parent.mkdir()
    path.write_text(__import__("json").dumps(row))
    (path.parent / "complete").touch()
    result = select_available({"equal_state_kd_open_t3_lr1e6": path})
    assert result["candidate_count"] == 1
    assert result["total_candidate_count"] == 24
    assert result["selected_candidate"]["candidate_id"] == (
        "equal_t3_t6_open_t3__state_kd__lr1e-06"
    )


def test_available_selection_includes_completed_two_stage_consolidation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "consolidation"
    root.mkdir()
    rows = []
    for profile in ("state_kd", "state_path"):
        passed = profile == "state_path"
        row = {
            "stage": "layerwise_single_task_retention_candidate",
            **candidate(
                "equal_t3_t6_open_t3",
                passed=passed,
                t3=1.0,
                t6=0.9,
                c4=0.2,
            ),
            "loss_profile": profile,
            "learning_rate": 5e-7,
            "candidate_id": f"equal_t3_t6_open_t3__{profile}",
        }
        (root / f"{profile}.summary.json").write_text(
            __import__("json").dumps(row)
        )
        rows.append(row)
    selection = {
        "stage": "layerwise_open_t3_consolidation_selection_v1",
        "all_gates_pass": True,
        "selected_candidate_id": "equal_t3_t6_open_t3__state_path",
    }
    path = root / "selection.json"
    path.write_text(__import__("json").dumps(selection))
    # A result without its owner's completion marker is never selectable.
    result = select_available({"open_t3_consolidation": path})
    assert result["candidate_count"] == 0
    (root / "complete").touch()
    result = select_available({"open_t3_consolidation": path})
    assert result["candidate_count"] == 2
    assert result["total_candidate_count"] == 24
    assert result["selected_candidate"]["candidate_id"] == (
        "equal_t3_t6_open_t3_two_stage__state_path"
    )


def test_two_stage_respects_its_preregistered_arm_selection(tmp_path: Path) -> None:
    root = tmp_path / "consolidation"
    root.mkdir()
    # Both pass. State-KD has the larger T3 score, but the upstream balanced
    # gate selected state-path. The global selector must not silently re-rank
    # the two arms under a different rule after seeing their results.
    for profile, t3 in (("state_kd", 1.5), ("state_path", 1.0)):
        row = {
            "stage": "layerwise_single_task_retention_candidate",
            **candidate(
                "equal_t3_t6_open_t3",
                passed=True,
                t3=t3,
                t6=0.9,
                c4=0.2,
            ),
            "loss_profile": profile,
            "learning_rate": 5e-7,
            "candidate_id": f"equal_t3_t6_open_t3__{profile}",
        }
        (root / f"{profile}.summary.json").write_text(
            __import__("json").dumps(row)
        )
    path = root / "selection.json"
    path.write_text(__import__("json").dumps({
        "stage": "layerwise_open_t3_consolidation_selection_v1",
        "all_gates_pass": True,
        "selected_candidate_id": "equal_t3_t6_open_t3__state_path",
    }))
    (root / "complete").touch()
    result = select_available({"open_t3_consolidation": path})
    assert result["candidate_count"] == 2
    assert result["selected_candidate"]["candidate_id"] == (
        "equal_t3_t6_open_t3_two_stage__state_path"
    )
    by_id = {row["candidate_id"]: row for row in result["candidates"]}
    assert by_id["equal_t3_t6_open_t3_two_stage__state_kd"][
        "promotion_eligible"
    ] is False


def test_completed_two_stage_accepts_but_never_promotes_c4_pruned_arm(
    tmp_path: Path,
) -> None:
    root = tmp_path / "consolidation"
    root.mkdir()
    state_kd = {
        "stage": "layerwise_single_task_retention_candidate",
        **candidate(
            "equal_t3_t6_open_t3",
            passed=True,
            t3=1.0,
            t6=0.9,
            c4=0.2,
        ),
        "loss_profile": "state_kd",
        "learning_rate": 5e-7,
        "candidate_id": "equal_t3_t6_open_t3__state_kd",
    }
    state_path = {
        "stage": "layerwise_single_task_retention_candidate_pruned_v1",
        "candidate_id": "equal_t3_t6_open_t3__state_path__lr5e-07",
        "loss_profile": "state_path",
        "learning_rate": 5e-7,
        "checkpoint_round": 3500,
        "all_gates_pass": False,
        "pruned": True,
        "prune_reason": "frozen_full_c4_specificity_gate_failed",
        "checks": {"C4": False, "specificity": False},
        "C4": {"D_specific": 0.0},
    }
    (root / "state_kd.summary.json").write_text(
        __import__("json").dumps(state_kd)
    )
    (root / "state_path.summary.json").write_text(
        __import__("json").dumps(state_path)
    )
    selection = {
        "stage": "layerwise_open_t3_consolidation_selection_v1",
        "all_gates_pass": True,
        "selected_candidate_id": "equal_t3_t6_open_t3__state_kd",
    }
    path = root / "selection.json"
    path.write_text(__import__("json").dumps(selection))
    (root / "complete").touch()

    result = select_available({"open_t3_consolidation": path})

    assert result["candidate_count"] == 2
    assert result["selected_candidate"]["candidate_id"] == (
        "equal_t3_t6_open_t3_two_stage__state_kd"
    )
    by_id = {row["candidate_id"]: row for row in result["candidates"]}
    pruned = by_id["equal_t3_t6_open_t3_two_stage__state_path"]
    assert pruned["pruned"] is True
    assert pruned["promotion_eligible"] is False
