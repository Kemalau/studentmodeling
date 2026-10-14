#!/usr/bin/env python3
"""Apply the exact joint gate to diagnosis-first consolidation milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from select_state8_exact_diagnosis_curriculum import score_candidate
except ImportError:
    from scripts.select_state8_exact_diagnosis_curriculum import score_candidate


STEPS = (200, 400, 600, 800)


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} below {root}, observed={paths}")
    return paths[0]


def candidate_config(step: int) -> dict[str, object]:
    if step not in STEPS:
        raise ValueError(f"Unsupported milestone step: {step}")
    return {
        "job": f"state8-exact-consolidation-step{step}-eval",
        "slug": f"exact_consolidation_milestone_step{step}",
        "stable_root": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            f"state8_exact_consolidation_milestone_step{step}_v1"
        ),
        # This experiment is diagnostic until a separate, fail-closed final-fit
        # route is materialized after observing a genuinely qualified checkpoint.
        "finalfit_variant": None,
        "finalfit_job": None,
    }


def select(root: Path, steps: tuple[int, ...] = STEPS) -> dict:
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("Milestone steps must be unique and non-empty")
    candidates = []
    for step in steps:
        config = candidate_config(step)
        candidate_root = root / f"step{step}"
        row = score_candidate(
            candidate_root,
            job=str(config["job"]),
            slug=str(config["slug"]),
            stable_root=str(config["stable_root"]),
            finalfit_variant="diagnostic_route_not_materialized",
            finalfit_job="diagnostic_route_not_materialized",
            allowed_repair_entrypoints=(
                "scripts/train_state8_diagnosis_first_causal_repair.py",
            ),
        )
        artifact = one(candidate_root / "download", f"**/{config['job']}").resolve()
        audit = read(artifact / "metrics/checkpoint_audit.json")
        completion = read(artifact / "metrics/completion.json")
        if audit.get("stage") != "state8_exact_consolidation_milestone_audit_v1":
            raise ValueError(f"Milestone audit stage drift: step={step}")
        if (
            int(audit.get("checkpoint_step", -1)) != step
            or int(audit.get("completed_optimizer_steps", -1)) != step // 4
            or int(audit.get("planned_micro_steps", -1)) != 1000
            or int(audit.get("planned_optimizer_steps", -1)) != 250
            or float(audit.get("consolidation_fraction", -1.0)) != step / 1000
        ):
            raise ValueError(f"Milestone coordinate drift: step={step}")
        if audit.get("history_dropout_counts") != {"1.0": step}:
            raise ValueError(f"Milestone history dropout drift: step={step}")
        if audit.get("artifact_sha256") != row["artifact_sha256"]:
            raise ValueError(f"Milestone artifact binding drift: step={step}")
        if (
            audit.get("state_shape") != [64, 8, 5120]
            or audit.get("raw_history_visible_to_decoder") is not False
            or audit.get("persistent_history_kv_cache") is not False
        ):
            raise ValueError(f"Milestone state-only contract drift: step={step}")
        if (
            completion.get("stage")
            != "state8_exact_consolidation_milestone_completion_v1"
            or completion.get("status") != "checkpoint_complete"
            or completion.get("diagnostic") is not True
            or completion.get("early_stop_candidate_only") is not True
            or completion.get("real_examples_seen_once") is not False
            or int(completion.get("checkpoint_step", -1)) != step
            or completion.get("sha256") != row["artifact_sha256"]
        ):
            raise ValueError(f"Milestone completion provenance drift: step={step}")
        row.update(
            checkpoint_step=step,
            consolidation_fraction=step / 1000,
            diagnostic_only=True,
            finalfit_route_materialized=False,
        )
        candidates.append(row)
    qualified = [row for row in candidates if row["qualified"]]
    selected = max(
        qualified, key=lambda row: (float(row["score"]), int(row["checkpoint_step"]))
    ) if qualified else None
    return {
        "stage": "state8_exact_consolidation_milestone_joint_gate_v1",
        "status": "complete",
        "pass": selected is not None,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: int(row["checkpoint_step"])),
        "diagnostic_only": True,
        "promotion_requires_explicit_finalfit_route_materialization": True,
        "selection_scope": "strict_state_only_early_stop_unified_answer_and_diagnosis_c4",
        "exact_match_authoritative": False,
        "semantic_judge_authoritative": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
