#!/usr/bin/env python3
"""Preregistered DBE smoke gate for promoting lrfix-v4 to a full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE = "state8_profile_ownership_lrfix_v4_promotion_gate_v1"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def validate(summary_path: Path) -> dict[str, Any]:
    summary = read(summary_path)
    metrics = summary.get("metrics") or {}
    completion = summary.get("completion") or {}
    repair = summary.get("repair") or {}
    checks = {
        "watcher_stage": summary.get("stage")
        in {
            "state8_profile_ownership_lrfix_v4_sparsekc_smoke_watcher_v1",
            "state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_watcher_v1",
            "state8_profile_ownership_lrfix_v4_smoke_recovery_watcher_v1",
        },
        "remote_passed": summary.get("remote_passed") is True,
        "artifact_contract": not summary.get("failures"),
        "diagnostic_complete_200": completion.get("status") == "diagnostic_complete"
        and int(completion.get("micro_steps", -1)) == 200,
        "strict_state_only": metrics.get("strict_state_only") is True,
        "heldout_development_support": int(metrics.get("students", -1)) == 88
        and int(metrics.get("supported_student_kc_cells", -1)) == 634,
        "permutation_replicates": int(metrics.get("ownership_permutation_replicates", -1))
        == 2000,
        "positive_ownership_direction": float(
            metrics.get("ownership_delta_permuted_mean_minus_native", 0.0)
        ) > 0.0,
        # This is an explicit promotion threshold on the DBE development split,
        # not the final revised-C4 claim and never a substitute for the complete
        # reusable-interface gates or untouched XES confirmation.
        "ownership_development_p_lt_0_05": float(
            metrics.get("ownership_permutation_p_one_sided", 1.0)
        ) < 0.05,
        "feature_normalization": metrics.get("profile_ownership_feature_normalization")
        == "layernorm",
        "head_learning_rate": float(repair.get("head_learning_rate", -1.0)) == 3e-4,
        "head_reset": repair.get("head_reset_from_training_targets") is True,
        "sparse_kc_support": int(repair.get("minimum_kc_train_support", -1)) == 4,
        "bias_prior": float(repair.get("bias_prior_strength", -1.0)) == 8.0,
        "sparse_kc_policy": repair.get("sparse_kc_policy")
        == "exclude_below_train_support_then_weight_equally",
        "no_confirmatory_claim": metrics.get("diagnostic_only") is True,
    }
    return {
        "stage": STAGE,
        "status": "pass" if all(checks.values()) else "fail",
        "promotion_allowed": all(checks.values()),
        "scope": "DBE development diagnostic only",
        "not_confirmatory": True,
        "xes_official_test_inspected": False,
        "checks": checks,
        "metrics": {
            "native_rmse": metrics.get("native_rmse"),
            "ownership_delta_permuted_mean_minus_native": metrics.get(
                "ownership_delta_permuted_mean_minus_native"
            ),
            "ownership_permutation_p_one_sided": metrics.get(
                "ownership_permutation_p_one_sided"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.smoke_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["promotion_allowed"] else 2)


if __name__ == "__main__":
    main()
