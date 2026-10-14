from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_unified_final_eval import FROZEN_GROUP_SUPPORT  # noqa: E402
from validate_layerwise_unified_pipeline import (  # noqa: E402
    EXPECTED_CAUSAL_EXPOSURES,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_REAL_EXAMPLES,
    EXPECTED_UNIQUE_CAUSAL_PAIRS,
    EXPECTED_DATA_COUNTS,
    EXPECTED_FINAL_CORPUS_SHA256,
    EXPECTED_TRAIN_SHA256,
    EXPECTED_VALID_SHA256,
    validate,
)


class OverallTest(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        markers = [
            "final_fit/complete",
            "final_fit/scientific_gate_pass",
            "final_eval/complete",
            "final_eval/scientific_gate_pass",
            "t3_semantic_judge/complete",
            "t3_semantic_judge/scientific_gate_pass",
        ]
        for marker in markers:
            path = root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        metrics = root / "final_fit/download/job/metrics"
        metrics.mkdir(parents=True)
        (metrics / "final_fit_receipt.json").write_text(json.dumps({
            "status": "final_fit_complete",
            "stage": "layerwise_unified_final_fit",
            "state_architecture": "layerwise",
            "state_shape": [64, 5120],
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "real_examples": EXPECTED_REAL_EXAMPLES,
            "one_epoch_exact": True,
            "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        }))
        model = root / "final_fit/download/job/model"
        model.mkdir()
        adapter = model / "adapter_model.safetensors"
        state = model / "recurrent_state.pt"
        adapter.write_bytes(b"adapter")
        state.write_bytes(b"state")
        adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
        state_sha = hashlib.sha256(state.read_bytes()).hexdigest()
        (metrics / "completion.json").write_text(json.dumps({
            "status": "complete",
            "diagnostic": False,
            "micro_steps": EXPECTED_REAL_EXAMPLES,
            "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "real_examples_seen": EXPECTED_REAL_EXAMPLES,
            "real_examples_seen_once": True,
            "history_rows": EXPECTED_REAL_EXAMPLES,
            "history_unique_global_positions": EXPECTED_REAL_EXAMPLES,
            "history_global_positions_contiguous": True,
            "history_unique_examples": EXPECTED_REAL_EXAMPLES,
            "history_integrity_pass": True,
            "exact_history_support": True,
            "causal_pairs_seen": EXPECTED_CAUSAL_EXPOSURES,
            "unique_causal_pairs_seen": EXPECTED_UNIQUE_CAUSAL_PAIRS,
            "causal_exposure_count_matches_plan": True,
            "all_causal_pairs_covered": True,
            "sha256": {"adapter_model": adapter_sha, "recurrent_state": state_sha},
        }))
        (metrics / "run_config.json").write_text(json.dumps({
            "state_architecture": "layerwise",
            "layerwise_operation_checkpointing": True,
            "layerwise_transition_rank": 256,
            "distributed": True,
            "distributed_world_size": 16,
            "real_train_rows": EXPECTED_REAL_EXAMPLES,
            "planned_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "planned_causal_exposures": EXPECTED_CAUSAL_EXPOSURES,
            "causal_train_pairs": EXPECTED_UNIQUE_CAUSAL_PAIRS,
            "causal_objective_mode": "specificity_triplet",
            "causal_repeat_fraction": 0.05,
            "normalize_causal_exposure": True,
            "state_token_alias": "<|vision_pad|>",
            "causal_branch_dropout_mode": "eval_deterministic",
            "train_sha256": EXPECTED_FINAL_CORPUS_SHA256,
        }))
        (metrics / "data_audit.json").write_text(json.dumps({
            "status": "passed",
            "required_tasks": [
                "diagnose_misconception",
                "predict_correctness",
                "predict_error_persistence",
                "predict_prospective_error",
                "predict_response",
            ],
            "train_valid_example_overlap": 0,
            "train_valid_real_student_overlap": {
                "dbe_kt22": 0,
                "eedi_mining": 0,
                "eedi_response": 0,
                "foundationalassist": 0,
                "malrule": 0,
                "xes3g5m": 0,
            },
            "train": {"rows": 244331, "decoder_history_leaks": 0, "carrier_errors": 0, "sha256": EXPECTED_TRAIN_SHA256},
            "valid": {"rows": 15514, "decoder_history_leaks": 0, "carrier_errors": 0, "sha256": EXPECTED_VALID_SHA256},
        }))
        (metrics / "final_fit_manifest.json").write_text(json.dumps({
            "purpose": "post_selection_final_fit_train_plus_validation",
            "unique_examples": EXPECTED_REAL_EXAMPLES,
            "counts": EXPECTED_DATA_COUNTS,
            "inputs": {
                "train": {"sha256": EXPECTED_TRAIN_SHA256},
                "valid": {"sha256": EXPECTED_VALID_SHA256},
            },
            "output": {"sha256": EXPECTED_FINAL_CORPUS_SHA256},
        }))
        (root / "final_fit/artifact_manifest.json").write_text(json.dumps({
            "status": "verified",
            "algorithm": "sha256",
            "artifacts": {
                "adapter_model.safetensors": {"bytes": adapter.stat().st_size, "sha256": adapter_sha},
                "recurrent_state.pt": {"bytes": state.stat().st_size, "sha256": state_sha},
            },
        }))
        groups = {name: {"n": support} for name, support in FROZEN_GROUP_SUPPORT.items()}
        tasks = sorted({name.split("/", 1)[0] for name in groups})
        checks = {
            "C4_architecture": True,
            "C4_closed": True,
            "C4_support": True,
            "C4_specificity": True,
            "C4_noop": True,
            "C4_same_rule": True,
            "C4_replication": True,
            "all_tasks_present": True,
            "all_protocol_groups_present": True,
            "all_protocol_support_exact": True,
            "full_formal_row_support": True,
            "task_performance_retained": True,
        }
        performance_tasks = set(tasks) - {"diagnose_misconception"}
        (root / "final_eval/summary.json").write_text(json.dumps({
            "stage": "layerwise_unified_final_evaluation",
            "scientific_gate_pass": True,
            "checks": checks,
            "tasks": tasks,
            "metrics_by_group": groups,
            "task_rows": [{"sampled_rows": sum(FROZEN_GROUP_SUPPORT.values())}],
            "performance_retention": {task: {"pass": True} for task in performance_tasks},
            "C4": {
                "clean_n": 80, "D_specific": 0.2, "mean_D_specific": 0.2,
                "noop": 0.0, "same_rule": 0.05, "passing_seeds": 3,
                "replication_gate_pass": True,
                "pair_seeds": [
                    {"pair_seed": seed, "all_pairs_n": 150, "clean_n": 80,
                     "D_specific": 0.2, "noop": 0.0, "same_rule": 0.05,
                     "scientific_gate_pass": True}
                    for seed in (13, 42, 97)
                ],
            },
        }))
        (root / "t3_semantic_judge/t3_semantic_gate.json").write_text(json.dumps({
            "stage": "final_T3_semantic_judge",
            "pass": True,
            "support": 1494,
            "semantic_accuracy": 0.5,
            "retention_ratio": 1.2,
            "judge_backend": "trapi",
            "judge_rubric_sha256": "a" * 64,
        }))

    def test_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            self.assertEqual(validate(root)["status"], "complete")

    def test_boolean_only_eval_summary_cannot_fake_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_eval/summary.json"
            value = json.loads(path.read_text())
            value["metrics_by_group"].pop(next(iter(value["metrics_by_group"])))
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "formal_tasks_and_C4"):
                validate(root)

    def test_wrong_exposure_count_cannot_fake_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_fit/download/job/metrics/completion.json"
            value = json.loads(path.read_text())
            value["causal_pairs_seen"] -= 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "causal_exposure"):
                validate(root)

    def test_promoted_remaining_optimizer_plan_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_fit/download/job/metrics/run_config.json"
            value = json.loads(path.read_text())
            value.update({
                "planned_optimizer_steps": EXPECTED_OPTIMIZER_STEPS - 750,
                "promoted_real_order_source_round": 750,
                "promoted_objective_transition_source_round": 750,
                "distributed_rounds": EXPECTED_OPTIMIZER_STEPS,
                "resume_from_checkpoint": "/remote/checkpoint-round-750/recurrent_state.pt",
                "validation_only_from_distributed_checkpoint": False,
            })
            path.write_text(json.dumps(value))
            self.assertEqual(validate(root)["status"], "complete")

    def test_shortened_optimizer_plan_without_promotion_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_fit/download/job/metrics/run_config.json"
            value = json.loads(path.read_text())
            value["planned_optimizer_steps"] = EXPECTED_OPTIMIZER_STEPS - 750
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "frozen_training_recipe"):
                validate(root)

    def test_noncontiguous_history_cannot_fake_exact_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_fit/download/job/metrics/completion.json"
            value = json.loads(path.read_text())
            value["history_global_positions_contiguous"] = False
            value["exact_history_support"] = False
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "exact_epoch"):
                validate(root)

    def test_training_receipt_hash_must_match_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_fit/download/job/metrics/completion.json"
            value = json.loads(path.read_text())
            value["sha256"]["recurrent_state"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "artifacts_match_training_receipt"):
                validate(root)

    def test_data_source_or_task_count_drift_cannot_fake_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "final_fit/download/job/metrics/final_fit_manifest.json"
            value = json.loads(path.read_text())
            value["counts"]["task/diagnose_misconception"] -= 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "frozen_data_provenance"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
