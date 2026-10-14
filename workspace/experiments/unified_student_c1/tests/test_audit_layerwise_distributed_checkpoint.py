from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_layerwise_distributed_checkpoint import (  # noqa: E402
    EXPECTED_CAUSAL_EXPOSURES,
    EXPECTED_CAUSAL_PAIRS,
    EXPECTED_CAUSAL_TRAIN_SHA256,
    EXPECTED_CAUSAL_VALID_SHA256,
    EXPECTED_ROWS,
    EXPECTED_TOTAL_ROUNDS,
    EXPECTED_TRAIN_SHA256,
    EXPECTED_VALID_SHA256,
    EXPECTED_WORLD_SIZE,
    audit,
)


class DistributedCheckpointAuditTest(unittest.TestCase):
    def payload(self, checkpoint_round: int = 2) -> dict:
        return {
            "distributed": True,
            "world_size": EXPECTED_WORLD_SIZE,
            "round": checkpoint_round,
            "optimizer": {"state": {0: {"step": 2}}, "param_groups": [{"lr": 1e-6}]},
            "scheduler": {"last_epoch": checkpoint_round},
            "rank_rng_states": [
                {
                    "rank": rank,
                    "python_random_state": (3, (), None),
                    "torch_random_state": torch.zeros(1, dtype=torch.uint8),
                    "cuda_random_state": torch.zeros(1, dtype=torch.uint8),
                }
                for rank in range(EXPECTED_WORLD_SIZE)
            ],
            "real_order": list(range(EXPECTED_ROWS)),
            "causal_schedule": {
                position: position % EXPECTED_CAUSAL_PAIRS
                for position in range(EXPECTED_CAUSAL_EXPOSURES)
            },
            "history": [
                {
                    "global_position": position,
                    "causal_pair_id": (
                        f"recurrent-causal:{position % EXPECTED_CAUSAL_PAIRS:07d}"
                        if position < EXPECTED_CAUSAL_EXPOSURES
                        else None
                    ),
                    "causal_loss": 0.5 if position < EXPECTED_CAUSAL_EXPOSURES else 0.0,
                    "causal_effective_weight": (
                        20.0 if position < EXPECTED_CAUSAL_EXPOSURES else 0.0
                    ),
                    "causal_objective_phase": (
                        "specificity_triplet"
                        if position < EXPECTED_CAUSAL_EXPOSURES
                        else None
                    ),
                }
                for position in range(checkpoint_round * EXPECTED_WORLD_SIZE)
            ],
            "config": {
                "state_architecture": "layerwise",
                "layerwise_operation_checkpointing": True,
                "layerwise_transition_rank": 256,
                "distributed": True,
                "distributed_world_size": EXPECTED_WORLD_SIZE,
                "real_train_rows": EXPECTED_ROWS,
                "planned_optimizer_steps": EXPECTED_TOTAL_ROUNDS,
                "causal_train_pairs": EXPECTED_CAUSAL_PAIRS,
                "planned_causal_exposures": EXPECTED_CAUSAL_EXPOSURES,
                "causal_objective_mode": "specificity_triplet",
                "causal_repeat_fraction": 0.05,
                "normalize_causal_exposure": True,
                "state_token_alias": "<|vision_pad|>",
                "causal_branch_dropout_mode": "eval_deterministic",
                "train_sha256": EXPECTED_TRAIN_SHA256,
                "valid_sha256": EXPECTED_VALID_SHA256,
                "causal_train_sha256": EXPECTED_CAUSAL_TRAIN_SHA256,
                "causal_valid_sha256": EXPECTED_CAUSAL_VALID_SHA256,
                "seed": 42,
            },
        }

    def write(self, root: Path, payload: dict) -> Path:
        path = root / "recurrent_state.pt"
        torch.save(payload, path)
        return path

    def test_complete_checkpoint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), self.payload())
            result = audit(path, expected_round=2)
            self.assertTrue(result["exact_resume_ready"])
            self.assertEqual(result["planned_optimizer_steps_semantics"], "full_epoch_total")

    def test_promoted_remaining_step_plan_passes_with_strict_resume_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload(checkpoint_round=2)
            payload["config"].update(
                {
                    "planned_optimizer_steps": EXPECTED_TOTAL_ROUNDS - 1,
                    "promoted_real_order_source_round": 1,
                    "promoted_objective_transition_source_round": 1,
                    "distributed_rounds": EXPECTED_TOTAL_ROUNDS,
                    "resume_from_checkpoint": "/remote/checkpoint-round-1/recurrent_state.pt",
                    "validation_only_from_distributed_checkpoint": False,
                }
            )
            path = self.write(Path(temporary), payload)
            result = audit(path, expected_round=2)
            self.assertEqual(
                result["planned_optimizer_steps_semantics"],
                "remaining_after_promoted_exact_resume",
            )
            self.assertEqual(result["continuation_source_round"], 1)

    def test_shortened_step_plan_without_resume_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload(checkpoint_round=2)
            payload["config"]["planned_optimizer_steps"] = EXPECTED_TOTAL_ROUNDS - 1
            path = self.write(Path(temporary), payload)
            with self.assertRaisesRegex(ValueError, "planned_optimizer_steps"):
                audit(path, expected_round=2)

    def test_promoted_step_plan_with_disagreeing_source_rounds_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload(checkpoint_round=2)
            payload["config"].update(
                {
                    "planned_optimizer_steps": EXPECTED_TOTAL_ROUNDS - 1,
                    "promoted_real_order_source_round": 1,
                    "promoted_objective_transition_source_round": 2,
                    "distributed_rounds": EXPECTED_TOTAL_ROUNDS,
                    "resume_from_checkpoint": "/remote/checkpoint-round-1/recurrent_state.pt",
                }
            )
            path = self.write(Path(temporary), payload)
            with self.assertRaisesRegex(ValueError, "continuation_evidence"):
                audit(path, expected_round=2)

    def test_missing_rank_rng_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload()
            payload["rank_rng_states"].pop()
            path = self.write(Path(temporary), payload)
            with self.assertRaisesRegex(ValueError, "RNG support"):
                audit(path, expected_round=2)

    def test_scientific_recipe_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload()
            payload["config"]["causal_weight"] = 0.5
            # Add the frozen expected value, then mutate it to exercise a field
            # that is actually audited.
            payload["config"]["state_token_alias"] = "<|image_pad|>"
            path = self.write(Path(temporary), payload)
            with self.assertRaisesRegex(ValueError, "frozen training config drift"):
                audit(path, expected_round=2)

    def test_non_permutation_real_order_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload()
            payload["real_order"][-1] = 0
            path = self.write(Path(temporary), payload)
            with self.assertRaisesRegex(ValueError, "not a permutation"):
                audit(path, expected_round=2)

    def test_missing_causal_activation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.payload()
            payload["history"][0]["causal_pair_id"] = None
            path = self.write(Path(temporary), payload)
            with self.assertRaisesRegex(ValueError, "causal pair mismatch"):
                audit(path, expected_round=2)



if __name__ == "__main__":
    unittest.main()
