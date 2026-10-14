from __future__ import annotations

import sys
import unittest
import ast
from pathlib import Path

import torch
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from train_recurrent_student_m4 import (  # noqa: E402
    backward_loss_branch,
    build_lr_scheduler,
    build_epoch_schedule,
    build_repeated_causal_schedule,
    causal_exposure_multiplier,
    causal_lm_loss,
    causal_lm_loss_per_sequence,
    final_decoder_norm,
    audit_history_integrity,
    layerwise_trajectory_loss,
    load_promoted_real_order_manifest,
    load_replicated_data_parallel_adapter,
    mirror_distributed_checkpoint,
    norm_matched_random_state,
    pad_supervised_batches,
    paired_preference_loss,
    paired_preference_split_coefficients,
    prune_distributed_checkpoints,
    publish_file,
    recurrent_objective_active,
    specificity_triplet_loss,
    specificity_triplet_branch_losses,
    seed_process_rngs,
    summarize_history,
    validate_distributed_resume_recipe,
)


class M4TrainingTest(unittest.TestCase):
    def test_promoted_real_order_preserves_prefix_and_full_epoch(self) -> None:
        import hashlib
        import json
        import tempfile

        rows = [
            {"example_id": f"row-{index}", "task": "t", "source": "s"}
            for index in range(8)
        ]
        payload = {
            "distributed": True,
            "world_size": 2,
            "round": 2,
            "real_order": list(range(8)),
            "history": [
                {"global_position": index, "example_id": f"row-{index}"}
                for index in range(4)
            ],
        }
        manifest = {
            "stage": "layerwise_task_retention_promoted_order_v1",
            "source_train_sha256": "abc",
            "one_epoch_permutation": True,
            "completed_prefix_unchanged": True,
            "formal_example_id_overlap": 0,
            "source_world_size": 2,
            "source_checkpoint_round": 2,
            "completed_prefix_rows": 4,
            "epoch_rows": 8,
            "promoted_rows": 2,
            "promoted_indices": [6, 4],
            "real_order": [0, 1, 2, 3, 6, 4, 7, 5],
        }
        manifest["real_order_sha256"] = hashlib.sha256(
            "0\n1\n2\n3\n6\n4\n7\n5".encode()
        ).hexdigest()
        manifest["promoted_example_ids_sha256"] = hashlib.sha256(
            "row-6\nrow-4".encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "promotion.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(
                load_promoted_real_order_manifest(
                    path,
                    rows,
                    source_train_sha256="abc",
                    checkpoint_payload=payload,
                ),
                manifest["real_order"],
            )

            continued_payload = dict(payload)
            continued_payload["round"] = 3
            continued_payload["real_order"] = manifest["real_order"]
            continued_payload["history"] = [
                {"global_position": index, "example_id": f"row-{row_index}"}
                for index, row_index in enumerate(manifest["real_order"][:6])
            ]
            continued_payload["config"] = {
                "promoted_real_order_applied": True,
                "promoted_real_order_manifest_sha256": __import__("hashlib").sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
            self.assertEqual(
                load_promoted_real_order_manifest(
                    path,
                    rows,
                    source_train_sha256="abc",
                    checkpoint_payload=continued_payload,
                ),
                manifest["real_order"],
            )

            changed = dict(manifest)
            changed["real_order"] = [1, 0, 2, 3, 6, 4, 7, 5]
            changed["real_order_sha256"] = hashlib.sha256(
                "1\n0\n2\n3\n6\n4\n7\n5".encode()
            ).hexdigest()
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "completed prefix"):
                load_promoted_real_order_manifest(
                    path,
                    rows,
                    source_train_sha256="abc",
                    checkpoint_payload=payload,
                )

    def test_history_integrity_requires_contiguous_unique_coordinates(self) -> None:
        valid = [
            {"global_position": index, "example_id": f"row-{index}"}
            for index in range(4)
        ]
        value = audit_history_integrity(valid)
        self.assertTrue(value["history_integrity_pass"])
        self.assertEqual(value["history_rows"], 4)
        self.assertEqual(value["history_unique_global_positions"], 4)
        self.assertEqual(value["history_unique_examples"], 4)
        self.assertEqual(value["history_global_position_start"], 0)
        self.assertEqual(value["history_global_position_end"], 3)

        segment = [
            {"global_position": index, "example_id": f"row-{index}"}
            for index in range(8_000, 8_004)
        ]
        segment_audit = audit_history_integrity(segment, expected_start=8_000)
        self.assertTrue(segment_audit["history_integrity_pass"])
        self.assertEqual(segment_audit["history_global_position_start"], 8_000)

        missing_position = [valid[0], valid[2], valid[3]]
        self.assertFalse(
            audit_history_integrity(missing_position)[
                "history_global_positions_contiguous"
            ]
        )
        duplicate_example = [dict(row) for row in valid]
        duplicate_example[-1]["example_id"] = "row-0"
        self.assertFalse(
            audit_history_integrity(duplicate_example)["history_integrity_pass"]
        )

    def test_distributed_training_always_checkpoints_the_final_round(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text()
        self.assertIn(
            "completed_round % args.save_steps == 0",
            source,
        )
        self.assertIn("completed_round == total_rounds", source)
        self.assertIn("completed_round in args.checkpoint_milestone_rounds", source)

    def test_exact_resume_completion_counts_the_full_logical_epoch(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text()
        self.assertIn(
            "elif context.enabled and args.resume_from_checkpoint is not None:",
            source,
        )
        self.assertIn("completed_micro_steps = len(history)", source)
        self.assertIn(
            "total_rounds / args.gradient_accumulation_steps",
            source,
        )

    def test_distributed_resume_recipe_allows_only_checkpoint_plumbing(self) -> None:
        checkpoint = {
            "causal_weight": 1.0,
            "state_token_alias": "<|vision_pad|>",
            "train_sha256": "abc",
            "entrypoint_sha256": "old-wrapper",
            "save_steps": 1000,
            "effective_init_adapter": "/pilot/adapter",
            "resume_from_checkpoint": None,
            "distributed_rounds": 16241,
        }
        current = {
            **checkpoint,
            "save_steps": 250,
            "entrypoint_sha256": "recovery-wrapper",
            "checkpoint_retention": 2,
            "effective_init_adapter": "/run/checkpoint-round-15000/adapter",
            "resume_from_checkpoint": Path("/run/checkpoint-round-15000/recurrent_state.pt"),
        }
        current.pop("distributed_rounds")
        validate_distributed_resume_recipe(checkpoint, current)

    def test_distributed_resume_still_rejects_training_view_hash_drift(self) -> None:
        checkpoint = {
            "entrypoint_sha256": "old-wrapper",
            "state8_natural_view_sha256": "science-view-a",
            "train_sha256": "abc",
        }
        current = {
            **checkpoint,
            "entrypoint_sha256": "recovery-wrapper",
            "state8_natural_view_sha256": "science-view-b",
        }
        with self.assertRaisesRegex(ValueError, "scientific recipe drift"):
            validate_distributed_resume_recipe(checkpoint, current)

    def test_distributed_resume_recipe_rejects_scientific_drift(self) -> None:
        checkpoint = {
            "causal_weight": 1.0,
            "state_token_alias": "<|vision_pad|>",
            "train_sha256": "abc",
        }
        for key, value in (
            ("causal_weight", 0.5),
            ("state_token_alias", "<|image_pad|>"),
            ("train_sha256", "def"),
        ):
            current = dict(checkpoint)
            current[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "scientific recipe drift"
            ):
                validate_distributed_resume_recipe(checkpoint, current)

    def test_distributed_resume_recipe_rejects_missing_scientific_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "scientific recipe drift"):
            validate_distributed_resume_recipe(
                {"causal_weight": 1.0, "train_sha256": "abc"},
                {"causal_weight": 1.0},
            )

    def test_checkpoint_retention_prunes_only_old_complete_rounds(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for round_index in (25, 50, 75):
                checkpoint = root / f"checkpoint-round-{round_index}"
                checkpoint.mkdir()
                (checkpoint / "checkpoint_completion.json").write_text("{}\n")
            incomplete = root / "checkpoint-round-100"
            incomplete.mkdir()
            (incomplete / "recurrent_state.pt.partial").write_text("partial")

            prune_distributed_checkpoints(root, retention=2)

            self.assertFalse((root / "checkpoint-round-25").exists())
            self.assertTrue((root / "checkpoint-round-50").exists())
            self.assertTrue((root / "checkpoint-round-75").exists())
            self.assertTrue(incomplete.exists())

    def test_checkpoint_retention_preserves_declared_curriculum_milestones(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for round_index in (100, 200, 300, 400):
                checkpoint = root / f"checkpoint-round-{round_index}"
                checkpoint.mkdir()
                (checkpoint / "checkpoint_completion.json").write_text("{}\n")
            prune_distributed_checkpoints(
                root, retention=1, preserve_rounds=(100, 300)
            )
            self.assertTrue((root / "checkpoint-round-100").exists())
            self.assertFalse((root / "checkpoint-round-200").exists())
            self.assertTrue((root / "checkpoint-round-300").exists())
            self.assertTrue((root / "checkpoint-round-400").exists())

    def test_checkpoint_mirror_publishes_verified_receipt_last(self) -> None:
        import hashlib
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source/checkpoint-round-3250"
            (source / "adapter").mkdir(parents=True)
            state = source / "recurrent_state.pt"
            adapter = source / "adapter/adapter_model.safetensors"
            state.write_bytes(b"state")
            adapter.write_bytes(b"adapter")
            receipt = {
                "status": "complete",
                "round": 3250,
                "world_size": 4,
                "sha256": {
                    "adapter_model": hashlib.sha256(b"adapter").hexdigest(),
                    "recurrent_state": hashlib.sha256(b"state").hexdigest(),
                },
            }
            (source / "checkpoint_completion.json").write_text(json.dumps(receipt))
            destination = mirror_distributed_checkpoint(source, root / "mirror")
            self.assertEqual(
                json.loads((destination / "checkpoint_completion.json").read_text()),
                receipt,
            )
            self.assertFalse(any((root / "mirror").glob("*.partial*")))

    def test_publish_file_fsyncs_and_atomically_removes_partial(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / "state.pt.partial"
            final = root / "state.pt"
            partial.write_bytes(b"durable-state")
            publish_file(partial, final)
            self.assertEqual(final.read_bytes(), b"durable-state")
            self.assertFalse(partial.exists())

    def test_negative_checkpoint_retention_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            prune_distributed_checkpoints(Path("unused"), retention=-1)

    def test_layerwise_trajectory_loss_preserves_direction_and_scale(self) -> None:
        direct = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[2.0, -1.0], [1.0, -3.0]]]
        )
        exact = direct.clone().requires_grad_(True)
        loss, cosine, relative_mse = layerwise_trajectory_loss(
            exact,
            direct,
            cosine_weight=1.0,
            relative_mse_weight=1.0,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)
        torch.testing.assert_close(cosine, torch.tensor(0.0), atol=1e-7, rtol=0)
        torch.testing.assert_close(relative_mse, torch.tensor(0.0))

        scaled = (2.0 * direct).requires_grad_(True)
        scaled_loss, scaled_cosine, scaled_relative_mse = layerwise_trajectory_loss(
            scaled,
            direct,
            cosine_weight=1.0,
            relative_mse_weight=1.0,
        )
        torch.testing.assert_close(scaled_cosine, torch.tensor(0.0), atol=1e-7, rtol=0)
        torch.testing.assert_close(scaled_relative_mse, torch.tensor(1.0))
        self.assertGreater(float(scaled_loss), 0.9)
        scaled_loss.backward()
        self.assertGreater(float(scaled.grad.abs().sum()), 0.0)

    def test_layerwise_trajectory_loss_rejects_unaligned_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            layerwise_trajectory_loss(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 3, 3),
                cosine_weight=1.0,
                relative_mse_weight=1.0,
            )

    def test_layerwise_pilot_warm_start_restores_recurrent_state(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("args.init_recurrent_state = run_state", source)
        self.assertIn(
            "if args.init_recurrent_state is not None:\n"
            "            initial_payload = torch.load(",
            source,
        )
        self.assertIn("load_recurrent_state_payload(wrapper, initial_payload)", source)

    def test_distributed_seed_targets_only_the_local_cuda_device(self) -> None:
        with (
            patch("train_recurrent_student_m4.random.seed") as python_seed,
            patch("train_recurrent_student_m4.torch.random.default_generator") as cpu_generator,
            patch("train_recurrent_student_m4.torch.cuda.set_device") as set_device,
            patch("train_recurrent_student_m4.torch.cuda.manual_seed") as cuda_seed,
            patch("train_recurrent_student_m4.torch.manual_seed") as all_device_seed,
        ):
            seed_process_rngs(47, torch.device("cuda:3"))
        python_seed.assert_called_once_with(47)
        cpu_generator.manual_seed.assert_called_once_with(47)
        set_device.assert_called_once_with(torch.device("cuda:3"))
        cuda_seed.assert_called_once_with(47)
        all_device_seed.assert_not_called()

    def test_layerwise_training_disables_hook_unsafe_checkpointing(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'use_gradient_checkpointing = args.state_architecture != "layerwise"',
            source,
        )
        self.assertNotIn("_layerwise_non_reentrant_checkpointing", source)
        self.assertIn(
            '"Layerwise training requires gradient checkpointing disabled"',
            source,
        )
        self.assertIn("--layerwise-transition-only", source)
        self.assertIn("--layerwise-operation-checkpointing", source)
        self.assertIn(
            "operation_checkpointing=args.layerwise_operation_checkpointing",
            source,
        )
        self.assertIn("wrapper.transition_in.to(device)", source)
        self.assertIn("parameter.requires_grad_(False)", source)

    def test_direct_token_count_is_defined_before_offload_and_receipt(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "backward_training_objective"
        )
        objective = ast.get_source_segment(source, node)
        self.assertIsNotNone(objective)
        definition = objective.index(
            'direct_token_count = int(direct["input_ids"].shape[1])'
        )
        offload_use = objective.index(
            "direct_token_count >= args.direct_offload_min_tokens"
        )
        receipt_use = objective.index('"direct_tokens": int(direct_token_count)')
        self.assertLess(definition, offload_use)
        self.assertLess(definition, receipt_use)

    def test_cold_start_layerwise_row_skips_undefined_consistency(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if state is None:", source)
        self.assertIn(
            "consistency_loss = teacher_state.new_zeros",
            source,
        )

    def test_post_training_source_closes_nccl_before_long_validation(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        gather = source.index(
            "full_history = gather_distributed_history(history, history_base, context)"
        )
        validation = source.index("validation = evaluate(", gather)
        close_group = source.index("close(context)", gather, validation)
        self.assertLess(close_group, validation)
        self.assertNotIn("barrier(context)", source[validation:])

    def test_completed_checkpoint_can_enter_validation_only_recovery(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "args.diagnostic_stop_after_round < start_round",
            source,
        )
        self.assertNotIn(
            "args.diagnostic_stop_after_round <= start_round",
            source,
        )

    def test_full_fit_can_warm_start_from_completed_diagnostic_pilot(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'pilot_completion.get("status") not in {"complete", "diagnostic_complete"}',
            source,
        )
        self.assertIn('args.init_adapter = str(run_adapter)', source)

    def test_diagnostic_warm_start_is_not_exact_resume(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--diagnostic-warm-start-checkpoint", source)
        self.assertIn("--diagnostic-start-global-position", source)
        self.assertIn("Exact resume and diagnostic warm start are mutually exclusive", source)
        self.assertIn('config["diagnostic_optimizer_reset"] = True', source)
        self.assertIn("expected_start = warm_world_size * warm_round", source)
        self.assertIn("Diagnostic warm-start compatibility drift", source)
        self.assertIn("Diagnostic warm-start real-data order drift", source)
        self.assertIn("completed_rounds = total_rounds - start_round", source)
        self.assertIn("segment_rounds = total_rounds - start_round", source)
        self.assertIn('config["diagnostic_segment_rounds"]', source)
        warm_start = source.index("if args.diagnostic_warm_start_checkpoint is not None:")
        resume = source.index("if args.resume_from_checkpoint:", warm_start)
        self.assertLess(warm_start, resume)

    def test_diagnostic_warm_start_accepts_same_state_capacity(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "warm_state_count != args.state_token_count",
            source,
        )
        self.assertIn(
            "State-token warm start must preserve capacity or widen a legacy one-slot source",
            source,
        )

    def test_single_gpu_validation_only_checkpoint_path_skips_optimizer(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--validation-only-from-distributed-checkpoint", source)
        self.assertIn(
            'raise ValueError("Validation-only checkpoint finalization must be single-GPU")',
            source,
        )
        start = source.index(
            "if args.validation_only_from_distributed_checkpoint:\n"
            "        if resume_payload is None:"
        )
        training = source.index("elif context.enabled:", start)
        self.assertNotIn("optimizer.step()", source[start:training])
        self.assertIn("full_history = list(history)", source[start:training])

    def test_complete_epoch_finalization_requires_exact_final_round(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text()
        self.assertIn("--finalize-complete-distributed-epoch", source)
        self.assertIn(
            "Complete distributed-epoch finalization requires validation-only recovery",
            source,
        )
        self.assertIn(
            "checkpoint_round != expected_final_round",
            source,
        )
        self.assertIn(
            "and not args.finalize_complete_distributed_epoch",
            source,
        )

    def test_masked_causal_lm_loss_matches_ignore_index_gradient(self) -> None:
        labels = torch.tensor([[-100, -100, 2, 1], [-100, 3, -100, 0]])
        baseline_logits = torch.randn(2, 4, 5, requires_grad=True)
        expected_loss = torch.nn.functional.cross_entropy(
            baseline_logits[:, :-1].float().reshape(-1, 5),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        expected_loss.backward()
        expected_gradient = baseline_logits.grad.detach().clone()

        selected_logits = baseline_logits.detach().clone().requires_grad_(True)
        actual_loss = causal_lm_loss(selected_logits, labels)
        actual_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss.detach())
        torch.testing.assert_close(selected_logits.grad, expected_gradient)

    def test_masked_causal_lm_loss_rejects_empty_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "no supervised target"):
            causal_lm_loss(torch.randn(1, 3, 5), torch.full((1, 3), -100))

    def test_per_sequence_masked_loss_matches_ignore_index_gradient(self) -> None:
        labels = torch.tensor([[-100, -100, 2, 1], [-100, 3, -100, 0]])
        baseline_logits = torch.randn(2, 4, 5, requires_grad=True)
        shifted_labels = labels[:, 1:]
        token_loss = torch.nn.functional.cross_entropy(
            baseline_logits[:, :-1].float().reshape(-1, 5),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels)
        valid = shifted_labels.ne(-100)
        expected_loss = (token_loss * valid).sum(dim=-1) / valid.sum(dim=-1)
        expected_loss.sum().backward()
        expected_gradient = baseline_logits.grad.detach().clone()

        selected_logits = baseline_logits.detach().clone().requires_grad_(True)
        actual_loss = causal_lm_loss_per_sequence(selected_logits, labels)
        actual_loss.sum().backward()
        torch.testing.assert_close(actual_loss, expected_loss.detach())
        torch.testing.assert_close(selected_logits.grad, expected_gradient)

    def test_per_sequence_sum_matches_frozen_sequence_logprob_scale(self) -> None:
        labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 0, -100]])
        logits = torch.randn(2, 4, 5)
        mean = causal_lm_loss_per_sequence(logits, labels, reduction="mean")
        total = causal_lm_loss_per_sequence(logits, labels, reduction="sum")
        counts = labels[:, 1:].ne(-100).sum(dim=-1)
        torch.testing.assert_close(total, mean * counts)
        with self.assertRaisesRegex(ValueError, "Unknown per-sequence reduction"):
            causal_lm_loss_per_sequence(logits, labels, reduction="median")

    def test_final_decoder_norm_resolves_base_model_norm(self) -> None:
        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.norm = torch.nn.LayerNorm(2)

        class Adapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = Backbone()

            def get_base_model(self):
                return self.base

        adapter = Adapter()
        self.assertIs(final_decoder_norm(adapter), adapter.base.model.norm)

    def test_final_decoder_norm_fails_closed_for_unknown_backbone(self) -> None:
        with self.assertRaisesRegex(ValueError, "final decoder norm"):
            final_decoder_norm(torch.nn.Linear(2, 2))

    def test_branchwise_backward_matches_weighted_sum_gradient(self) -> None:
        combined = torch.nn.Parameter(torch.tensor([0.3, -0.4]))
        combined_loss = (
            1.0 * (combined.square().sum())
            + 0.5 * ((combined + 1).square().sum())
            + 0.25 * ((2 * combined - 1).square().sum())
        )
        combined_loss.backward()
        expected = combined.grad.detach().clone()

        split = torch.nn.Parameter(torch.tensor([0.3, -0.4]))
        backward_loss_branch(split.square().sum(), 1.0, 1.0)
        backward_loss_branch((split + 1).square().sum(), 0.5, 1.0)
        backward_loss_branch((2 * split - 1).square().sum(), 0.25, 1.0)
        torch.testing.assert_close(split.grad, expected)

    def test_branchwise_backward_applies_accumulation_scale(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        backward_loss_branch(parameter.square(), 0.5, 0.25)
        self.assertAlmostEqual(float(parameter.grad), 0.5)

    def test_zero_weight_detached_branch_is_an_exact_noop(self) -> None:
        backward_loss_branch(torch.tensor(0.0), 0.0, 1.0)

    def test_active_detached_branch_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Active loss branch is detached"):
            backward_loss_branch(torch.tensor(0.0), 1.0, 1.0)

    def test_teacher_only_cold_start_has_no_recurrent_backward_branch(self) -> None:
        self.assertFalse(
            recurrent_objective_active(
                recurrent_state_present=False,
                generation_weight=0.0,
                behavior_kd_weight=0.0,
                online_kd_weight=0.5,
                online_kd_phase_weight=0.0,
                consistency_weight=0.0,
            )
        )
        self.assertTrue(
            recurrent_objective_active(
                recurrent_state_present=False,
                generation_weight=0.25,
                behavior_kd_weight=0.0,
                online_kd_weight=0.5,
                online_kd_phase_weight=1.0,
                consistency_weight=0.0,
            )
        )
        self.assertFalse(
            recurrent_objective_active(
                recurrent_state_present=False,
                generation_weight=0.0,
                behavior_kd_weight=0.0,
                online_kd_weight=0.0,
                online_kd_phase_weight=0.0,
                consistency_weight=1.0,
            )
        )

    def test_replicated_adapter_load_disables_only_peft_tp_hook(self) -> None:
        model = torch.nn.Linear(2, 2)
        sentinel = object()
        original = __import__(
            "peft.utils.save_and_load", fromlist=["_maybe_shard_state_dict_for_tp"]
        )._maybe_shard_state_dict_for_tp
        with patch(
            "train_recurrent_student_m4.PeftModel.from_pretrained",
            return_value=sentinel,
        ) as loader:
            result = load_replicated_data_parallel_adapter(model, Path("adapter"))
        self.assertIs(result, sentinel)
        loader.assert_called_once_with(model, "adapter", is_trainable=True)
        current = __import__(
            "peft.utils.save_and_load", fromlist=["_maybe_shard_state_dict_for_tp"]
        )._maybe_shard_state_dict_for_tp
        self.assertIs(current, original)

    def test_replicated_adapter_load_rejects_real_tp_mesh(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        model[0]._hf_device_mesh = object()
        with self.assertRaisesRegex(ValueError, "tensor-parallel device mesh"):
            load_replicated_data_parallel_adapter(model, Path("adapter"))

    def test_cosine_scheduler_warms_up_then_decays(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = build_lr_scheduler(
            optimizer, schedule="cosine", total_steps=10, warmup_ratio=0.2
        )
        values = []
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])
        self.assertGreater(values[1], values[0])
        self.assertGreater(values[1], values[-1])
        self.assertAlmostEqual(values[-1], 0.0)

    def test_epoch_schedule_exposes_every_row_and_pair_once(self) -> None:
        order, schedule = build_epoch_schedule(12_000, 1_726, seed=42)
        self.assertEqual(len(order), 12_000)
        self.assertEqual(set(order), set(range(12_000)))
        self.assertEqual(len(schedule), 1_726)
        self.assertEqual(set(schedule.values()), set(range(1_726)))
        self.assertEqual(len(set(schedule)), 1_726)
        self.assertIn(0, schedule)

    def test_repeated_causal_schedule_preserves_every_pair_at_fixed_fraction(self) -> None:
        schedule = build_repeated_causal_schedule(259_845, 1_726, 0.10, seed=42)
        self.assertEqual(len(schedule), round(259_845 * 0.10))
        self.assertEqual(set(schedule.values()), set(range(1_726)))
        self.assertEqual(len(set(schedule)), len(schedule))
        self.assertEqual(schedule, build_repeated_causal_schedule(259_845, 1_726, 0.10, seed=42))

    def test_causal_exposure_normalization_is_world_size_independent(self) -> None:
        self.assertEqual(causal_exposure_multiplier(12_000, 1_500, False), 1.0)
        self.assertEqual(causal_exposure_multiplier(12_000, 1_500, True), 8.0)
        self.assertAlmostEqual(
            0.05 * causal_exposure_multiplier(259_845, 12_992, True),
            1.0,
            places=3,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            causal_exposure_multiplier(12_000, 0, True)

    def test_paired_preference_penalizes_wrong_direction(self) -> None:
        good, good_margin = paired_preference_loss(
            torch.tensor(0.1),
            torch.tensor(2.0),
            torch.tensor(0.1),
            torch.tensor(2.0),
            0.5,
        )
        bad, bad_margin = paired_preference_loss(
            torch.tensor(2.0),
            torch.tensor(0.1),
            torch.tensor(2.0),
            torch.tensor(0.1),
            0.5,
        )
        self.assertEqual(float(good_margin), 0.0)
        self.assertGreater(float(bad_margin), 0.5)
        self.assertGreater(float(bad), float(good))

    def test_asymmetric_directional_bootstrap_breaks_collapsed_state_cancellation(self) -> None:
        shared_preference = torch.tensor(0.0, requires_grad=True)
        symmetric, _ = paired_preference_loss(
            torch.tensor(0.0),
            shared_preference,
            torch.tensor(0.0),
            -shared_preference,
            0.5,
        )
        symmetric.backward()
        self.assertEqual(float(shared_preference.grad), 0.0)

        shared_preference = torch.tensor(0.0, requires_grad=True)
        bootstrap, _ = paired_preference_loss(
            torch.tensor(0.0),
            shared_preference,
            torch.tensor(0.0),
            -shared_preference,
            0.5,
            objective_mode="directional",
            native_margin_weight=0.25,
            donor_margin_weight=1.0,
        )
        bootstrap.backward()
        self.assertGreater(abs(float(shared_preference.grad)), 0.5)

    def test_directional_objective_requires_a_positive_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one directional"):
            paired_preference_loss(
                torch.tensor(0.1),
                torch.tensor(0.2),
                torch.tensor(0.1),
                torch.tensor(0.2),
                0.5,
                objective_mode="directional",
                native_margin_weight=0.0,
                donor_margin_weight=0.0,
            )

    def test_balanced_directional_tail_cannot_hide_native_failure_with_donor_success(self) -> None:
        symmetric, symmetric_margin = paired_preference_loss(
            torch.tensor(0.0),
            torch.tensor(-0.7),
            torch.tensor(0.0),
            torch.tensor(0.7),
            0.5,
            objective_mode="symmetric",
        )
        directional, directional_margin = paired_preference_loss(
            torch.tensor(0.0),
            torch.tensor(-0.7),
            torch.tensor(0.0),
            torch.tensor(0.7),
            0.5,
            objective_mode="directional",
        )
        self.assertAlmostEqual(float(symmetric_margin), 0.5, places=6)
        self.assertAlmostEqual(float(directional_margin), 0.6, places=6)
        self.assertGreater(float(directional), float(symmetric))

    def test_bootstrap_then_directional_is_registered_as_a_training_mode(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"bootstrap_then_directional"', source)
        self.assertIn('"directional_bootstrap"', source)
        self.assertIn('"directional_balanced"', source)

    def test_donor_directional_keeps_gold_ce_and_avoids_opposed_margin(self) -> None:
        native_gold = torch.tensor(0.2, requires_grad=True)
        native_donor = torch.tensor(0.2, requires_grad=True)
        donor_gold = torch.tensor(0.4, requires_grad=True)
        donor_native = torch.tensor(0.4, requires_grad=True)
        loss, margin = paired_preference_loss(
            native_gold,
            native_donor,
            donor_gold,
            donor_native,
            0.5,
            objective_mode="directional",
            native_margin_weight=0.0,
            donor_margin_weight=1.0,
        )
        loss.backward()
        self.assertAlmostEqual(float(margin), 0.5, places=6)
        # Native gold CE remains active even though the native preference hinge
        # is disabled; the donor margin supplies the non-cancelling direction.
        self.assertAlmostEqual(float(native_gold.grad), 0.5, places=6)
        self.assertGreater(float(donor_gold.grad), 0.0)
        self.assertLess(float(donor_native.grad), 0.0)
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('args.causal_objective_mode == "donor_directional"', source)

    def test_two_view_paired_split_matches_joint_value_and_gradient(self) -> None:
        for mode, native_weight, donor_weight in (
            ("symmetric", 1.0, 1.0),
            ("directional", 1.0, 1.0),
            ("directional", 0.0, 1.0),
        ):
            joint = torch.tensor([0.2, 0.8, 0.3, 0.7], requires_grad=True)
            joint_loss, joint_margin = paired_preference_loss(
                *joint,
                0.5,
                objective_mode=mode,
                native_margin_weight=native_weight,
                donor_margin_weight=donor_weight,
            )
            joint_loss.backward()
            expected_gradient = joint.grad.detach().clone()

            split = torch.tensor([0.2, 0.8, 0.3, 0.7], requires_grad=True)
            coefficients = paired_preference_split_coefficients(
                float(split[1] - split[0]),
                float(split[3] - split[2]),
                0.5,
                objective_mode=mode,
                native_margin_weight=native_weight,
                donor_margin_weight=donor_weight,
            )
            nc, nk, dc, dk = coefficients
            native_advantage = split[1] - split[0]
            donor_advantage = split[3] - split[2]
            native_loss = 0.5 * split[0] + nc * native_advantage + nk
            donor_loss = 0.5 * split[2] + dc * donor_advantage + dk
            native_loss.backward(retain_graph=True)
            donor_loss.backward()
            self.assertAlmostEqual(
                float(native_loss + donor_loss), float(joint_loss), places=6
            )
            expected_margin = float(joint_margin)
            self.assertAlmostEqual(
                float(native_loss + donor_loss - 0.5 * (split[0] + split[2])),
                expected_margin,
                places=6,
            )
            torch.testing.assert_close(split.grad, expected_gradient)

    def test_paired_training_uses_two_view_memory_safe_backward(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def backward_paired_causal_objective")
        end = source.index("def backward_specificity_triplet_objective", start)
        body = source[start:end]
        self.assertIn('"causal_backward_graph_max_views": 2', body)
        self.assertIn("backward_loss_branch(native_loss", body)
        self.assertIn("backward_loss_branch(donor_loss", body)
        self.assertIn('"causal_dropout_mode": "eval_deterministic"', body)
        self.assertIn('"causal_branch_dropout_mode": "eval_deterministic"', source)

    def test_specificity_triplet_matches_swap_minus_random_direction(self) -> None:
        # Native prefers B, donor prefers A, and random prefers B: no hinge.
        good, good_margin, diagnostics = specificity_triplet_loss(
            torch.tensor(0.1),
            torch.tensor(1.1),
            torch.tensor(0.1),
            torch.tensor(1.1),
            torch.tensor(0.1),
            torch.tensor(1.1),
            0.5,
        )
        self.assertEqual(float(good_margin), 0.0)
        self.assertAlmostEqual(float(diagnostics["specificity_advantage"]), 2.0)

        # Donor still prefers A, but random also flips to A.  This is precisely
        # the round-400 failure (swap == random) and must incur a penalty.
        bad, bad_margin, bad_diagnostics = specificity_triplet_loss(
            torch.tensor(0.1),
            torch.tensor(1.1),
            torch.tensor(0.1),
            torch.tensor(1.1),
            torch.tensor(1.1),
            torch.tensor(0.1),
            0.5,
        )
        self.assertGreater(float(bad_margin), 0.0)
        self.assertEqual(float(bad_diagnostics["specificity_advantage"]), 0.0)
        self.assertGreater(float(bad), float(good))

    def test_specificity_random_state_matches_frozen_control_and_is_detached(self) -> None:
        reference = torch.tensor([[3.0, 4.0]], requires_grad=True)
        first = norm_matched_random_state(reference, 7)
        second = norm_matched_random_state(reference, 7)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            torch.linalg.vector_norm(first),
            torch.linalg.vector_norm(reference.detach()),
        )
        self.assertFalse(first.requires_grad)

    def test_specificity_triplet_is_registered_as_a_training_mode(self) -> None:
        source = (SCRIPT_DIR / "train_recurrent_student_m4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"specificity_triplet"', source)
        self.assertIn("causal_random_native_advantage", source)
        self.assertIn("causal_specificity_advantage", source)
        self.assertIn('"causal_backward_graph_max_views": 2', source)

    def test_memory_safe_specificity_split_matches_joint_value_and_gradient(self) -> None:
        joint_values = torch.tensor(
            [0.2, 0.8, 0.3, 0.7, 0.9, 0.1], requires_grad=True
        )
        joint, joint_margin, _ = specificity_triplet_loss(
            *joint_values,
            0.5,
        )
        joint.backward()
        joint_gradient = joint_values.grad.detach().clone()

        split_values = torch.tensor(
            [0.2, 0.8, 0.3, 0.7, 0.9, 0.1], requires_grad=True
        )
        native_loss, donor_loss, random_loss, split_margin, _ = (
            specificity_triplet_branch_losses(
                split_values[:2],
                split_values[:2],
                split_values[2:4],
                split_values[2:4],
                split_values[4:],
                split_values[4:],
                0.5,
            )
        )
        native_loss.backward(retain_graph=True)
        donor_loss.backward(retain_graph=True)
        random_loss.backward()
        self.assertAlmostEqual(
            float(native_loss + donor_loss + random_loss), float(joint), places=6
        )
        self.assertAlmostEqual(float(split_margin), float(joint_margin), places=6)
        torch.testing.assert_close(split_values.grad, joint_gradient)

    def test_sum_preference_does_not_rescale_gold_ce(self) -> None:
        mean_losses = tuple(torch.tensor(value) for value in (0.2, 0.2, 0.4, 0.4))
        summed_losses = tuple(torch.tensor(value) for value in (2.0, 2.6, 4.0, 4.8))
        loss, margin = paired_preference_loss(
            *mean_losses,
            0.5,
            preference_nlls=summed_losses,
        )
        self.assertAlmostEqual(float(margin), 0.0, places=6)
        self.assertAlmostEqual(float(loss), 0.3, places=6)

    def test_summary_keeps_real_tasks_and_causal_loss_separate(self) -> None:
        row = {
            "task": "predict_correctness",
            "source": "dbe_kt22",
            "loss": 1.0,
            "generation_loss": 0.5,
            "behavior_kd_loss": 0.1,
            "consistency_loss": 0.2,
            "direct_loss": 0.3,
            "causal_loss": 0.4,
            "causal_margin_loss": 0.25,
            "causal_objective_phase": "directional",
            "causal_native_advantage": 0.1,
            "causal_donor_advantage": 0.3,
            "causal_preference": 0.2,
            "causal_state_cosine": 0.9,
            "causal_state_delta_norm": 2.0,
            "causal_state_relative_delta": 0.1,
            "causal_pair_id": "pair-1",
        }
        result = summarize_history([row])
        self.assertEqual(result["by_task_source"]["predict_correctness/dbe_kt22"]["n"], 1)
        self.assertEqual(result["causal"]["n"], 1)
        self.assertEqual(result["causal"]["mean_causal_margin_loss"], 0.25)
        self.assertEqual(result["causal"]["mean_causal_state_cosine"], 0.9)
        self.assertEqual(result["causal"]["objective_phase_counts"], {"directional": 1})

    def test_padded_candidate_batch_preserves_per_sequence_targets(self) -> None:
        batches = [
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "labels": torch.tensor([[-100, 2, 3]]),
                "prompt_ends": torch.tensor([1]),
            },
            {
                "input_ids": torch.tensor([[1, 4]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "labels": torch.tensor([[-100, 4]]),
                "prompt_ends": torch.tensor([1]),
            },
        ]
        combined = pad_supervised_batches(batches, pad_token_id=0)
        self.assertEqual(tuple(combined["input_ids"].shape), (2, 3))
        self.assertEqual(int(combined["attention_mask"][1, 2]), 0)
        self.assertEqual(int(combined["labels"][1, 2]), -100)
        logits = torch.zeros(2, 3, 5)
        losses = causal_lm_loss_per_sequence(logits, combined["labels"])
        self.assertEqual(tuple(losses.shape), (2,))


if __name__ == "__main__":
    unittest.main()
