from __future__ import annotations

import unittest

import torch

from scripts.t3_judge_utils import natural_schema_valid
from scripts.train_state8_semantic_causal_repair import (
    NEUTRAL_TARGET,
    aggregate_early_label_counts,
    aggregate_geometry_counts,
    backward_updater_only_specificity,
    backward_early_label_controls,
    control_random_seeds,
    early_label_decode_text,
    early_binding_labels,
    early_label_target,
    is_early_label_pair,
    is_eedi_semantic_pair,
    random_neutral_two_view_branch_loss,
    recipient_control_two_view_loss,
    schema_weighted_per_sequence_factory,
    semantic_state_geometry_loss,
)


class TinyTokenizer:
    mapping = {"<label>": [10, 11], "</label>": [12, 13]}

    def __call__(self, value, add_special_tokens=False):
        return {"input_ids": self.mapping[value]}


class SemanticCausalRepairTest(unittest.TestCase):
    def test_early_label_readout_keeps_closed_state_query_without_gold_prefix(self) -> None:
        source = (
            "Old output instruction.\n<STUDENT_STATE>\n<query>same item</query>\nAnswer:"
        )
        # Pin the legacy/current-query protocol explicitly so this unit test is
        # not changed by a state-only experiment's process environment.
        prompt = early_label_decode_text(source, "with_query")
        self.assertEqual(prompt.count("<STUDENT_STATE>"), 1)
        self.assertIn("<query>same item</query>", prompt)
        self.assertNotIn("Old output instruction", prompt)
        self.assertNotIn("<student_history>", prompt.lower())
        self.assertTrue(prompt.endswith("Canonical misconception label:"))
        self.assertEqual(early_label_target("2016"), "2016")
        with self.assertRaisesRegex(ValueError, "Invalid early-label target"):
            early_label_target("<label>2016</label>")

    def test_state_only_label_readout_removes_current_query(self) -> None:
        source = (
            "Old output instruction.\n<STUDENT_STATE>\n"
            "<query>same item</query>\nAnswer:"
        )
        prompt = early_label_decode_text(source, "state_only")
        self.assertEqual(prompt.count("<STUDENT_STATE>"), 1)
        self.assertNotIn("<query>", prompt)
        self.assertNotIn("same item", prompt)
        self.assertTrue(prompt.endswith("Canonical misconception label:"))

    def test_early_label_controls_are_explicit_and_callable(self) -> None:
        self.assertTrue(callable(backward_early_label_controls))

    def test_early_label_pair_requires_distinct_diagnosis_labels(self) -> None:
        base = {
            "causal_target_kind": "diagnosis_natural_schema",
            "native_target_label": "recipient",
            "donor_target_label": "donor",
        }
        self.assertTrue(is_early_label_pair(base))
        self.assertFalse(is_early_label_pair({**base, "donor_target_label": "recipient"}))
        self.assertFalse(is_early_label_pair({**base, "causal_target_kind": "answer"}))

    def test_early_binding_prefers_semantic_label_without_dropping_taxonomy_id(self) -> None:
        row = {
            "native_target_label": "1847",
            "donor_target_label": "952",
            "native_semantic_label": "eedi.confuses_part_and_whole_id_1847",
            "donor_semantic_label": "eedi.adds_index_to_number_id_952",
        }
        native, donor, mode = early_binding_labels(row)
        self.assertEqual(native, row["native_semantic_label"])
        self.assertEqual(donor, row["donor_semantic_label"])
        self.assertEqual(mode, "semantic_identifier_only_no_gold_prefix")

    def test_early_label_receipt_uses_global_history(self) -> None:
        value = aggregate_early_label_counts(
            [
                {"rank": 0, "source": "eedi_mining", "early_label_binding_effective_weight": 0.5, "early_label_binding_source": "eedi_mining"},
                {"rank": 1, "source": "malrule", "early_label_binding_effective_weight": 0.5, "early_label_binding_source": "malrule"},
            ],
            0.5,
        )
        self.assertEqual(value["applied"], 2)
        self.assertEqual(value["by_source"], {"eedi_mining": 1, "malrule": 1})
        with self.assertRaisesRegex(RuntimeError, "produced no"):
            aggregate_early_label_counts([], 0.5)

    def test_control_random_seeds_use_frozen_c4_stride(self) -> None:
        self.assertEqual(
            control_random_seeds(17, 4),
            [17, 1_000_020, 2_000_023, 3_000_026],
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            control_random_seeds(17, 0)

    def test_recipient_control_loss_enforces_recipient_margin(self) -> None:
        values = torch.tensor([0.8, 0.2], requires_grad=True)
        loss = recipient_control_two_view_loss(
            values,
            values,
            margin=0.5,
            active=1.0,
        )
        # recipient CE + (margin - (donor CE - recipient CE))
        self.assertAlmostEqual(float(loss.detach()), 1.9, places=6)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([2.0, -1.0]))

    def test_recipient_control_loss_has_no_hinge_after_margin(self) -> None:
        losses = torch.tensor([0.3, 1.0], requires_grad=True)
        loss = recipient_control_two_view_loss(
            losses,
            losses,
            margin=0.5,
            active=0.0,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.3, places=6)

    def test_distributed_geometry_receipt_uses_global_history(self) -> None:
        history = [
            {"rank": 0, "semantic_geometry_effective_weight": 0.2},
            {"rank": 2, "semantic_geometry_effective_weight": 0.2},
            {"rank": 1, "eedi_semantic_geometry_effective_weight": 0.5},
        ]
        causal, eedi = aggregate_geometry_counts(
            history, geometry_weight=0.1, eedi_geometry_weight=0.5
        )
        self.assertEqual(causal["applied"], 2)
        self.assertEqual(causal["by_rank"], {"0": 1, "2": 1})
        self.assertEqual(eedi["applied"], 1)
        self.assertEqual(causal["audit_source"], "global_trainer_history")

    def test_positive_geometry_weight_fails_closed_without_history_marker(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "produced no"):
            aggregate_geometry_counts(
                [{"rank": 0}], geometry_weight=1.0, eedi_geometry_weight=0.0
            )

    def test_updater_only_objective_is_a_distinct_callable(self) -> None:
        self.assertTrue(callable(backward_updater_only_specificity))

    def test_eedi_pair_marker_is_fail_closed(self) -> None:
        self.assertTrue(
            is_eedi_semantic_pair(
                {
                    "eedi_semantic_pair": True,
                    "eedi_pair_supervision_visibility": "train_only_not_decoder_input",
                }
            )
        )
        self.assertFalse(
            is_eedi_semantic_pair(
                {
                    "eedi_semantic_pair": True,
                    "eedi_pair_supervision_visibility": "decoder_input",
                }
            )
        )

    def test_random_target_is_natural_schema(self) -> None:
        self.assertTrue(natural_schema_valid(NEUTRAL_TARGET))
        self.assertIn("insufficient_evidence", NEUTRAL_TARGET)

    def test_random_neutral_two_view_split_is_scalar_and_gradient_exact(self) -> None:
        margin = 0.5
        original_parameter = torch.tensor(0.2, requires_grad=True)
        neutral = (original_parameter - 0.4).square()
        native = (original_parameter + 0.3).square()
        donor = (original_parameter - 0.8).square()
        native_advantage = native - neutral
        donor_advantage = donor - neutral
        native_active = float(native_advantage.detach() < margin)
        donor_active = float(donor_advantage.detach() < margin)
        original = (
            neutral / 3.0
            + native_active * (margin - native_advantage) / 4.0
            + donor_active * (margin - donor_advantage) / 4.0
        )
        original.backward()
        original_gradient = original_parameter.grad.detach().clone()

        split_parameter = torch.tensor(0.2, requires_grad=True)
        neutral_native = (split_parameter - 0.4).square()
        native_split = (split_parameter + 0.3).square()
        neutral_donor = (split_parameter - 0.4).square()
        donor_split = (split_parameter - 0.8).square()
        branch_native = random_neutral_two_view_branch_loss(
            torch.stack((neutral_native, native_split)),
            torch.stack((neutral_native, native_split)),
            margin=margin,
            active=native_active,
        )
        branch_donor = random_neutral_two_view_branch_loss(
            torch.stack((neutral_donor, donor_split)),
            torch.stack((neutral_donor, donor_split)),
            margin=margin,
            active=donor_active,
        )
        branch_native.backward()
        branch_donor.backward()
        self.assertAlmostEqual(
            float(original.detach()),
            float((branch_native + branch_donor).detach()),
            places=7,
        )
        torch.testing.assert_close(original_gradient, split_parameter.grad)

    def test_random_neutral_split_rejects_three_view_graphs(self) -> None:
        values = torch.ones(3)
        with self.assertRaisesRegex(ValueError, "requires two views"):
            random_neutral_two_view_branch_loss(
                values,
                values,
                margin=0.5,
                active=1.0,
            )

    def test_per_sequence_label_weight_changes_only_schema_row(self) -> None:
        loss = schema_weighted_per_sequence_factory(TinyTokenizer(), 4.0)
        labels = torch.tensor(
            [
                [-100, 10, 11, 99, 12, 13, 42],
                [-100, 7, 8, 9, -100, -100, -100],
            ]
        )
        logits = torch.zeros(2, 7, 128, requires_grad=True)
        logits.data[0, 2, 99] = -4.0
        weighted = loss(logits, labels, "mean")
        plain = []
        for index in range(2):
            valid = labels[index, 1:].ne(-100)
            plain.append(
                torch.nn.functional.cross_entropy(
                    logits[index, :-1][valid], labels[index, 1:][valid]
                )
            )
        self.assertGreater(float(weighted[0].detach()), float(plain[0].detach()))
        self.assertAlmostEqual(
            float(weighted[1].detach()), float(plain[1].detach()), places=6
        )
        weighted.sum().backward()
        self.assertIsNotNone(logits.grad)

    def test_geometry_attracts_same_rule_and_separates_donor(self) -> None:
        recipient = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
        same_rule = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
        donor = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)
        loss, metrics = semantic_state_geometry_loss(
            recipient, same_rule, donor, different_max_cosine=0.25
        )
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertGreater(
            float(metrics["semantic_same_rule_cosine"].detach()),
            float(metrics["semantic_recipient_donor_cosine"].detach()),
        )
        loss.backward()
        self.assertIsNotNone(recipient.grad)
        self.assertIsNotNone(same_rule.grad)
        self.assertIsNotNone(donor.grad)


if __name__ == "__main__":
    unittest.main()
