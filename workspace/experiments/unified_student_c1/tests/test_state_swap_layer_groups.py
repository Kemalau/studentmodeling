from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from state_swap_misconception_smoke import (  # noqa: E402
    build_real_random_controls,
    dose_condition,
    interpolate_patch_vectors,
    load_pair_manifest,
    parse_dose_alphas,
    parse_layer_groups,
)
import torch


class LayerGroupParserTest(unittest.TestCase):
    def test_named_ranges_and_singletons(self):
        self.assertEqual(
            parse_layer_groups("final=63;late=56-63;mixed=48,56,63", 64),
            [("final", [63]), ("late", list(range(56, 64))), ("mixed", [48, 56, 63])],
        )

    def test_rejects_duplicate_or_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_layer_groups("x=1;x=2", 4)
        with self.assertRaises(ValueError):
            parse_layer_groups("x=3-4", 4)

    def test_manifest_prefix_is_explicit(self):
        import json
        import tempfile

        rows = []
        specs = []
        for index in range(3):
            donor = {
                "example_id": f"d{index}", "task": "malrule_next_response",
                "metadata": {"malrule_id": "m1", "family": "f"},
                "candidate_labels": ["a", "b"], "target_label": "a",
                "query": {"correct_answer": "c"},
            }
            recipient = {
                "example_id": f"r{index}", "task": "malrule_next_response",
                "metadata": {"malrule_id": "m2", "family": "f", "query_answers_by_malrule": {"m1": "b"}},
                "candidate_labels": ["a", "b"], "target_label": "a",
                "query": {"correct_answer": "c"},
            }
            control = {
                "example_id": f"c{index}", "task": "malrule_next_response",
                "metadata": {"malrule_id": "m2", "family": "f"},
                "candidate_labels": ["a", "b"], "target_label": "a",
                "query": {"correct_answer": "c"},
            }
            rows.extend([donor, recipient, control])
            specs.append({"donor_id": donor["example_id"], "recipient_id": recipient["example_id"], "samemal_id": control["example_id"]})
        payload = {"task": "malrule_next_response", "swap_target": "response", "pairs": specs}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_pair_manifest(path, rows, "malrule_next_response", "response", 2)
            pairs, controls = load_pair_manifest(
                path, rows, "malrule_next_response", "response", 2, allow_prefix=True
            )
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len(controls), 2)

    def test_real_random_control_is_family_matched_and_reproducible(self):
        def row(example_id, rule, family="f"):
            return {
                "example_id": example_id,
                "task": "malrule_next_response",
                "metadata": {"malrule_id": rule, "family": family},
            }
        donor, recipient, same = row("d", "a"), row("r", "b"), row("s", "b")
        rows = [donor, recipient, same, row("c1", "c"), row("c2", "d"), row("x", "c", "g")]
        first = build_real_random_controls(rows, [(donor, recipient)], [same], 17)
        second = build_real_random_controls(list(reversed(rows)), [(donor, recipient)], [same], 17)
        self.assertEqual(first[0]["example_id"], second[0]["example_id"])
        self.assertIn(first[0]["example_id"], {"c1", "c2"})

    def test_dose_interpolation_and_names(self):
        self.assertEqual(parse_dose_alphas("0.75,0.25,0.5"), [0.25, 0.5, 0.75])
        self.assertEqual(dose_condition(0.25), "dose_0p25")
        recipient = {0: torch.tensor([0.0, 2.0])}
        donor = {0: torch.tensor([4.0, 6.0])}
        torch.testing.assert_close(
            interpolate_patch_vectors(recipient, donor, [0], 0.25)[0],
            torch.tensor([1.0, 3.0]),
        )
        with self.assertRaises(ValueError):
            parse_dose_alphas("0,0.5")


if __name__ == "__main__":
    unittest.main()
