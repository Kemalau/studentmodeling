from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from state_swap_misconception_smoke import load_pair_manifest, parse_layer_groups  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
