import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_recurrent_diagnosis_c4 import CONDITIONS
from scripts.merge_state8_diagnosis_c4_shards import merge


class MergeDiagnosisC4ShardsTest(unittest.TestCase):
    def test_merge_requires_exact_pair_condition_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / "s0", Path(temporary) / "s1"]
            for root, pair_idx in zip(roots, (1, 2), strict=True):
                root.mkdir()
                (root / "metrics.json").write_text(
                    json.dumps({"artifact_sha256": {"adapter_model": "a", "recurrent_state": "s"}})
                )
                records = []
                predictions = []
                eval_rows = []
                for condition in CONDITIONS:
                    records.append(
                        {
                            "pair_idx": pair_idx,
                            "condition": condition,
                            "category": "other",
                            "natural_schema_valid": True,
                            "raw_output": condition,
                        }
                    )
                    for role in ("donor", "recipient"):
                        example_id = f"diagnosis-c4:{pair_idx:04d}:{condition}:{role}"
                        predictions.append({"example_id": example_id})
                        eval_rows.append({"example_id": example_id})
                for name, rows in (
                    ("predictions.jsonl", records),
                    ("judge_predictions.jsonl", predictions),
                    ("judge_eval.jsonl", eval_rows),
                ):
                    (root / name).write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
            result = merge(roots, 2)
            self.assertEqual(result["metrics"]["num_pairs"], 2)
            self.assertEqual(len(result["records"]), 2 * len(CONDITIONS))


if __name__ == "__main__":
    unittest.main()
