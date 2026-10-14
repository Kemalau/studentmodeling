from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_one_shot_layer_transport import summarize_payload  # noqa: E402


class LayerTransportSummaryTest(unittest.TestCase):
    def test_specificity_uses_same_clean_subset(self):
        records = []
        predictions = {
            1: {"none": "B", "noop": "B", "random": "B", "samemal": "B", "swap": "A"},
            2: {"none": "A", "noop": "A", "random": "A", "samemal": "A", "swap": "A"},
        }
        for pair, conditions in predictions.items():
            for condition, prediction in conditions.items():
                category = "flip_to_A" if prediction == "A" else "stay_B"
                records.append({
                    "layer_config": "L1", "layers": [1], "pair_idx": pair,
                    "condition": condition, "category": category, "pred": prediction,
                    "score_A": float(prediction == "A"), "score_B": float(prediction == "B"),
                })
        row = summarize_payload({"records": records})[0]
        self.assertEqual(row["clean_n"], 1)
        self.assertEqual(row["D_specific"], 1.0)
        self.assertEqual(row["D_damage"], 0.0)


if __name__ == "__main__":
    unittest.main()
