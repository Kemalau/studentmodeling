import unittest
from pathlib import Path

from scripts.validate_state8_diagnosis_c4_shards_amlt import load, validate


class State8DiagnosisC4ShardsAmLtTest(unittest.TestCase):
    def test_config_is_fail_closed_and_disjoint(self):
        path = Path("amlt/feeds_state8_diagnosis_c4_shards_20260806.yaml")
        validate(load(path))


if __name__ == "__main__":
    unittest.main()
