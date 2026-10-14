import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from feeds_quota_utils import parse_json_table, select_a100_80gb  # noqa: E402


class FeedsQuotaUtilsTest(unittest.TestCase):
    def test_parses_prefixed_table_and_null_user_usage(self) -> None:
        text = 'warning [not json]\n[{"target_name":"Feeds","series":"NDAMv4","accelerator":{"type":"A100","memory_gb":80},"quotas_(used/limit)":{"standard":{"used":10,"limit":97},"user_max":{"used":null,"limit":32}}}]\n'
        row = select_a100_80gb(parse_json_table(text))
        quotas = row["quotas_(used/limit)"]
        self.assertEqual(int(quotas["user_max"].get("used") or 0), 0)
        self.assertEqual(int(quotas["standard"]["limit"]) - int(quotas["standard"]["used"]), 87)

    def test_wrong_gpu_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_a100_80gb([{"target_name": "Feeds", "series": "NDv2", "accelerator": {"type": "V100", "memory_gb": 32}}])


if __name__ == "__main__":
    unittest.main()
