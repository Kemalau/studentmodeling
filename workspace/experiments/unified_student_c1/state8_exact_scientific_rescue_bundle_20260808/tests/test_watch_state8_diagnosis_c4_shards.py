import unittest
from pathlib import Path


class WatchState8DiagnosisC4ShardsTest(unittest.TestCase):
    def test_watcher_closes_merge_and_semantic_judge_loop(self):
        source = Path("scripts/watch_state8_diagnosis_c4_shards.sh").read_text()
        self.assertIn("all_expected_jobs_pass(sys.argv[1],5)", source)
        self.assertIn("--expected-pairs 50", source)
        self.assertIn("--judge-backend trapi", source)
        self.assertIn("summarize_state8_diagnosis_c4.py", source)
        self.assertIn("touch \"$state/complete\"", source)


if __name__ == "__main__":
    unittest.main()
