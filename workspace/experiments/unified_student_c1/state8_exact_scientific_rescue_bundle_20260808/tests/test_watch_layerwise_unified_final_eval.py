from __future__ import annotations
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class FinalEvalWatchTest(unittest.TestCase):
 def test_fail_closed(self):
  x=(ROOT/'scripts/watch_layerwise_unified_final_eval.sh').read_text();self.assertIn('[[ ! -f "$fit_dir/scientific_gate_pass" ]]',x);self.assertNotIn('final fit needs attention',x);self.assertIn('validate_layerwise_unified_final_eval_amlt.py',x);self.assertIn('all_expected_jobs_pass(sys.argv[1],int(sys.argv[2]))',x);self.assertIn('terminal_nonpass_job_names',x);self.assertIn('preserve_passed',x);self.assertIn('summarize_layerwise_unified_final_eval.py',x);self.assertIn('scientific_gate_failed',x);self.assertIn('while ! az account get-access-token',x);self.assertIn('while ! az account set',x)
if __name__=='__main__':unittest.main()
