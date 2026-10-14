from __future__ import annotations
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class TestWatch(unittest.TestCase):
 def test_protocol(self):
  x=(ROOT/'scripts/watch_layerwise_t3_semantic_judge.sh').read_text();self.assertIn('final_eval',x);self.assertNotIn('final eval needs attention',x);self.assertIn('--max-records 16',x);self.assertIn('full1494',x);self.assertIn('validate_t3_semantic_result.py',x);self.assertIn('validate_layerwise_unified_pipeline.py',x);self.assertIn('overall_complete',x);self.assertIn('/home/v-yuejiali/fkft/.venv/bin/python',x);self.assertIn('/home/v-yuejiali/fkft/src',x)
if __name__=='__main__':unittest.main()
