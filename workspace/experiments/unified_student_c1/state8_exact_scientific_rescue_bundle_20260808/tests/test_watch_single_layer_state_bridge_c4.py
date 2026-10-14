from pathlib import Path
import unittest
class Test(unittest.TestCase):
 def test_fail_closed(self):
  t=(Path(__file__).resolve().parents[1]/'scripts/watch_single_layer_state_bridge_c4.sh').read_text();self.assertIn('all_expected_jobs_pass(sys.argv[1],5)',t);self.assertIn("state_shape']==[5120]",t);self.assertIn('needs_attention',t);self.assertNotIn('amlt run',t)
if __name__=='__main__':unittest.main()
