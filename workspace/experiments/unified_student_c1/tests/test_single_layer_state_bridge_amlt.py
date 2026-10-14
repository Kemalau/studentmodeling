from __future__ import annotations
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_single_layer_state_bridge_c4 import load,render
from validate_single_layer_state_bridge_amlt import validate
class SingleLayerBridgeAmltTest(unittest.TestCase):
 def test_five_jobs_validate(self):
  v=render(load(ROOT/'amlt/feeds_one_shot_layer_transport_c4_20260731.yaml'));validate(v);self.assertEqual(len(v['jobs']),5)
  self.assertTrue(all('state_capacity=one_hidden_vector' in str(j['command']) for j in v['jobs']))
if __name__=='__main__':unittest.main()
