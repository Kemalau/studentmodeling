from __future__ import annotations
import sys,unittest
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPT_DIR))
from evaluate_single_layer_state_bridge import summarize


class SingleLayerStateBridgeTest(unittest.TestCase):
    def test_gate_uses_single_clean_subset(self):
        records=[]
        for i in range(1,51):
            for cond,cat,pred in [('native','stay_B','B'),('noop','stay_B','B'),('samemal','stay_B','B'),('random','stay_B','B'),('swap','flip_to_A','A')]:records.append({'pair_idx':i,'condition':cond,'category':cat,'prediction':pred})
        value=summarize(records);self.assertEqual(value['clean_n'],50);self.assertEqual(value['D_specific'],1.0);self.assertTrue(value['scientific_gate_pass'])


if __name__=='__main__':unittest.main()
