"""Resource routing tests: no GPU allocation or model inference."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_c4_unseen_rules_local import ready, adapter


def test_only_start_on_empty_gpu_with_headroom():
    assert ready(81000,'')
    assert not ready(17338,'144597, 63802')
    assert not ready(81000,'1234, 20')
    assert not ready(44999,'')


def test_checkpoint_paths_are_final_and_arm_specific():
    assert adapter('A')!=adapter('B')
    assert adapter('A').name==adapter('B').name=='final_adapter'
    assert '/execution_A/' in str(adapter('A'))
    assert '/execution_B/' in str(adapter('B'))
