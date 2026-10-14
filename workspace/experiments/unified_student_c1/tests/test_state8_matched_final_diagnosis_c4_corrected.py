from __future__ import annotations
import json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from materialize_state8_matched_final_diagnosis_c4_corrected import materialize
from validate_state8_matched_final_diagnosis_c4_corrected_amlt import validate

TEMPLATE=ROOT/'amlt/feeds_state8_matched_final_diagnosis_c4_corrected_20260810.yaml'
LAUNCHER=ROOT/'scripts/run_state8_matched_final_diagnosis_c4_corrected.sh'

def test_template_and_hash_bound_render_validate(tmp_path: Path):
    validate(TEMPLATE,allow_template=True)
    receipt=tmp_path/'receipt.json'; receipt.write_text(json.dumps({'status':'complete','state_shape':[64,8,5120],'sha256':{'adapter_model':'a'*64,'recurrent_state':'b'*64}}))
    out=tmp_path/'amlt'/'rendered.yaml'; out.parent.mkdir()
    materialize(TEMPLATE,receipt,out,LAUNCHER); validate(out)

def test_bad_receipt_fails(tmp_path: Path):
    receipt=tmp_path/'bad.json'; receipt.write_text('{}')
    with pytest.raises(ValueError,match='receipt drift'):
        materialize(TEMPLATE,receipt,tmp_path/'amlt'/'x.yaml',LAUNCHER)

def test_launcher_enforces_correct_taxonomy_and_strict_state_only():
    text=LAUNCHER.read_text()
    for fragment in ('For this Malrule diagnosis','For this Eedi diagnosis','raw_history_visible_to_decoder','--num-pairs 20','EXPECTED_RECEIPT_SHA256'):
        assert fragment in text
