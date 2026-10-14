from pathlib import Path
import copy
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_c4_unseen_rules_feeds import build,FILES
from preflight_c4_unseen_rules import validate


def test_frozen_remote_config():
    c=build()
    assert c['target']['name']=='Feeds-LLM'
    assert len(c['jobs'])==2
    for arm,j in zip(('A','B'),c['jobs']):
        assert j['sku']=='80G1-A100' and j['process_count_per_node']==1
        assert j['command']==f'bash scripts/run_c4_unseen_rules_feeds.sh {arm}'
        assert j['submit_args']['env']['_AZUREML_SINGULARITY_JOB_UAI']=='rankfun_aml'
    assert not any(f.endswith('.safetensors') or f.startswith('results/') for f in FILES)
    assert all((ROOT/f).is_file() for f in FILES)


def test_remote_panel_preflight_rejects_mutation():
    m=json.loads((ROOT/'manifests/c4_unseen_rules_v1_20260906.json').read_text())
    validate(m)
    bad=copy.deepcopy(m);bad['groups'][0]['queries'][0]['targets']['donor']='999999'
    with pytest.raises(AssertionError):validate(bad)
