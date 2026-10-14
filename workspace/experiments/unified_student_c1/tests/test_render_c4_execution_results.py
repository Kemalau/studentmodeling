"""Synthetic report checks; numbers are not experimental evidence."""
import copy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_c4_execution_results import render


def example():
    effect=dict(estimate=0.,pair_cluster_95ci=[-.1,.1])
    counts={c:dict(n=96,donor_hits=0,recipient_hits=0,invalid=0)
            for c in ('native','swap','random')}
    arm=dict(counts=counts,contrasts=dict(individual=effect),
        fingerprint=dict(adapter_sha256='synthetic'),sha256='synthetic_predictions')
    model=dict(integrity=dict(records=960,groups=32,token_text_checked=True),
        all_three_by_condition=dict(swap=dict(all3_donor_hits=0,n_groups=32)),
        noop_token_mismatches=[],per_donor_rule={})
    train=dict(unique_steps=240,logged_updates=240,replayed_steps={},validation={})
    return dict(status='complete',core=dict(status='complete',B_minus_A_specificity=effect,
        A_minus_baseline_specificity=effect,B_minus_baseline_specificity=effect,
        arms={k:copy.deepcopy(arm) for k in ('baseline','A','B')}),
        models={k:copy.deepcopy(model) for k in ('baseline','A','B')},
        training={k:copy.deepcopy(train) for k in ('A','B')},examples={})


def test_report_requires_full_audit_and_discloses_noop():
    a=example();m=dict(init_adapter_sha256='synthetic_initial')
    text=render(a,m)
    assert 'does not establish an incremental specificity benefit' in text
    for n in range(1,14):assert f'## {n}.' in text
    a['models']['B']['noop_token_mismatches']=[dict(group_id='synthetic')]
    assert 'not exactly no-op invariant' in render(a,m)
    a['models']['B']['integrity']['records']=930
    with pytest.raises(AssertionError):render(a,m)
    a=example();a['models']['B']['integrity']['token_text_checked']=False
    with pytest.raises(AssertionError):render(a,m)
