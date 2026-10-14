import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from c4_feeds_quota import summarize


def test_workers_not_parents_or_other_target_and_deduplicate():
    base = dict(experiment='x', cluster='Feeds', status='queued', sku='80G1-A100')
    rows = [dict(base, job_id='parent', flags='HD|STD'),
            dict(base, job_id='child', flags='STD'),
            dict(base, job_id='child', flags='STD'),
            dict(base, job_id='alt', cluster='Feeds-LLM'),
            dict(base, job_id='done', status='pass'),
            dict(base, job_id='multi', sku='2x80G2-A100')]
    result = summarize(rows, 32)
    assert result['running_a100_requests'] == 5
    assert result['available_user_slots'] == 27
    assert result['auditable']


def test_unknown_rows_close_slot_estimate():
    for row in (dict(job_id='a', status='running', sku='80G1-A100'),
                dict(job_id='b', cluster='Feeds', status='mystery', sku='80G1-A100'),
                dict(job_id='c', cluster='Feeds', status='running', sku='new-A100')):
        result = summarize([row], 32)
        assert not result['auditable']
        assert result['available_user_slots'] == 0


def test_no_exposed_personal_cap_is_not_reported_as_pool_entitlement():
    rows=[dict(job_id='a',cluster='Feeds-LLM',status='running',sku='80G1-A100'),
          dict(job_id='b',cluster='Feeds',status='running',sku='80G1-A100')]
    result=summarize(rows,None,'Feeds-LLM')
    assert result['auditable'] and result['running_a100_requests']==1
    assert result['user_limit'] is None and result['available_user_slots'] is None


def test_submission_capacity_checks_pool_and_personal_limits():
    from watch_c4_execution_training import capacity_available
    row={'quotas_(used/limit)':{'standard':{'used':3,'limit':80}}}
    assert capacity_available(row,dict(auditable=True,available_user_slots=None))
    assert not capacity_available(row,dict(auditable=False,available_user_slots=None))
    assert not capacity_available(row,dict(auditable=True,available_user_slots=1))
    row['quotas_(used/limit)']['standard']['used']=79
    assert not capacity_available(row,dict(auditable=True,available_user_slots=None))
