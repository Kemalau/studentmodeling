"""Target-scoped, fresh AMLT A100 request audit for the C4 supervisor.

HyperDrive orchestration parents carry a displayed SKU but are not workers.
Explicitly refresh their children; keep unknown rows fail-closed.
"""
from concurrent.futures import ThreadPoolExecutor
import re

from count_active_a100_requests import json_table

TERMINAL = {'pass', 'failed', 'killed', 'cancelled', 'canceled', 'completed',
            'stopped', 'expired'}
ACTIVE = {'preparing', 'pre-queued', 'queued', 'running', 'paused', 'resuming'}


def summarize(rows, limit, target='Feeds'):
    jobs = {}
    excluded = []
    errors = []
    for row in rows:
        state = str(row.get('status', '')).lower()
        if state in TERMINAL:
            continue
        cluster = row.get('cluster')
        if cluster and cluster != target:
            excluded.append({'job_id': row.get('job_id'), 'reason': 'other_target',
                             'cluster': cluster})
            continue
        if not cluster or state not in ACTIVE:
            errors.append({'job_id': row.get('job_id'), 'reason': 'unknown_target_or_state'})
            continue
        if 'HD' in str(row.get('flags', '')).split('|'):
            excluded.append({'job_id': row.get('job_id'), 'reason': 'hyperdrive_parent'})
            continue
        sku = str(row.get('sku', ''))
        match = re.fullmatch(r'(?:(\d+)x)?80G(\d+)-A100(?:-.*)?', sku)
        if not match:
            if not sku or 'A100' in sku:
                errors.append({'job_id': row.get('job_id'), 'reason': 'unknown_sku'})
            continue
        job_id = row.get('job_id')
        if not job_id:
            errors.append({'reason': 'missing_job_id'})
            continue
        value = {k: row.get(k) for k in ('experiment', 'job_id', 'job_name', 'cluster', 'status', 'sku')}
        value['gpu_count'] = int(match.group(1) or 1) * int(match.group(2))
        if job_id in jobs and jobs[job_id] != value:
            errors.append({'job_id': job_id, 'reason': 'conflicting_duplicate'})
        jobs[job_id] = value
    total = sum(j['gpu_count'] for j in jobs.values())
    return dict(stage='c4_target_scoped_a100_audit_v1', target=target, user_limit=limit,
                running_a100_requests=total, auditable=not errors,
                available_user_slots=(None if limit is None else max(0, limit-total)) if not errors else 0,
                jobs=sorted(jobs.values(), key=lambda r: r['job_id']), excluded=excluded,
                errors=errors)


def count(limit, recent=400, target='Feeds'):
    experiments = json_table(['amlt', '--json-tables', 'list', '-n', str(recent)])
    names = sorted({r['experiment_name'] for r in experiments
                    if not (r.get('_statuses') and
                            set(str(x).lower() for x in r['_statuses']) <= TERMINAL)})

    def fetch(name):
        try:
            rows = json_table(['amlt', '--json-tables', 'status', name, '--hide-urls',
                               '--force-hd-refresh', '--force-status-update'])
            return rows, None
        except Exception as exc:
            return [], {'experiment': name, 'reason': str(exc)[:1000]}

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for found, error in pool.map(fetch, names):
            rows.extend(found)
            if error:
                errors.append(error)
    audit = summarize(rows, limit, target)
    audit['experiments_inspected'] = names
    audit['errors'].extend(errors)
    if errors:
        audit['auditable'] = False
        audit['available_user_slots'] = 0
    return audit
