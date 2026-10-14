"""One authorized Feeds-LLM submission with fresh quota and duplicate guards."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import yaml

from c4_feeds_quota import count
from c4_freegen_protocol import sha
from c4_history_diagnostic import ROOT, MANIFEST
from count_active_a100_requests import json_table
from prepare_c4_history_diagnostic_feeds import build, CONFIG, HASH

EXP='c4-history-diagnostic-v1-seed42-feedsllm-20260907'
PREFIX=ROOT/'results/c4_history_diagnostic_v1_20260907'


def resolve_empty(audit, recent):
    by_name={r['experiment_name']:r for r in recent};unresolved=[];empty=[]
    for error in audit['errors']:
        r=by_name.get(error.get('experiment'),{})
        if (error.get('reason','').startswith('missing JSON table:') and
            r.get('n_jobs')==0 and r.get('_all_terminated') is True):empty.append(r)
        else:unresolved.append(error)
    assert not unresolved,unresolved
    audit.update(original_errors=audit['errors'],errors=[],resolved_empty_experiments=empty,auditable=True)
    return audit


def main():
    attempt=Path(str(PREFIX)+'.submission.json')
    with (ROOT/'results/feeds_submission_global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not attempt.exists(),'Prior attempt exists; inspect, never duplicate'
        config=yaml.safe_load(CONFIG.read_text());assert config==build()
        assert len(config['jobs'])==8
        subprocess.run(['sha256sum','-c',HASH],cwd=ROOT,check=True)
        rows=json_table(['amlt','--json-tables','ti','sing','-t','Feeds-LLM','-a'])
        quota=next(r for r in rows if r['series']=='NDAMv4' and r['accelerator']==dict(type='A100',memory_gb=80))
        limits=quota['quotas_(used/limit)'];personal=limits.get('user_max')
        cap=personal.get('limit') if personal else None
        audit=count(cap,target='Feeds-LLM')
        recent=json_table(['amlt','--json-tables','list','-n','400'])
        assert EXP not in {r['experiment_name'] for r in recent},'Experiment exists; do not submit'
        audit=resolve_empty(audit,recent)
        assert limits['standard']['limit']-limits['standard']['used']>=8
        assert cap is None or cap-audit['running_a100_requests']>=8
        cmd=['amlt','run',str(CONFIG),EXP,'-t','Feeds-LLM','-y','--description',config['description']]
        receipt=dict(status='submitting',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            authorization='User requested to try diagnosing insufficient history after authorizing Feeds-LLM; inference only.',
            experiment=EXP,quota=quota,user_audit=audit,azure_config_dir=os.environ.get('AZURE_CONFIG_DIR'),
            config_sha256=sha(CONFIG),manifest_sha256=sha(ROOT/MANIFEST),command=cmd,
            image=config['environment']['image'],uai='rankfun_aml',gpu_count=8)
        attempt.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(status='submitting',experiment=EXP,quota=quota,a100_requested=8,existing_requests=audit['running_a100_requests'])),flush=True)
        with Path(str(PREFIX)+'.submission.log').open('a') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(returncode=result.returncode,status='submitted_unverified' if result.returncode==0 else 'needs_inspection_no_retry')
        attempt.write_text(json.dumps(receipt,indent=2)+'\n')
        if result.returncode:raise RuntimeError(f'Submission rc={result.returncode}; inspect complete log, no retry')
        jobs=json_table(['amlt','--json-tables','status',EXP,'--force-status-update'])
        assert len(jobs)==8
        receipt.update(status='submitted',jobs=jobs)
        attempt.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
