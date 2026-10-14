"""One authorized submission; never retries an ambiguous write."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
import yaml
from c4_feeds_quota import count
from c4_freegen_protocol import sha
from count_active_a100_requests import json_table
from prepare_c4_unseen_rules_feeds import build,CONFIG,HASH,ROOT
from watch_c4_unseen_rules_feeds import EXP,PREFIX


def main():
    attempt=Path(str(PREFIX)+'.submission.json')
    with (ROOT/'results/feeds_submission_global.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not attempt.exists(),'Prior submission attempt exists: inspect rather than resubmit'
        config=yaml.safe_load(CONFIG.read_text());assert config==build()
        subprocess.run(['sha256sum','-c',HASH],cwd=ROOT,check=True)
        rows=json_table(['amlt','--json-tables','ti','sing','-t','Feeds-LLM','-a'])
        quota=next(r for r in rows if r['series']=='NDAMv4' and r['accelerator']=={'type':'A100','memory_gb':80})
        limits=quota['quotas_(used/limit)'];personal=limits.get('user_max')
        cap=personal.get('limit') if personal else None
        audit=count(cap,target='Feeds-LLM')
        recent=json_table(['amlt','--json-tables','list','-n','400'])
        by_name={r['experiment_name']:r for r in recent}
        assert EXP not in by_name,'Experiment already exists; inspect before submission'
        # AMLT emits no JSON table for empty experiments. Resolve only this
        # evidenced zero-job case; preserve all unknown/error cases fail-closed.
        unresolved=[];empty=[]
        for error in audit['errors']:
            r=by_name.get(error.get('experiment'),{})
            if (error.get('reason','').startswith('missing JSON table:') and
                r.get('n_jobs')==0 and r.get('_all_terminated') is True):empty.append(r)
            else:unresolved.append(error)
        assert not unresolved,unresolved
        audit['resolved_empty_experiments']=empty
        audit['original_errors']=audit['errors'];audit['errors']=[];audit['auditable']=True
        assert limits['standard']['limit']-limits['standard']['used']>=2
        assert cap is None or cap-audit['running_a100_requests']>=2
        cmd=['amlt','run',str(CONFIG),EXP,'-t','Feeds-LLM','-y','--description',config['description']]
        receipt=dict(status='submitting',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            authorization='User explicitly requested Feeds-LLM',quota=quota,user_audit=audit,
            config_sha256=sha(CONFIG),manifest_sha256=sha(ROOT/'manifests/c4_unseen_rules_v1_20260906.json'),
            command=cmd)
        attempt.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(status='submitting',experiment=EXP,quota=quota,a100_requested=2)),flush=True)
        with Path(str(PREFIX)+'.submission.log').open('a') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(returncode=result.returncode,status='submitted_unverified' if result.returncode==0 else 'needs_inspection_no_retry')
        attempt.write_text(json.dumps(receipt,indent=2)+'\n')
        if result.returncode:raise RuntimeError(f'Submission rc={result.returncode}; inspect complete submission log, no duplicate retry')
        jobs=json_table(['amlt','--json-tables','status',EXP,'--force-status-update'])
        assert len(jobs)==2
        receipt.update(status='submitted',jobs=jobs)
        attempt.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
