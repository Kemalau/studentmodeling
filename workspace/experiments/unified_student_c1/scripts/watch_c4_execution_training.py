"""Authorized submit when quota permits, native watch, download and CPU audit.

Only an explicitly selected target; ambiguous submissions stop for inspection.
No scientific-outcome-driven restarts. Terminal-only native notifications.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from count_active_a100_requests import json_table
from c4_feeds_quota import count
from collect_c4_execution_training import collect
from c4_freegen_protocol import sha

ROOT=Path(__file__).resolve().parents[1]
EXP='c4-execution-training-v1-seed42-20260906'
PREFIX=ROOT/'results/c4_execution_training_v1_20260906'

def write(value):
    path=Path(str(PREFIX)+'.status.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def capacity_available(row,audit,required=2):
    quotas=row['quotas_(used/limit)']
    standard=quotas['standard']
    # A missing personal cap is not a promise of pool ownership. For this
    # bounded two-job submission, require audited requests and pool headroom.
    personal=audit['available_user_slots']
    return (audit['auditable'] and standard['limit']-standard['used']>=required
            and (personal is None or personal>=required))

def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true')
    p.add_argument('--target',choices=('Feeds','Feeds-LLM'),default='Feeds');a=p.parse_args()
    if a.background:
        rec=Path(str(PREFIX)+'.status.json')
        if rec.exists():
            old=json.loads(rec.read_text())
            if old.get('pid'):
                try:os.kill(old['pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('Already active')
        with Path(str(PREFIX)+'.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--target',a.target],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,experiment=EXP)));return
    base=dict(pid=os.getpid(),experiment=EXP,started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    config=ROOT/'amlt/feeds_c4_execution_training_v1_20260906.yaml'
    try:
        import yaml
        c=yaml.safe_load(config.read_text());target=c['target']['name']
        assert target==a.target,'Config differs from explicitly authorized target'
        config_hash=sha(config)
        # Never auto-resubmit after an attempted or ambiguous submission.
        attempted=Path(str(PREFIX)+'.submission_attempt.json')
        if attempted.exists():raise RuntimeError('Submission was already attempted; inspect before restarting watcher')
        while True:
            diagnostic=ROOT/'results/c4_execution_training_20260906_backward_diagnostic/diagnostic.json'
            if not diagnostic.exists():
                runtime=ROOT/'results/c4_execution_training_20260906_backward_diagnostic/runtime.json'
                if runtime.exists():
                    pid=json.loads(runtime.read_text())['pid']
                    try:os.kill(pid,0)
                    except ProcessLookupError:raise RuntimeError('Backward diagnostic exited without success receipt; inspect failure, do not submit')
                write(dict(base,status='waiting_for_backward_diagnostic',target=target))
                time.sleep(30)
                continue
            check=json.loads(diagnostic.read_text())
            assert check['source_embedding_gradient_norm']>0 and check['gradient_norm']>0
            subprocess.run(['sha256sum','-c','manifests/c4_execution_training_v1_20260906.sha256'],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
            assert sha(config)==config_hash
            with (ROOT/'results/feeds_submission_global.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                quota=json_table(['amlt','--json-tables','ti','sing','-t',target,'-a'])
                row=next(r for r in quota if r['series']=='NDAMv4' and r['accelerator']['type']=='A100')
                user_max=row['quotas_(used/limit)'].get('user_max')
                cap=int(user_max['limit']) if user_max and user_max.get('limit') is not None else None
                audit=count(cap,400,target)
                stamp=dict(base,status='waiting_for_quota',target=target,quota=row,user_audit=audit,config_sha256=config_hash)
                write(stamp)
                if capacity_available(row,audit):
                    # Recent-list collision check before write to avoid duplicate names.
                    recent=json_table(['amlt','--json-tables','list','-n','400'])
                    assert not any(r['experiment_name']==EXP for r in recent)
                    attempted.write_text(json.dumps(dict(stamp,status='submitting'),indent=2)+'\n')
                    write(dict(stamp,status='submitting'))
                    result=subprocess.run(['amlt','run',str(config),EXP,'-t',target,'-y','--description',c['description']],cwd=ROOT)
                    if result.returncode:raise RuntimeError(f'Submission rc={result.returncode}; no automatic duplicate retry')
                    subprocess.run(['amlt','status',EXP],cwd=ROOT,check=True)
                    write(dict(base,status='submitted',target=target,config_sha256=config_hash))
                    break
            # Quota wait is persisted; no repeated unchanged user notifications.
            time.sleep(60)
        subprocess.run(['amlt','watch',EXP,'--json','--interval','60','--notify','terminal','--state-file',str(PREFIX)+'.watch.json'],cwd=ROOT,check=True)
        # Keep optimizer states/adapters remote; submission host has little disk.
        subprocess.run(['amlt','results','download',EXP,'--output',str(ROOT/'results/downloads'),
                        '--include','*.json','--include','*.jsonl'],cwd=ROOT,check=True,timeout=600)
        summary=collect(ROOT/'manifests/c4_execution_training_v1_20260906.json',ROOT/'results/downloads'/EXP)
        Path(str(PREFIX)+'.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        write(dict(base,status='complete',summary=str(PREFIX)+'.summary.json'))
    except Exception as exc:
        write(dict(base,status='needs_attention',error=str(exc)));raise

if __name__=='__main__':main()
