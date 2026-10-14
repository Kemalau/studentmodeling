"""Wait on the live existing collector, then audit and report all outcomes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from watch_c4_unseen_rules_feeds import ROOT,EXP,PREFIX


def atomic(value):
    path=Path(str(PREFIX)+'.finalizer.json');tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(value,pid=os.getpid(),experiment=EXP,
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n');tmp.replace(path)


def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    return True


def ready(state,is_alive=alive):
    assert state['experiment']==EXP
    if state['status']=='complete':return True
    if state['status']=='needs_attention':raise RuntimeError('Collector needs attention; inspect original job/receipt')
    assert state['status'] in ('watching','collecting','observation_error')
    if not is_alive(state['pid']):raise RuntimeError('Existing collector handle is missing; no automatic resubmission')
    return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');args=p.parse_args()
    if args.background:
        with Path(str(PREFIX)+'.finalizer.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,experiment=EXP)));return
    with Path(str(PREFIX)+'.finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            atomic(dict(status='waiting_on_collector'))
            while not ready(json.loads(Path(str(PREFIX)+'.status.json').read_text())):time.sleep(30)
            atomic(dict(status='auditing'))
            report=ROOT/'reports/c4_unseen_rules_results_20260906.md'
            audit=Path(str(PREFIX)+'.delivery_audit.json')
            subprocess.run([sys.executable,'scripts/audit_c4_unseen_rules_delivery.py',
                '--audit',str(audit),'--report',str(report)],cwd=ROOT,check=True)
            result=json.loads(audit.read_text())
            assert result['status']=='complete' and result['aggregate']['n_records']==1728
            atomic(dict(status='complete',audit=str(audit),report=str(report)))
        except Exception as exc:
            atomic(dict(status='needs_attention',error=str(exc)));raise


if __name__=='__main__':main()
