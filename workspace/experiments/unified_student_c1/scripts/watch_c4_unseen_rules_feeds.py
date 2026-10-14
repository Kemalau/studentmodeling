"""Read-only native watch plus download and exact paired collection. No submit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from count_active_a100_requests import json_table

ROOT=Path(__file__).resolve().parents[1]
EXP='c4-unseen-rules-v1-seed42-feedsllm-20260906'
PREFIX=ROOT/'results/c4_unseen_rules_v1_feedsllm_20260906'


def write(status,**fields):
    path=Path(str(PREFIX)+'.status.json');tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),experiment=EXP,
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');a=p.parse_args()
    if a.background:
        with Path(str(PREFIX)+'.monitor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,experiment=EXP)));return
    with Path(str(PREFIX)+'.monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write('watching')
            while True:
                subprocess.run(['amlt','watch',EXP,'--json','--interval','60','--notify','terminal',
                    '--state-file',str(PREFIX)+'.watch.json'],cwd=ROOT,check=False)
                try:
                    jobs=json_table(['amlt','--json-tables','status',EXP,'--hide-urls','--force-status-update'])
                except Exception as exc:
                    write('observation_error',error=str(exc));time.sleep(60);continue
                states={str(j['status']).lower() for j in jobs}
                if len(jobs)==2 and states=={'pass'}:break
                if states & {'failed','killed','expired','cancelled','canceled'}:
                    write('needs_attention',jobs=jobs);return
                write('watching',jobs=jobs);time.sleep(60)
            write('collecting',jobs=jobs)
            downloads=ROOT/'results/downloads'
            subprocess.run(['amlt','results','download',EXP,'--output',str(downloads),
                '--include','*.json','--include','*.log'],cwd=ROOT,check=True,timeout=600)
            roots={arm:downloads/EXP/f'c4-unseen-rules-arm-{arm.lower()}-seed42'/f'unseen_{arm}' for arm in ('A','B')}
            runtimes={arm:json.loads((root.parent/f'runtime_{arm}.json').read_text()) for arm,root in roots.items()}
            assert runtimes['A']['versions']==runtimes['B']['versions']
            subprocess.run([sys.executable,'scripts/collect_c4_unseen_rules.py',
                '--manifest',str(ROOT/'manifests/c4_unseen_rules_v1_20260906.json'),
                '--a',str(roots['A']),'--b',str(roots['B']),
                '--tokenizer','/home/v-yuejiali/studentmodeling/models/Qwen3-32B',
                '--output',str(PREFIX)+'.summary.json'],cwd=ROOT,check=True)
            write('complete',summary=str(PREFIX)+'.summary.json',jobs=jobs)
        except Exception as exc:
            write('needs_attention',error=str(exc));raise


if __name__=='__main__':main()
