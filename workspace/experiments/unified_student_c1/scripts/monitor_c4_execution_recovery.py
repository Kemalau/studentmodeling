"""Monitor an already submitted C4 recovery and collect only small artifacts.

Never submits or cancels a job. Native AMLT terminal-only notification watch;
poll errors are not evidence of job failure. Scientific success requires audit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from collect_c4_execution_training import collect
from count_active_a100_requests import json_table

ROOT=Path(__file__).resolve().parents[1]
EXP='c4-execution-training-v1-seed42-recovery1-20260906'
PREFIX=ROOT/'results/c4_execution_training_v1_recovery1_20260906'

def write(status,**fields):
    path=Path(str(PREFIX)+'.status.json')
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(pid=os.getpid(),experiment=EXP,status=status,
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),indent=2)+'\n')
    temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');a=p.parse_args()
    if a.background:
        state=Path(str(PREFIX)+'.status.json')
        if state.exists():
            old=json.loads(state.read_text())
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Existing monitor PID is still live; inspect before restart')
        with Path(str(PREFIX)+'.monitor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,experiment=EXP)));return
    try:
        write('watching')
        # amlt watch handles temporary fetch failures. Its exit alone is not
        # sufficient: read authoritative state again before deciding terminal.
        while True:
            subprocess.run(['amlt','watch',EXP,'--json','--interval','60','--notify','terminal',
                '--state-file',str(PREFIX)+'.watch.json'],cwd=ROOT,check=False)
            try:
                jobs=json_table(['amlt','--json-tables','status',EXP,'--force-status-update'])
            except Exception as exc:
                write('observation_error',error=str(exc));time.sleep(60);continue
            states={str(j['status']).lower() for j in jobs}
            if len(jobs)==2 and states=={'pass'}:break
            if states & {'failed','killed','expired','cancelled','canceled'}:
                write('needs_attention',jobs=jobs);return
            write('watching',jobs=jobs);time.sleep(60)
        write('collecting')
        downloads=ROOT/'results/downloads'
        subprocess.run(['amlt','results','download',EXP,'--output',str(downloads),
            '--include','*.json','--include','*.jsonl','--include','*.log'],cwd=ROOT,check=True,timeout=600)
        summary=collect(ROOT/'manifests/c4_execution_training_v1_20260906.json',downloads/EXP)
        dest=Path(str(PREFIX)+'.summary.json');dest.write_text(json.dumps(summary,indent=2)+'\n')
        write('complete',summary=str(dest))
    except Exception as exc:
        write('needs_attention',error=str(exc));raise

if __name__=='__main__':main()
