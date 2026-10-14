"""Native AMLT terminal monitor and exact collector. No submit or auto-retry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from c4_history_diagnostic import ROOT, MANIFEST
from count_active_a100_requests import json_table
from submit_c4_history_diagnostic import EXP, PREFIX


def write(status,**fields):
    path=Path(str(PREFIX)+'.status.json');tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),experiment=EXP,
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true');a=p.parse_args()
    if a.background:
        with Path(str(PREFIX)+'.monitor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,experiment=EXP,status_file=str(PREFIX)+'.status.json')));return
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
                if len(jobs)==8 and states=={'pass'}:break
                if states & {'failed','killed','expired','cancelled','canceled'}:
                    write('needs_attention',jobs=jobs);return
                write('watching',jobs=jobs);time.sleep(60)
            write('collecting',jobs=jobs)
            downloads=ROOT/'results/downloads'
            subprocess.run(['amlt','results','download',EXP,'--output',str(downloads),
                '--include','*.json','--include','*.log'],cwd=ROOT,check=True,timeout=900)
            subprocess.run([sys.executable,'scripts/collect_c4_history_diagnostic.py',
                '--manifest',str(ROOT/MANIFEST),'--root',str(downloads/EXP),
                '--tokenizer','/home/v-yuejiali/studentmodeling/models/Qwen3-32B',
                '--output',str(PREFIX)+'.summary.json',
                '--report',str(ROOT/'reports/c4_history_diagnostic_results_20260907.md')],cwd=ROOT,check=True)
            summary=json.loads(Path(str(PREFIX)+'.summary.json').read_text())
            assert summary['status']=='complete' and summary['n_records']==5376
            write('complete',summary=str(PREFIX)+'.summary.json',jobs=jobs)
        except Exception as exc:
            write('needs_attention',error=str(exc));raise


if __name__=='__main__':main()
