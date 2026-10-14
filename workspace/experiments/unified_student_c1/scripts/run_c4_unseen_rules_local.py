"""Local inference-only queue. Never submits remotely or stops another job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/datadisk/v-yuejiali/studentmodeling/c4_unseen_rules_20260906')
MODEL = Path('/home/v-yuejiali/studentmodeling/models/Qwen3-32B')
MANIFEST = ROOT/'manifests/c4_unseen_rules_v1_20260906.json'
EXP = 'c4-execution-training-v1-seed42-recovery1-20260906'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def adapter(arm):
    return OUT/'checkpoints'/EXP/f'c4-execution-arm-{arm.lower()}-seed42'/f'execution_{arm}'/'final_adapter'


def write(status,**fields):
    value=dict(status=status,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               scope='Local inference only; A then B; no training, no remote submission, no process cancellation',**fields)
    tmp=OUT/'status.json.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(OUT/'status.json')


def ready(free_mib,process_rows):
    return free_mib>=45000 and not process_rows.strip()


def capacity():
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip())
    processes=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    return free,processes


def wait_idle(phase):
    deadline=time.monotonic()+24*3600; stable=0
    while time.monotonic()<deadline:
        free,processes=capacity()
        stable=stable+1 if ready(free,processes) else 0
        write('waiting_for_local_gpu',next_phase=phase,free_mib=free,other_gpu_processes=processes,
              stable_idle_checks=stable)
        if stable>=2:return
        time.sleep(30)
    raise RuntimeError('Local GPU not idle within 24 hours; no other process was stopped')


def run_eval(arm,preflight=False):
    phase=f'{arm}_preflight' if preflight else f'{arm}_full'
    wait_idle(phase)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_c4_unseen_rules.py'),
        '--model',str(MODEL),'--adapter',str(adapter(arm)),'--manifest',str(MANIFEST),
        '--output',str(OUT/arm),'--arm',arm,'--runtime-report',str(OUT/f'{phase}.runtime.json')]
    if preflight:cmd+=['--max-new-groups','1']
    write('running',phase=phase,command=cmd)
    with (OUT/f'{phase}.log').open('a') as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=4*3600)
    if preflight:
        report=json.loads((OUT/f'{phase}.runtime.json').read_text())
        timing=report['groups'][0]['seconds']
        write('preflight_complete',runtime=report,
            estimated_total_generation_minutes=timing*64/60,
            estimate_note='First group only; includes capture, excludes future loading; not a completion promise')


def main():
    p=argparse.ArgumentParser();p.add_argument('--background',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.background:
        with (OUT/'supervisor.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,status_file=str(OUT/'status.json'))));return
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            m=json.loads(MANIFEST.read_text())
            for arm in ('A','B'):
                assert sha(adapter(arm)/'adapter_model.safetensors')==m['checkpoints'][arm]['adapter_sha256']
                assert (adapter(arm)/'adapter_config.json').is_file()
            for name in ('config.json','tokenizer.json','model.safetensors.index.json'):
                assert (MODEL/name).is_file()
            write('preflight_inputs_verified',manifest_sha256=sha(MANIFEST),
                  adapters={arm:str(adapter(arm)) for arm in ('A','B')})
            env=os.environ
            env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            if not (OUT/'A').exists():run_eval('A',preflight=True)
            for arm in ('A','B'):
                if not (OUT/arm/'complete.json').exists():run_eval(arm)
            write('auditing')
            subprocess.run([sys.executable,str(ROOT/'scripts/collect_c4_unseen_rules.py'),
                '--manifest',str(MANIFEST),'--a',str(OUT/'A'),'--b',str(OUT/'B'),
                '--tokenizer',str(MODEL),'--output',str(OUT/'summary.json')],cwd=ROOT,check=True)
            result=json.loads((OUT/'summary.json').read_text())
            assert result['status']=='complete' and result['n_records']==1728
            write('complete',summary=str(OUT/'summary.json'),n_records=1728)
        except Exception as exc:
            write('needs_attention',error=str(exc));raise


if __name__=='__main__':main()
