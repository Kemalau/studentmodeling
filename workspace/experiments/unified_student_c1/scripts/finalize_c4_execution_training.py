"""Finish the existing C4 pilot with a full audit, irrespective of effect sign.

Never submits, cancels, resumes, or changes remote scientific work. Waits for
the existing native-watcher collector, checks its process while waiting, then
audits all outputs and renders the already prepared report.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'c4-execution-training-v1-seed42-recovery1-20260906'
PREFIX = ROOT / 'results/c4_execution_training_v1_recovery1_20260906'


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write(value):
    path = Path(str(PREFIX) + '.finalizer.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def ready(receipt, is_alive):
    """Completion requires collector evidence; live waiting is not failure."""
    if receipt['experiment'] != EXPERIMENT:
        raise RuntimeError('Unexpected monitored experiment')
    status = receipt['status']
    if status == 'complete':
        expected = Path(str(PREFIX) + '.summary.json')
        if Path(receipt.get('summary', '')) != expected:
            raise RuntimeError('Missing or unexpected collector summary')
        result = json.loads(expected.read_text())
        if result['status'] != 'complete':
            raise RuntimeError('Collector summary is not complete')
        return True
    if status == 'needs_attention':
        raise RuntimeError('Existing collector needs attention; inspect its receipt')
    if status not in {'watching', 'observation_error', 'collecting'}:
        raise RuntimeError('Unrecognized collector state: ' + status)
    if not is_alive(receipt['pid']):
        raise RuntimeError('Collector process absent; inspect remote state before recovery')
    return False


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.background:
        receipt = Path(str(PREFIX) + '.finalizer.json')
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['status'] in {'waiting', 'auditing'} and alive(previous['pid']):
                raise RuntimeError('Finalizer already alive')
            if previous['status'] == 'complete':
                raise RuntimeError('Finalizer already completed; verify existing artifacts')
        with Path(str(PREFIX) + '.finalizer.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps(dict(pid=child.pid, experiment=EXPERIMENT)))
        return
    base = dict(pid=os.getpid(), experiment=EXPERIMENT, started_utc=timestamp())
    write(dict(base, status='waiting'))
    try:
        while True:
            receipt = json.loads(Path(str(PREFIX) + '.status.json').read_text())
            if ready(receipt, alive):
                break
            time.sleep(30)
        write(dict(base, status='auditing'))
        manifest = ROOT / 'manifests/c4_execution_training_v1_20260906.json'
        downloads = ROOT / 'results/downloads' / EXPERIMENT
        audit = Path(str(PREFIX) + '.delivery_audit.json')
        report = ROOT / 'reports/c4_execution_training_results_20260906.md'
        subprocess.run([sys.executable, 'scripts/audit_c4_execution_delivery.py',
            '--manifest', str(manifest), '--downloads', str(downloads),
            '--tokenizer', '/home/v-yuejiali/studentmodeling/models/Qwen3-32B',
            '--output', str(audit)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, 'scripts/render_c4_execution_results.py',
            '--manifest', str(manifest), '--audit', str(audit),
            '--output', str(report)], cwd=ROOT, check=True)
        write(dict(base, status='complete', finished_utc=timestamp(),
            audit=str(audit), report=str(report)))
    except Exception as exc:
        write(dict(base, status='needs_attention', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
