"""Resume-safe local finalizer for the existing native AMLT watcher.

Never submits, cancels, or modifies remote work. Waits for the watcher receipt,
then audits complete outputs and renders a report; scientific success is not
required and never used as a stopping criterion.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PREFIX=ROOT/"results/c4_rule_followup_v2_seed42_20260905"


def write(value):
    path=Path(str(PREFIX)+".finalizer.json")
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--background",action="store_true");a=p.parse_args()
    if a.background:
        receipt=Path(str(PREFIX)+".finalizer.json")
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous.get("status") in ("waiting","auditing"):
                try: os.kill(previous["pid"],0)
                except ProcessLookupError: pass
                else: raise RuntimeError("Finalizer already alive")
        with Path(str(PREFIX)+".finalizer.log").open("a") as log:
            proc=subprocess.Popen([sys.executable,"-u",__file__],cwd=ROOT,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"pid":proc.pid}));return
    base={"pid":os.getpid(),"started":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write(dict(base,status="waiting"))
    try:
        while True:
            monitor=json.loads(Path(str(PREFIX)+".monitor.json").read_text())
            if monitor["status"]=="complete": break
            if monitor["status"]=="needs_attention":
                raise RuntimeError("Remote watcher needs attention: "+str(monitor.get("error")))
            try: os.kill(monitor["pid"],0)
            except ProcessLookupError: raise RuntimeError("Remote watcher disappeared")
            time.sleep(30)
        write(dict(base,status="auditing"))
        audit=Path(str(PREFIX)+".extended_audit.json")
        report=ROOT/"reports/c4_rule_followup_results_20260905.md"
        subprocess.run([sys.executable,"scripts/audit_c4_rule_followup_extended.py",
                        "--download","results/downloads/c4-rule-followup-seed42-v2-20260905",
                        "--output",str(audit)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,"scripts/render_c4_rule_followup_report.py",
                        "--audit",str(audit),"--output",str(report)],cwd=ROOT,check=True)
        write(dict(base,status="complete",audit=str(audit),report=str(report),
                   finished=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    except Exception as exc:
        write(dict(base,status="needs_attention",error=str(exc)))
        raise


if __name__=="__main__": main()
