"""Native AMLT terminal watch, download and strict audit; never submits jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collect_c4_rule_followup import collect

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="c4-rule-followup-seed42-v2-20260905"
PREFIX=ROOT/"results/c4_rule_followup_v2_seed42_20260905"


def write(suffix,value):
    p=Path(str(PREFIX)+suffix)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(value,indent=2)+"\n")
    t.replace(p)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--background",action="store_true"); a=ap.parse_args()
    if a.background:
        receipt=Path(str(PREFIX)+".monitor.json")
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get("status")=="monitoring":
                try: os.kill(old["pid"],0)
                except ProcessLookupError: pass
                else: raise RuntimeError("Monitor already running")
        with Path(str(PREFIX)+".monitor.log").open("a") as log:
            proc=subprocess.Popen([sys.executable,"-u",__file__],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"pid":proc.pid,"experiment":EXPERIMENT})); return
    write(".monitor.json",{"status":"monitoring","pid":os.getpid(),"started":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"experiment":EXPERIMENT})
    try:
        subprocess.run(["amlt","watch",EXPERIMENT,"--json","--interval","60","--notify","terminal","--state-file",str(PREFIX)+".watch.json"],cwd=ROOT,check=True)
        subprocess.run(["amlt","results","download",EXPERIMENT,"--output",str(ROOT/"results/downloads")],cwd=ROOT,check=True,timeout=300)
        summary=collect(ROOT/"manifests/c4_rule_followup_v2_seed42_20260905.json",ROOT/"results/downloads"/EXPERIMENT)
        write(".summary.json",summary)
        write(".monitor.json",{"status":"complete","experiment":EXPERIMENT})
    except Exception as exc:
        write(".monitor.json",{"status":"needs_attention","experiment":EXPERIMENT,"error":str(exc)})
        raise


if __name__=="__main__": main()
