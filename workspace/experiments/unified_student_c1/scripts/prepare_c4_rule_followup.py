"""Derive config from an accepted run, restrict uploads, freeze file hashes."""
import copy
import hashlib
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
FILES=["requirements-feeds.txt","scripts/copy_model_parallel.py","scripts/state_swap_misconception_smoke.py",
       "scripts/c4_freegen_protocol.py","scripts/evaluate_c4_freegen_novel.py","scripts/c4_format_diagnostic.py",
       "scripts/evaluate_c4_format_diagnostic.py","scripts/c4_rule_followup.py","scripts/evaluate_c4_rule_followup.py",
       "scripts/run_c4_rule_followup.sh","tests/test_c4_freegen_decode.py","tests/test_c4_rule_followup.py",
       "manifests/c4_rule_followup_v2_seed42_20260905.json","reports/c4_rule_followup_protocol_20260905.md","reports/c4_rule_followup_v2_amendment_20260905.md"]
HASHFILE="manifests/c4_rule_followup_v2_20260905.sha256"


def main():
    c=yaml.safe_load((ROOT/"amlt/feeds_c4_format_diagnostic_seed42_20260905.yaml").read_text())
    assert c["target"]["name"]=="Feeds"
    assert c["environment"]["image"]=="amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
    c["description"]="Frozen C4 explicit-rule positive controls and same-menu two-query state dependence; no training"
    ignore=["*","!requirements-feeds.txt"]
    for d in ("scripts","tests","manifests","reports"):
        ignore += [f"!{d}/",f"{d}/*"]+["!"+p for p in FILES+[HASHFILE] if p.startswith(d+"/")]
    c["code"]["ignore"]=ignore+["__pycache__/","*.pyc"]
    for i,j in enumerate(c["jobs"]):
        assert j["sku"]=="80G1-A100" and j["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"]=="rankfun_aml"
        j["name"]=f"c4-rule-followup-seed42-shard{i}"
        j["command"]=f"bash scripts/run_c4_rule_followup.sh {i}"
    dest=ROOT/"amlt/feeds_c4_rule_followup_v2_seed42_20260905.yaml"
    dest.write_text("# Derived from accepted feeds_c4_format_diagnostic_seed42_20260905.yaml.\n"+yaml.safe_dump(c,sort_keys=False))
    (ROOT/HASHFILE).write_text("".join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()+"  "+p+"\n" for p in FILES))
    print(dest)
    print(f"Frozen {len(FILES)} files, {sum((ROOT/p).stat().st_size for p in FILES)} bytes")


if __name__=="__main__": main()
