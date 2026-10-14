"""Post-hoc semantic-valid sensitivity; preserves original 81-query analysis."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

from c4_freegen_protocol import sha
from c4_format_diagnostic import collect, key, paired_ci


def audit(root):
    download=root/"results/downloads/c4-format-diagnostic-seed42-v1-20260905"
    original=collect(root/"manifests/c4_format_diagnostic_seed42_20260905.json",download)
    saved=json.loads((root/"results/c4_format_diagnostic_seed42_20260905.summary.json").read_text())
    assert original=={k:v for k,v in saved.items() if k!="experiment"}
    followup=json.loads((root/"manifests/c4_rule_followup_v2_seed42_20260905.json").read_text())
    groups=followup["oracle_groups"]
    valid={(g["group_idx"],q["query_idx"]) for g in groups for q in g["queries"] if q["semantic_valid"]}
    records=[json.loads(s) for f in sorted(download.glob("*/format_diagnostic/records.jsonl")) for s in f.read_text().splitlines()]
    idx={key(r):r for r in records}
    from transformers import AutoTokenizer
    tokenizer_path=root.parents[2]/"models/Qwen3-32B"
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    assert sha(tokenizer_path/"tokenizer.json")==original["fingerprints"][0]["tokenizer_sha256"]
    for fp in original["fingerprints"]:
        for name,digest in fp["code"].items():
            assert sha(root/"scripts"/name)==digest
    for r in records:
        if r["mode"]=="freegen":
            assert tok.decode(r["tokens"],skip_special_tokens=False)==r["text"]
    result={"label":"Post-hoc semantic-valid sensitivity on previously observed panel, not new replication",
            "n_queries":len(valid),"n_donors":len({g for g,q in valid}),
            "original_3240_records_reaggregate_exact_match":True,"cells":{},
            "excluded_query_keys":sorted(set((g["group_idx"],q["query_idx"]) for g in groups for q in g["queries"])-valid)}
    for fmt in ("both","candidates_only","correct_only","neither"):
        result["cells"][fmt]={}
        for mode in ("candidate_score","freegen"):
            cells={}
            for c in ("none","noop","swap","random_real","samemal"):
                rs=[idx[(gi,qi,fmt,mode,c)] for gi,qi in sorted(valid)]
                cells[c]={"n":len(rs),"A":sum(r["matches_A"] for r in rs),
                          "B":sum(r["matches_B"] for r in rs),"correct":sum(r["matches_correct"] for r in rs),
                          "invalid":sum(r["parsed"] is None for r in rs)}
            values=defaultdict(list)
            for g in groups:
                for q in g["queries"]:
                    gi,qi=g["group_idx"],q["query_idx"]
                    if (gi,qi) not in valid: continue
                    get=lambda c: int(idx[(gi,qi,fmt,mode,c)]["matches_A"])
                    values[g["ids"]["donor"]].append(get("swap")-get("random_real"))
            result["cells"][fmt][mode]={"counts":cells,"contrast":paired_ci(values)}
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
