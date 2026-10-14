"""Read-only raw-output audit; only the resulting local summary is written."""
import argparse
import json
from pathlib import Path
from c4_freegen_protocol import sha
from c4_rule_followup import aggregate, expected, group_keys


def collect(manifest_path,download,bootstrap=10000):
    m=json.loads(Path(manifest_path).read_text())
    files=sorted(Path(download).rglob("rule_followup/records.jsonl"))
    assert len(files)==3, f"Expected 3 shards, got {len(files)}"
    all_records, fps=[],[]
    for file in files:
        fp=json.loads(file.with_name("lineage.json").read_text())
        assert fp["manifest_sha256"]==sha(manifest_path) and fp["shards"]==3
        rs=[json.loads(s) for s in file.read_text().splitlines()]
        want={x for x in expected(m) if x[1]%3==fp["shard"]}
        assert len(rs)==len(want) and {tuple(r["key"]) for r in rs}==want
        group_rs=[]
        for arm,k in (("oracle","oracle_groups"),("menu","menu_groups")):
            for g in m[k]:
                if g["group_idx"]%3!=fp["shard"]: continue
                gr=json.loads((file.parent/f"{arm}_{g['group_idx']:03d}.json").read_text())
                assert gr["fingerprint"]==fp
                assert len(gr["records"])==len(group_keys(arm,g))
                if arm=="menu":
                    assert len(gr["state_hashes"])==5
                    assert all(r["state_hashes"]==gr["state_hashes"] for r in gr["records"])
                group_rs.extend(gr["records"])
        assert sorted(group_rs,key=lambda r:r["key"])==sorted(rs,key=lambda r:r["key"])
        fps.append(fp); all_records.extend(rs)
    assert {f["shard"] for f in fps}=={0,1,2}
    stable=[{k:v for k,v in f.items() if k!="shard"} for f in fps]
    assert stable[0]==stable[1]==stable[2]
    result=aggregate(m,all_records,bootstrap)
    result["files"]=[{"path":str(f),"sha256":sha(f)} for f in files]
    result["fingerprints"]=fps
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--download",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=collect(a.manifest,a.download)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
