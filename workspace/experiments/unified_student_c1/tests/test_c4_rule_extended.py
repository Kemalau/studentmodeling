"""Synthetic checks for supplemental diagnostics, not model-result evidence."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_c4_rule_followup_extended import details
from c4_rule_followup import aggregate, group_keys, record_id
from render_c4_rule_followup_report import report


def test_perfect_directed_imperfect_constant_and_controls():
    manifest=json.loads((ROOT/"manifests/c4_rule_followup_v2_seed42_20260905.json").read_text())
    records=[]
    for arm in ("oracle","menu"):
        for g in manifest[arm+"_groups"]:
            for key in group_keys(arm,g):
                qi,order,c=key[2:]
                q=g["queries"][max(qi,0)]
                y=q["donor_target"] if c in ("swap","same_donor_rule","donor_rule") else q["recipient_target"]
                records.append({"key":key,"text":f"<predicted_response>{y}</predicted_response>",
                                "parsed":y,"parse_type":"tagged","stop":"closing_tag",
                                "prompt_sha256":g["prompt_hashes"][record_id(key)]})
    result=details(manifest,records)
    assert result["menu_all"]["swap"]["joint_hits"]==46
    assert result["menu_all"]["random_real"]["joint_hits"]==0
    assert result["oracle_semantic_valid"]["donor_rule"]["donor_hits"]==69
    assert result["supplemental_contrasts"]["joint_swap_minus_random_real:donor"]["estimate"]==1
    assert result["supplemental_contrasts"]["joint_swap_minus_same_donor_rule:rule"]["estimate"]==0
    result.update(primary=aggregate(manifest,records,bootstrap=20),
                  token_text_audit_count=1071,cross_role_state_hash_disagreements={},
                  posthoc_numeric_format_sensitivity={"changes":[]})
    rendered=report(result)
    assert "1,071" in rendered and "23" in rendered
    assert "First directed-only joint success" in rendered
    assert "0/46" in rendered and "46/46" in rendered
    constant=copy.deepcopy(records)
    first={g["group_idx"]:g["queries"][0]["donor_target"] for g in manifest["menu_groups"]}
    for r in constant:
        if r["key"][0]=="menu" and r["key"][-1]=="swap":
            r["parsed"]=r["text"]=first[r["key"][1]]
    result=details(manifest,constant)
    assert result["menu_all"]["swap"]["donor_hits"]==46
    assert result["menu_all"]["swap"]["joint_hits"]==0
