"""Supplemental CPU audit; never alters frozen endpoints or remote inference.

Detailed subgroup and rule-cluster analyses are exploratory sensitivity checks.
The primary estimates are imported unchanged from the frozen collector.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from c4_freegen_protocol import sha
from c4_rule_followup import ORACLE_CONDITIONS, STATE_CONDITIONS, ci
from collect_c4_rule_followup import collect
from audit_c4_freegen_results import format_sensitivity


def details(m, records):
    idx = {tuple(r["key"]): r for r in records}

    def menu_table(groups):
        cells = {}
        for c in STATE_CONDITIONS:
            counts = Counter(n_queries=0, donor_hits=0, recipient_hits=0,
                             correct_hits=0, random_rule_hits=0, invalid=0,
                             off_menu=0, token_limit=0, joint_hits=0,
                             blank_invalid=0, blank_off_menu=0)
            for g in groups:
                for order in (0, 1):
                    hits = []
                    for q in g["queries"]:
                        r = idx[("menu", g["group_idx"], q["query_idx"], order, c)]
                        y = r["parsed"]
                        counts["n_queries"] += 1
                        for name, target in (("donor_hits", "donor_target"),
                                             ("recipient_hits", "recipient_target"),
                                             ("correct_hits", "correct_target"),
                                             ("random_rule_hits", "random_rule_target")):
                            counts[name] += y == q[target]
                        counts["invalid"] += y is None
                        counts["off_menu"] += y is not None and y not in g["menus"][order]
                        counts["token_limit"] += r["stop"] == "token_limit"
                        hits.append(y == q["donor_target"])
                    counts["joint_hits"] += all(hits)
                    blank = idx[("menu", g["group_idx"], -1, order, c)]["parsed"]
                    counts["blank_invalid"] += blank is None
                    counts["blank_off_menu"] += blank is not None and blank not in g["menus"][order]
            cells[c] = dict(counts, n_donors=len(groups), n_pair_orders=2*len(groups))
        return cells

    def oracle_table(groups):
        result = {}
        for c in ORACLE_CONDITIONS:
            counts = Counter(n=0, donor_hits=0, recipient_hits=0, correct_hits=0,
                             invalid=0, token_limit=0)
            for g in groups:
                for q in g["queries"]:
                    if not q["semantic_valid"]:
                        continue
                    r = idx[("oracle",g["group_idx"],q["query_idx"],0,c)]
                    counts["n"] += 1
                    for name,target in (("donor_hits","donor_target"),
                                        ("recipient_hits","recipient_target"),
                                        ("correct_hits","correct_target")):
                        counts[name] += r["parsed"] == q[target]
                    counts["invalid"] += r["parsed"] is None
                    counts["token_limit"] += r["stop"] == "token_limit"
            result[c] = dict(counts)
        return result

    result = {"label":"Supplemental exploratory analysis; frozen primary unchanged",
              "menu_all":menu_table(m["menu_groups"]),
              "oracle_semantic_valid":oracle_table(m["oracle_groups"])}
    supported=[g for g in m["menu_groups"] if all(q["random_rule_target"] in g["menus"][0] for q in g["queries"])]
    result["random_rule_both_targets_on_menu_subset"]={
        "label":"Exploratory; random-state target support is not guaranteed by original union menu",
        "groups":[g["group_idx"] for g in supported],"cells":menu_table(supported)}
    for name,gs,control in (
            ("random_rule_both_targets_on_menu_subset",supported,"random_real"),
            ("native_joint_B_subset",[g for g in m["menu_groups"] if all(
                idx[("menu",g["group_idx"],q["query_idx"],o,"none")]["parsed"]==q["recipient_target"]
                for q in g["queries"] for o in (0,1))],"random_real")):
        values=defaultdict(list)
        for g in gs:
            for o in (0,1):
                def joint(c):
                    return all(idx[("menu",g["group_idx"],q["query_idx"],o,c)]["parsed"]==q["donor_target"] for q in g["queries"])
                values[g["ids"]["donor"]].append(int(joint("swap"))-int(joint(control)))
        result.setdefault(name,{"label":"Exploratory outcome-conditioned diagnostic, not primary",
                               "groups":[g["group_idx"] for g in gs],"cells":menu_table(gs)})
        result[name]["joint_swap_minus_random_real"]=ci(values) if values else None
    for arm, table in (("menu", menu_table), ("oracle", oracle_table)):
        for dimension in ("rule", "family"):
            grouped = defaultdict(list)
            for g in m[arm+"_groups"]:
                grouped[g["labels"]["donor"] if dimension == "rule" else g["family"]].append(g)
            result[arm+"_by_"+dimension] = {k:table(v) for k,v in sorted(grouped.items())}

    contrasts = {}
    for other in ("random_real", "same_donor_rule"):
        for cluster in ("donor", "rule"):
            values = defaultdict(list)
            for g in m["menu_groups"]:
                for order in (0,1):
                    def joint(c):
                        return all(idx[("menu",g["group_idx"],q["query_idx"],order,c)]["parsed"] == q["donor_target"] for q in g["queries"])
                    unit = g["ids"]["donor"] if cluster == "donor" else g["labels"]["donor"]
                    values[unit].append(int(joint("swap"))-int(joint(other)))
            stat = ci(values)
            stat["cluster"] = cluster
            stat["n_clusters"] = stat.pop("n_donors")
            stat["cluster_95ci"] = stat.pop("donor_cluster_95ci")
            contrasts["joint_swap_minus_"+other+":"+cluster] = stat
    result["supplemental_contrasts"] = contrasts

    cases = []
    for g in m["menu_groups"]:
        case = {"group_idx":g["group_idx"], "labels":g["labels"],
                "menus":g["menus"], "random_pool_n":g["random_pool_n"],
                "queries":[{k:q[k] for k in ("problem_text","donor_target","recipient_target","correct_target")} for q in g["queries"]],
                "orders":[]}
        for order in (0,1):
            outputs = {}
            for c in STATE_CONDITIONS:
                rs = [idx[("menu",g["group_idx"],q,order,c)] for q in (0,1,-1)]
                outputs[c] = {"text":[r["text"] for r in rs],
                              "parsed":[r["parsed"] for r in rs],
                              "joint_A":all(r["parsed"]==q["donor_target"] for r,q in zip(rs,g["queries"]))}
            case["orders"].append(outputs)
        cases.append(case)
    result["all_menu_cases"] = cases
    result["all_oracle_cases"] = [
        {"group_idx":g["group_idx"],"query_idx":q["query_idx"],
         "semantic_valid":q["semantic_valid"],"labels":g["labels"],
         "problem":q["problem_text"],"A":q["donor_target"],"B":q["recipient_target"],
         "correct":q["correct_target"],
         "outputs":{c:idx[("oracle",g["group_idx"],q["query_idx"],0,c)]["text"] for c in ORACLE_CONDITIONS}}
        for g in m["oracle_groups"] for q in g["queries"]]
    return result


def audit(root, download, tokenizer_path):
    manifest = root/"manifests/c4_rule_followup_v2_seed42_20260905.json"
    m = json.loads(manifest.read_text())
    primary = collect(manifest, download)
    files = sorted(download.glob("*/rule_followup/records.jsonl"))
    records = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    assert sha(tokenizer_path/"tokenizer.json") == primary["fingerprints"][0]["tokenizer_sha256"]
    lookup = {(arm,g["group_idx"]):g for arm in ("oracle","menu") for g in m[arm+"_groups"]}
    states = defaultdict(set)
    for fp in primary["fingerprints"]:
        assert fp["adapter_sha256"] == m["adapter_sha256"]
        for name, digest in fp["code"].items():
            assert sha(root/"scripts"/name) == digest, name
    for r in records:
        assert tok.decode(r["tokens"], skip_special_tokens=False) == r["text"]
        assert 0 <= len(r["tokens"]) <= m["max_new_tokens"]
        assert r["stop"] in ("closing_tag","eos","token_limit")
        if r["stop"] == "token_limit":
            assert len(r["tokens"]) == m["max_new_tokens"]
        g = lookup[tuple(r["key"][:2])]
        assert r["donor_id"] == g["ids"]["donor"]
        if r["key"][0] == "menu":
            for role, digest in r["state_hashes"].items():
                states[g["contexts"][role]["example_id"]].add(digest)
        else:
            assert r["state_hashes"] == {}
    result = details(m, records)
    relaxed=copy.deepcopy(records)
    format_changes=[]
    for r in relaxed:
        value,kind=format_sensitivity(r["text"])
        if value != r["parsed"]:
            format_changes.append({"key":r["key"],"text":r["text"],"value":value,"kind":kind})
        r["parsed"]=value
    sensitivity=details(m,relaxed)
    result["posthoc_numeric_format_sensitivity"]={
        "label":"Mixed-number/kg parser from prior audit; NOT the frozen primary",
        "changes":format_changes,
        "oracle_semantic_valid":sensitivity["oracle_semantic_valid"],
        "menu_all":sensitivity["menu_all"],
        "contrasts":sensitivity["supplemental_contrasts"]}
    result.update(primary=primary,
                  token_text_audit_count=len(records),
                  code_hashes_match=True,
                  state_identity_count=len(states),
                  cross_role_state_hash_disagreements={k:sorted(v) for k,v in states.items() if len(v)>1})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--download",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,default=Path("/home/v-yuejiali/studentmodeling/models/Qwen3-32B"))
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    result = audit(a.root,a.download,a.tokenizer)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("token_text_audit_count","cross_role_state_hash_disagreements","supplemental_contrasts","oracle_semantic_valid","menu_all")},indent=2))


if __name__ == "__main__":
    main()
