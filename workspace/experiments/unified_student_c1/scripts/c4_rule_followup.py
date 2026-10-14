"""CPU protocol for explicit-rule execution and fixed-menu query dependence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

from c4_freegen_protocol import canonical, numeric_tokens, parse_output, sha

SEED = 2026090517
RULES = {
    "subtraction.smaller_from_larger": "Align the two integers by place value. In every column subtract the smaller digit from the larger digit, regardless of which is on top. Never borrow. Join the resulting digits.",
    "subtraction.borrow_no_decrement": "Subtract column by column from right to left. If the top digit is smaller, add ten to it, but do not decrement any digit to its left. Otherwise subtract normally. Join the resulting digits.",
    "subtraction.always_borrow_left": "Work right to left. In EVERY column except the leftmost, add ten to the current top digit and decrement the immediately adjacent top digit to its left, even when borrowing is unnecessary. Write only the units digit of that column's difference. In the leftmost column subtract normally, writing zero if the result is negative.",
    "subtraction.borrow_from_bottom": "Work right to left. When the current top digit is too small, add ten to it by borrowing from the BOTTOM number's higher digits, leaving the top higher digits unchanged. Find the nearest nonzero bottom digit to the left, decrement it, and turn intervening bottom zeros into nines. If no such bottom digit exists, write the absolute difference in the current column. Otherwise subtract the current digits normally.",
    "subtraction.diff_0_n_equals_n": "Work right to left. If the ORIGINAL top digit is zero and the bottom digit is nonzero, write that bottom digit without borrowing. For every other column, do ordinary subtraction with normal cascading borrowing through zeros; keep track of changes to the top digits.",
    "subtraction.stops_borrow_at_zero": "Work right to left. If the current top digit is smaller than the bottom digit and the immediately adjacent top digit to its left is zero, do not borrow: write the absolute difference in this column. If that adjacent top digit is nonzero, decrement it and add ten to the current top digit. In the leftmost column use the absolute difference when needed. Otherwise subtract normally.",
    "fractions.ignore_denominators": "Convert mixed numbers to improper fractions without otherwise reducing the written fractions. Ignore the denominators. Apply the question's operation to the first two numerators only and return an integer. For division use integer floor division of the first numerator by the second. Do not use any later operands.",
    "fractions.multiply_across_for_division": "Convert mixed numbers to improper fractions. For a division problem, multiply the two fractions directly WITHOUT inverting the second fraction: multiply their numerators and multiply their denominators, then simplify.",
    "fractions.keep_common_denominator_for_multiplication": "Convert mixed numbers to improper fractions. When multiplying fractions with a common denominator, multiply all their numerators but keep the common denominator only once. Simplify the resulting fraction.",
    "fractions.common_denominator_numerator": "For fraction addition, add all the written numerators together and separately add all the written denominators together. Form a fraction from those two sums and simplify; do not find a common denominator.",
    "fractions.natural_number_bias_numerator_only": "Compare the first two fractions by their numerators alone, ignoring their denominators. Return the fraction with the larger numerator, or equal if the numerators are equal. This is a comparison rule, not a subtraction algorithm.",
    "decimals.longer_is_larger": "For comparing decimals, choose the original number with the most digits after its decimal point. With two numbers and equal decimal lengths, choose the numerically larger. With more than two numbers, break a longest-length tie by choosing the first such number.",
    "decimals.whole_number_thinking": "For comparing two decimals, treat the digits after each decimal point as an integer. Choose the original decimal whose fractional-part integer is larger. Break ties by choosing the second decimal. Ignore the whole-number part in this comparison.",
}
STATE_CONDITIONS = ("none", "noop", "swap", "same_donor_rule", "same_recipient_rule", "random_real")
ORACLE_CONDITIONS = ("donor_rule", "recipient_rule", "no_rule")


def independent_answer(rule, p):
    """Independent numeric implementation, not generator metadata target fields."""
    family, name = rule.split(".")
    if family == "subtraction":
        a, b = map(int, p.operands[:2])
        top = list(map(int, str(a)))[::-1]
        bot = list(map(int, str(b).zfill(len(top))))[::-1]
        original, out = top[:], []
        for i in range(len(top)):
            x, y = top[i], bot[i]
            if name == "smaller_from_larger":
                z = abs(x-y)
            elif name == "borrow_no_decrement":
                z = x-y if x >= y else x+10-y
            elif name == "always_borrow_left":
                if i+1 < len(top):
                    top[i+1] -= 1
                    z = (x+10-y) % 10
                else:
                    z = max(0, x-y)
            elif name == "diff_0_n_equals_n" and original[i] == 0 and y > 0:
                z = y
            elif name in ("borrow_from_bottom", "diff_0_n_equals_n") and x < y:
                digits = bot if name == "borrow_from_bottom" else top
                j = i+1
                while j < len(digits) and digits[j] == 0:
                    j += 1
                if j < len(digits):
                    digits[j] -= 1
                    for k in range(i+1, j):
                        digits[k] = 9
                    z = x+10-y
                else:
                    z = abs(x-y)
            elif name == "stops_borrow_at_zero" and x < y:
                if i+1 < len(top) and top[i+1] != 0:
                    top[i+1] -= 1
                    z = x+10-y
                else:
                    z = abs(x-y)
            else:
                z = x-y
            out.append(z)
        return str(sum(d * 10**i for i,d in enumerate(out)))
    if family == "decimals":
        values = list(p.operands)
        if name == "longer_is_larger":
            lens = [len(str(x).split(".")[1]) if "." in str(x) else 0 for x in values]
            idx = lens.index(max(lens))
            if len(values) == 2 and lens[0] == lens[1]:
                idx = 0 if values[0] >= values[1] else 1
        else:
            nums = [int(str(x).split(".")[1] or "0") if "." in str(x) else 0 for x in values[:2]]
            idx = 0 if nums[0] > nums[1] else 1
        return canonical(values[idx])
    a,b,c,d = map(int,p.operands[:4])
    if name == "ignore_denominators":
        op = p.metadata.get("operation",p.operation)
        v = a-c if "subtraction" in op else a*c if "multiplication" in op else a//c if "division" in op else a+c
        return str(v)
    if name == "multiply_across_for_division":
        return str(Fraction(a*c,b*d))
    if name == "keep_common_denominator_for_multiplication":
        # Operands are written improper numerators/denominators in this panel.
        nums, dens = p.operands[::2], p.operands[1::2]
        assert len(set(dens)) == 1
        n = 1
        for x in nums:
            n *= int(x)
        return str(Fraction(n,int(dens[0])))
    if name == "common_denominator_numerator":
        return str(Fraction(sum(p.operands[::2]),sum(p.operands[1::2])))
    if name == "natural_number_bias_numerator_only":
        return str(Fraction(a,b)) if a > c else str(Fraction(c,d)) if c > a else "equal"
    raise ValueError(rule)


def applicable(rule, p):
    if rule.startswith("subtraction."):
        return p.operation == "subtraction" and len(p.operands) == 2
    if rule == "fractions.ignore_denominators":
        return p.operation.startswith("fraction_") and any(x in p.operation for x in ("addition","subtraction","multiplication","division"))
    if rule.endswith("multiply_across_for_division"):
        return p.operation == "fraction_division"
    if rule.endswith("keep_common_denominator_for_multiplication"):
        return p.operation == "fraction_multiplication"
    if rule.endswith("common_denominator_numerator"):
        return p.operation == "fraction_addition"
    if rule.endswith("natural_number_bias_numerator_only"):
        return p.operation == "fraction_comparison"
    return rule.startswith("decimals.") and p.operation == "decimal_comparison" and len(p.operands) == 2


def context(row):
    return {k:row[k] for k in ("example_id","seg_instruction","seg_history")}


def build(root):
    from prepare_malrule_c1 import discover_malrules, generate_problem, difficulty_enum, solve_malrule_answer
    root = Path(root)
    source = root / "manifests/c4_freegen_novel_seed42_20260905.json"
    old = json.loads(source.read_text())
    repo = root / "data/raw/malrulelib"
    infos = {i.malrule_id:i for i in discover_malrules(repo,["subtraction","fractions","decimals"])}
    rows_path = root / "data/processed/malrule_c1_clean/test_same_template.jsonl"
    rows = {r["example_id"]:r for r in map(json.loads,rows_path.open())}
    out = {"protocol":"c4_rule_followup_v2", "seed":SEED, "adapter_sha256":old["adapter_sha256"],
           "parent_manifest_sha256":sha(source), "history_pool_sha256":sha(rows_path),
           "source_lineage":old["source_lineage"], "rule_library_sha256":old["rule_library_sha256"],
           "max_new_tokens":48, "rules":RULES, "oracle_groups":[], "menu_groups":[], "exclusions":[],
           "scope":"Exploratory follow-up; no retraining, no independent replication or rule-execution claim from visible-menu results."}
    for oldg in old["groups"]:
        g = copy.deepcopy(oldg)
        r = rows[g["ids"]["recipient"]]
        owner = infos[g["family"]+"."+r["query"]["surface_owner"]]
        problems = []
        for q in g["queries"]:
            p,_ = generate_problem(owner.generator_cls,q["generation_seed"],difficulty_enum(repo,"MEDIUM"),[9])
            assert p.text == q["problem_text"]
            problems.append(p)
            q["semantic_audit"] = {}
            for role in ("donor","recipient"):
                label = g["labels"][role]
                legacy = canonical(solve_malrule_answer(infos[label].malrule_cls,p))
                assert legacy == q[role+"_target"]
                reference = independent_answer(label,p)
                q["semantic_audit"][role] = {"applicable":applicable(label,p), "reference_target":reference,
                                                   "agrees_with_legacy":reference == legacy}
            q["semantic_valid"] = all(a["applicable"] and a["agrees_with_legacy"] for a in q["semantic_audit"].values())
        out["oracle_groups"].append(g)
        # Fixed, pre-inference menu panel: first two queries, arithmetic only.
        if g["family"] == "decimals" or not all(q["semantic_valid"] for q in g["queries"][:2]):
            out["exclusions"].append({"group_idx":g["group_idx"],"reason":"decimal selection task or semantic-rule/target mismatch", "audit":[q["semantic_audit"] for q in g["queries"][:2]]})
            continue
        mg = copy.deepcopy(g)
        mg["queries"] = mg["queries"][:2]
        reserved = set(g["ids"].values())
        reserved_histories = {c["seg_history"] for c in g["contexts"].values()}
        a = g["labels"]["donor"]
        same = sorted((r for r in rows.values() if r.get("task") == "malrule_next_response" and r.get("metadata",{}).get("malrule_id") == a and r["example_id"] not in reserved and r["seg_history"] not in reserved_histories),key=lambda r:r["example_id"])
        if not same:
            raise ValueError(f"No independent same-donor-rule history {a}")
        same = same[0]
        reserved.add(same["example_id"])
        reserved_histories.add(same["seg_history"])
        pool = []
        for r in sorted(rows.values(),key=lambda r:r["example_id"]):
            label = r.get("metadata",{}).get("malrule_id")
            if r.get("task") != "malrule_next_response" or label == a or label not in RULES or r["example_id"] in reserved or r["seg_history"] in reserved_histories:
                continue
            if not all(applicable(label,p) for p in problems[:2]):
                continue
            try:
                targets = [canonical(solve_malrule_answer(infos[label].malrule_cls,p)) for p in problems[:2]]
                if targets != [independent_answer(label,p) for p in problems[:2]]:
                    continue
                if any(t == q["donor_target"] for t,q in zip(targets,mg["queries"])):
                    continue
            except (ValueError,KeyError,ZeroDivisionError,IndexError):
                continue
            pool.append((r,targets))
            reserved_histories.add(r["seg_history"])
        if not pool:
            raise ValueError(f"No operation-applicable random history: group {g['group_idx']}")
        rr, rt = random.Random(SEED+g["group_idx"]).choice(pool)
        mg["contexts"]["same_donor_rule"] = context(same)
        mg["contexts"]["random_real"] = context(rr)
        mg["ids"]["same_donor_rule"] = same["example_id"]
        mg["ids"]["random_real"] = rr["example_id"]
        mg["labels"]["same_donor_rule"] = a
        mg["labels"]["random_real"] = rr["metadata"]["malrule_id"]
        mg["random_pool_n"] = len(pool)
        for q,t in zip(mg["queries"],rt):
            q["random_rule_target"] = t
        menu = sorted({q[k] for q in mg["queries"] for k in ("donor_target","recipient_target","correct_target")})
        random.Random(SEED*2+g["group_idx"]).shuffle(menu)
        mg["menus"] = [menu,list(reversed(menu))]
        assert mg["queries"][0]["donor_target"] != mg["queries"][1]["donor_target"]
        numbers = set(numeric_tokens(" ".join(c["seg_history"] for c in mg["contexts"].values())))
        mg["all_control_history_answer_exclusion"] = all(q["donor_target"] not in numbers for q in mg["queries"])
        out["menu_groups"].append(mg)
    out["n_oracle_queries"] = sum(len(g["queries"]) for g in out["oracle_groups"])
    out["n_semantic_valid_oracle_queries"] = sum(q["semantic_valid"] for g in out["oracle_groups"] for q in g["queries"])
    out["n_menu_groups"] = len(out["menu_groups"])
    out["n_expected_records"] = len(expected(out))
    for arm,groups in (("oracle",out["oracle_groups"]),("menu",out["menu_groups"])):
        for g in groups:
            g["prompt_hashes"] = {record_id(x):hashlib.sha256(prompt(out,g,*x[2:]).encode()).hexdigest() for x in group_keys(arm,g)}
    return out


def group_keys(arm,g):
    if arm == "oracle":
        return [(arm,g["group_idx"],q["query_idx"],0,c) for q in g["queries"] for c in ORACLE_CONDITIONS]
    return [(arm,g["group_idx"],q,o,c) for q in (0,1,-1) for o in (0,1) for c in STATE_CONDITIONS]


def expected(m):
    return {x for arm,k in (("oracle","oracle_groups"),("menu","menu_groups")) for g in m[k] for x in group_keys(arm,g)}


def record_id(x):
    return ":".join(map(str,x))


def prompt(m,g,qi,order,condition):
    instruction = g["contexts"]["recipient"]["seg_instruction"]
    if condition in ORACLE_CONDITIONS:
        rule = "The student's rule is unspecified. Predict the student's answer."
        if condition != "no_rule":
            rule = "Follow this student's rule exactly, even if it is mathematically incorrect: " + m["rules"][g["labels"][condition.removesuffix("_rule")]]
        return (instruction+"<student_history>\n</student_history>\n<STUDENT_STATE>\n<query>\nStudent rule: "+rule+
                "\nProblem: "+g["queries"][qi]["problem_text"]+"\n</query>\nAnswer:")
    question = "[Problem withheld.]" if qi == -1 else g["queries"][qi]["problem_text"]
    outputs = "\n".join(f"<predicted_response>{a}</predicted_response>" for a in g["menus"][order])
    return instruction+g["contexts"]["recipient"]["seg_history"]+"<STUDENT_STATE>\n<query>\nProblem: "+question+"\nValid outputs:\n"+outputs+"\n</query>\nAnswer:"


def ci(values,n=10000):
    rng = random.Random(SEED)
    units = list(values.values())
    def avg(us):
        return sum(sum(x) for x in us)/sum(map(len,us))
    ds = sorted(avg([rng.choice(units) for _ in units]) for _ in range(n))
    return {"estimate":avg(units),"donor_cluster_95ci":[ds[int(.025*n)],ds[min(n-1,int(.975*n))]],"n_donors":len(units),"n":sum(map(len,units))}


def aggregate(m,records,bootstrap=10000):
    keys = [tuple(r["key"]) for r in records]
    assert len(keys) == len(set(keys)) and set(keys) == expected(m), "Missing/duplicate/unexpected records"
    idx = dict(zip(keys,records))
    lookup = {(arm,g["group_idx"]):g for arm,k in (("oracle","oracle_groups"),("menu","menu_groups")) for g in m[k]}
    for key,r in idx.items():
        g = lookup[key[:2]]
        assert r["prompt_sha256"] == g["prompt_hashes"][record_id(key)]
        assert parse_output(r["text"]) == (r["parsed"],r["parse_type"])
    oracle = {}
    for subset in ("all_legacy81","semantic_valid","semantic_valid_arithmetic"):
        selected = [(g,q) for g in m["oracle_groups"] for q in g["queries"] if subset == "all_legacy81" or q["semantic_valid"] and (subset != "semantic_valid_arithmetic" or g["family"] != "decimals")]
        cells = {}
        for c in ORACLE_CONDITIONS:
            rs = [idx[("oracle",g["group_idx"],q["query_idx"],0,c)] for g,q in selected]
            cells[c] = {"n":len(rs),"donor_hits":sum(r["parsed"]==q["donor_target"] for r,(_,q) in zip(rs,selected)),
                        "recipient_hits":sum(r["parsed"]==q["recipient_target"] for r,(_,q) in zip(rs,selected)),
                        "invalid":sum(r["parsed"] is None for r in rs),"token_limit":sum(r["stop"]=="token_limit" for r in rs)}
        contrasts = {}
        for control in ("recipient_rule","no_rule"):
            v = defaultdict(list)
            for g,q in selected:
                get = lambda c: idx[("oracle",g["group_idx"],q["query_idx"],0,c)]["parsed"] == q["donor_target"]
                v[g["ids"]["donor"]].append(int(get("donor_rule"))-int(get(control)))
            contrasts["donor_minus_"+control] = ci(v,bootstrap)
        oracle[subset] = {"cells":cells,"contrasts":contrasts}
    menu = {}
    for subset in ("all","all_control_history_answer_exclusion"):
        gs = [g for g in m["menu_groups"] if subset == "all" or g[subset]]
        cells = {}
        for c in STATE_CONDITIONS:
            hits, joint, blanks, invalid = 0,0,0,0
            for g in gs:
                for o in (0,1):
                    rs = [idx[("menu",g["group_idx"],q,o,c)] for q in (0,1)]
                    flags = [r["parsed"] == q["donor_target"] for r,q in zip(rs,g["queries"])]
                    hits += sum(flags); joint += all(flags); invalid += sum(r["parsed"] is None for r in rs)
                    blank = idx[("menu",g["group_idx"],-1,o,c)]["parsed"]
                    blanks += sum(blank == q["donor_target"] for q in g["queries"])
            cells[c] = {"queries":4*len(gs),"donor_hits":hits,"query_pair_orders":2*len(gs),"joint_hits":joint,
                        "queryless_donor_hits":blanks,"queryless_joint_hits":0,"invalid":invalid}
        contrasts = {}
        for c in ("random_real","noop","same_recipient_rule"):
            v = defaultdict(list)
            for g in gs:
                for o in (0,1):
                    def joint(cond):
                        return all(idx[("menu",g["group_idx"],q["query_idx"],o,cond)]["parsed"]==q["donor_target"] for q in g["queries"])
                    v[g["ids"]["donor"]].append(int(joint("swap"))-int(joint(c)))
            contrasts["joint_swap_minus_"+c] = ci(v,bootstrap) if v else None
        disagreements = {"noop_native_text":0,"noop_native_A":0,"donor_same_rule_output":0,"swap_order_output":0}
        for g in gs:
            for q in (0,1):
                get = lambda o,c: idx[("menu",g["group_idx"],q,o,c)]
                disagreements["swap_order_output"] += get(0,"swap")["parsed"] != get(1,"swap")["parsed"]
                for o in (0,1):
                    disagreements["noop_native_text"] += get(o,"none")["text"] != get(o,"noop")["text"]
                    disagreements["noop_native_A"] += (get(o,"none")["parsed"]==g["queries"][q]["donor_target"]) != (get(o,"noop")["parsed"]==g["queries"][q]["donor_target"])
                    disagreements["donor_same_rule_output"] += get(o,"swap")["parsed"] != get(o,"same_donor_rule")["parsed"]
        menu[subset] = {"n_donors":len(gs),"cells":cells,"contrasts":contrasts,"disagreements":disagreements}
    return {"status":"complete","n_records":len(records),"oracle":oracle,"menu":menu,
            "interpretation":"Exploratory fixed checkpoint diagnostic, not new independent replication. Joint fixed-menu success rejects constant answer/position preference only, not all question-dependent heuristics or independent open execution."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    m = build(args.root)
    args.output.write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps({k:v for k,v in m.items() if k.startswith("n_")}))
    print(json.dumps(m["exclusions"],indent=2))


if __name__ == "__main__":
    main()
