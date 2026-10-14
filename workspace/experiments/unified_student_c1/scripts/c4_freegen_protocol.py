"""CPU-only construction and scoring for the frozen C4 novel-number audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

SEED = 20260905
CONDITIONS = ("none", "noop", "swap", "random_real", "samemal")
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    text = str(value).strip().replace("−", "-")
    if not re.fullmatch(NUMBER, text):
        raise ValueError(f"Not a scalar numeric answer: {text!r}")
    return str(Fraction(re.sub(r"\s+", "", text)))


def parse_output(text):
    """One numeric tagged response or one bare number; never cherry-pick numbers."""
    match = re.fullmatch(r"\s*<predicted_response>\s*(.*?)\s*</predicted_response>\s*", text, re.S)
    try:
        return canonical(match.group(1) if match else text), "tagged" if match else "bare"
    except (ValueError, ZeroDivisionError):
        return None, "invalid"


def numeric_tokens(text):
    values = []
    for match in re.finditer(NUMBER, text):
        try:
            values.append(canonical(match.group()))
        except ZeroDivisionError:
            values.append(match.group())
    return tuple(values)


def signatures(problem):
    operands = tuple(canonical(x) for x in problem.get("operands", []))
    numbers = numeric_tokens(problem["problem_text"])
    # Do not allow rewording, swapped order, or equivalent numeric spellings to
    # make an already-seen number combination count as unseen.
    return {("operands", tuple(sorted(operands))), ("text", tuple(sorted(numbers)))} - {
        ("operands", ()), ("text", ())
    }


def render_prompt(row, question):
    return (row["seg_instruction"] + row["seg_history"] + "<STUDENT_STATE>\n"
            + "<query>\nProblem: " + question + "\n</query>\nAnswer:")


def build(args):
    from prepare_malrule_c1 import discover_malrules, difficulty_enum, generate_problem
    from prepare_malrule_c1 import solve_answer, solve_malrule_answer, is_consistent_with_metadata

    root = args.root.resolve()
    source = root / "data/processed/malrule_c1_clean"
    paths = [root / "data/processed/state_curriculum_stage_a_v1" / name
             for name in ("train.jsonl", "valid.jsonl")]
    paths += [source / name for name in ("train.jsonl", "val.jsonl", "test_same_template.jsonl", "test_cross_template.jsonl")]
    seen, training_answers, lineage = set(), set(), []
    for path in paths:
        count = 0
        for line in path.open():
            row = json.loads(line)
            count += 1
            for problem in [*row.get("history", []), row["query"]]:
                try:
                    seen.update(signatures(problem))
                except (ValueError, ZeroDivisionError):
                    # Some nonnumeric comparison operands cannot be canonicalized;
                    # their rendered numeric combinations remain auditable.
                    seen.add(("text", tuple(sorted(numeric_tokens(problem["problem_text"])))))
            if path == paths[0]:
                try:
                    training_answers.add(canonical(row["target_label"]))
                except (ValueError, ZeroDivisionError):
                    pass
        lineage.append({"path": str(path.relative_to(root)), "sha256": sha(path), "rows": count})
    assert lineage[0]["sha256"] == "929487329576607572aa49b8e2daba53605b3c733749e7e631baa376a9df65f9"
    rows = {r["example_id"]: r for r in map(json.loads, (source / "test_same_template.jsonl").open())}
    manifest_path = root / "manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
    pairs = json.loads(manifest_path.read_text())["pairs"]
    # Match the existing real-random donor identities; ignore every model output.
    previous = json.loads(args.previous.read_text())
    random_ids = {r["pair_idx"]: r["real_random_id"] for r in previous["records"] if r["condition"] == "none"}
    repo = root / "data/raw/malrulelib"
    infos = {i.malrule_id: i for i in discover_malrules(repo, ["subtraction", "decimals", "fractions"])}
    rng = random.Random(SEED)
    chosen, per_rule, used_donors = [], Counter(), set()
    for pair_idx, pair in enumerate(pairs, 1):
        label = rows[pair["donor_id"]]["metadata"]["malrule_id"]
        if pair["donor_id"] not in used_donors and per_rule[label] < 3:
            chosen.append((pair_idx, pair))
            used_donors.add(pair["donor_id"])
            per_rule[label] += 1
    groups, excluded, used_queries = [], [], set()
    for pair_idx, pair in chosen:
        ids = {"donor": pair["donor_id"], "recipient": pair["recipient_id"],
               "samemal": pair["samemal_id"], "random_real": random_ids[pair_idx]}
        contexts = {k: rows[v] for k, v in ids.items()}
        labels = {k: r["metadata"]["malrule_id"] for k, r in contexts.items()}
        recipient = contexts["recipient"]
        family = recipient["metadata"]["family"]
        owner = infos[family + "." + recipient["query"]["surface_owner"]]
        visible_numbers = set()
        for row in contexts.values():
            visible_numbers.update(numeric_tokens(row["seg_history"]))
        queries, answers, reasons = [], set(), Counter()
        for attempt in range(10000):
            generation_seed = rng.randrange(1_000_000_000)
            try:
                problem, level = generate_problem(owner.generator_cls, generation_seed,
                                                 difficulty_enum(repo, "MEDIUM"), [9])
                correct = canonical(solve_answer(owner.correct_cls, problem))
                if not is_consistent_with_metadata(problem, solve_answer(owner.correct_cls, problem)):
                    reasons["metadata_inconsistent"] += 1
                    continue
                raw = {"problem_text": problem.text, "operands": [str(x) for x in problem.operands],
                       "operation": problem.operation}
                sig = signatures(raw)
                if sig & (seen | used_queries):
                    reasons["seen_number_combination"] += 1
                    continue
                targets = {}
                for key in ("donor", "recipient"):
                    info = infos[labels[key]]
                    if canonical(solve_answer(info.correct_cls, problem)) != correct:
                        raise ValueError("incompatible domain")
                    targets[key] = canonical(solve_malrule_answer(info.malrule_cls, problem))
                a, b = targets["donor"], targets["recipient"]
                if len({a, b, correct}) != 3:
                    reasons["non_distinct_answers"] += 1
                    continue
                if a in visible_numbers or b in visible_numbers or a in answers:
                    reasons["history_copy_or_repeated_target"] += 1
                    continue
                try:
                    info = infos[labels["random_real"]]
                    if canonical(solve_answer(info.correct_cls, problem)) != correct:
                        raise ValueError("random rule outside domain")
                    random_target = canonical(solve_malrule_answer(info.malrule_cls, problem))
                except Exception:
                    random_target = None
                queries.append({**raw, "query_idx": len(queries), "generation_seed": generation_seed,
                                "level": level, "donor_target": a, "recipient_target": b,
                                "correct_target": correct, "random_rule_target": random_target,
                                "target_seen_in_finetuning_answers": a in training_answers,
                                "number_combination_unseen": True, "answer_exclusion_pass": True})
                used_queries.update(sig)
                answers.add(a)
                if len(queries) == 3:
                    break
            except Exception as exc:
                reasons[type(exc).__name__] += 1
        if len(queries) != 3:
            excluded.append({"pair_idx": pair_idx, "ids": ids, "labels": labels,
                             "reason": "fewer_than_three_eligible_numeric_queries", "eligible": len(queries),
                             "filter_counts": dict(reasons)})
            print(f"exclude pair={pair_idx} eligible={len(queries)}", flush=True)
            continue
        groups.append({"group_idx": len(groups), "pair_idx": pair_idx, "ids": ids, "labels": labels,
                       "family": family, "contexts": {k: {"example_id": r["example_id"],
                       "seg_instruction": r["seg_instruction"], "seg_history": r["seg_history"]}
                       for k, r in contexts.items()}, "queries": queries, "filter_counts": dict(reasons)})
        print(f"group={len(groups)} pair={pair_idx} rule={labels['donor']}", flush=True)
    result = {"protocol": "c4_freegen_novel_v1", "seed": SEED, "conditions": CONDITIONS,
              "rule_library_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "rule_library_sha256": {str(p.relative_to(repo)): sha(p) for p in sorted((repo / "malrules").rglob("*.py"))
                                      if p.relative_to(repo).parts[1] in ("subtraction", "decimals", "fractions")},
              "selection": "first unique donor in frozen manifest; up to 3 donors per rule; no model-output selection",
              "novelty": "unordered canonical operand/numeric-text combinations absent from frozen fine-tuning, validation, and both test histories/queries; not a pretraining novelty claim",
              "max_new_tokens": 48, "decoding": "greedy, no prefill, no candidates, no query correct answer",
              "source_lineage": lineage, "pair_manifest_sha256": sha(manifest_path),
              "random_donor_source_sha256": sha(args.previous),
              "adapter_sha256": "c9b463ed12f11ef84d346c1399d2588b9b396ecb9e15c5f4ca85c7f1d7fe991f",
              "groups": groups, "excluded_groups": excluded, "n_selected": len(chosen),
              "n_groups": len(groups), "n_queries": 3 * len(groups),
              "hypothesis": "positive directed-minus-real-random exact numeric agreement; donor-cluster bootstrap; no comparison to candidate-score 0.411"}
    assert groups
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("n_selected", "n_groups", "n_queries")}, indent=2))


def summarize(manifest, records, bootstrap=10000):
    expected = {(g["group_idx"], q["query_idx"], c) for g in manifest["groups"]
                for q in g["queries"] for c in CONDITIONS}
    keys = [(r["group_idx"], r["query_idx"], r["condition"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicate, or unexpected generation records")
    by_query = defaultdict(dict)
    for record in records:
        by_query[(record["group_idx"], record["query_idx"])][record["condition"]] = record

    def endpoint(clean=False, compatible=False, unseen_answer=False, family=None):
        selected = [cs for cs in by_query.values()
                    if (not clean or cs["none"]["matches_B"])
                    and (not compatible or cs["none"].get("random_rule_applicable", False))
                    and (not unseen_answer or not cs["none"].get("target_seen_in_finetuning_answers", True))
                    and (family is None or cs["none"].get("family") == family)]
        if not selected:
            return {"n": 0, "estimate": None}
        grouped = defaultdict(list)
        for cs in selected:
            grouped[cs["none"]["donor_id"]].append(int(cs["swap"]["matches_A"]) - int(cs["random_real"]["matches_A"]))
        units = list(grouped.values())
        rng = random.Random(SEED)
        draws = []
        for _ in range(bootstrap):
            sample = [units[rng.randrange(len(units))] for _ in units]
            draws.append(sum(map(sum, sample)) / sum(map(len, sample)))
        draws.sort()
        n = len(selected)
        return {"n": n, "n_donors": len(units),
                "rates": {c: sum(cs[c]["matches_A"] for cs in selected) / n for c in CONDITIONS},
                "directed_minus_real_random": sum(map(sum, units)) / n,
                "donor_cluster_bootstrap_95ci": [draws[int(.025 * bootstrap)], draws[min(bootstrap-1, int(.975 * bootstrap))]]}

    groups = defaultdict(list)
    for cs in by_query.values():
        groups[cs["none"]["group_idx"]].append(cs)
    return {"status": "complete", "n_records": len(records), "n_queries": len(by_query),
            "n_donor_groups": len(groups), "all_queries": endpoint(False),
            "native_B_subset": endpoint(True),
            "random_rule_applicable_subset": endpoint(compatible=True),
            "random_rule_applicable_native_B_subset": endpoint(clean=True, compatible=True),
            "target_string_unseen_in_finetuning_answers_subset": endpoint(unseen_answer=True),
            "by_family": {family: endpoint(family=family) for family in sorted({r.get("family", "unknown") for r in records})},
            "all_three_match_A": {c: sum(all(cs[c]["matches_A"] for cs in items) for items in groups.values()) for c in CONDITIONS},
            "parse_failures": {c: sum(r["parsed"] is None for r in records if r["condition"] == c) for c in CONDITIONS},
            "noop_text_disagreements": sum(cs["none"]["text"] != cs["noop"]["text"] for cs in by_query.values()),
            "baseline_B_accuracy": sum(cs["none"]["matches_B"] for cs in by_query.values()) / len(by_query)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", type=Path, default=Path.cwd())
    b.add_argument("--previous", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--records", type=Path, nargs="+", required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        records = [json.loads(line) for path in args.records for line in path.open() if line.strip()]
        result = summarize(json.loads(args.manifest.read_text()), records)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
