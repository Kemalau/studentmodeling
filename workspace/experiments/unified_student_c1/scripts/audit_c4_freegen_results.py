"""Read frozen C4 outputs; verify lineage and score without changing the protocol.

The optional numeric-format sensitivity analysis is explicitly post hoc. It
never replaces the frozen primary endpoint or drops a failed generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re

from c4_freegen_protocol import CONDITIONS, parse_output, sha, summarize


def format_sensitivity(text):
    value, kind = parse_output(text)
    if value is not None:
        return value, kind
    tag = re.fullmatch(r"\s*<predicted_response>\s*(.*?)\s*</predicted_response>\s*", text, re.S)
    if not tag:
        return None, "invalid"
    body = tag.group(1).strip()
    # Accept a single mixed number, never an equation or several guesses.
    mixed = re.fullmatch(r"([+-]?)(\d+)\s+(\d+)\s*/\s*(\d+)", body)
    if mixed:
        sign, whole, numerator, denominator = mixed.groups()
        if int(denominator) == 0 or int(numerator) >= int(denominator):
            return None, "invalid"
        answer = Fraction(int(whole)) + Fraction(int(numerator), int(denominator))
        return str(-answer if sign == "-" else answer), "posthoc_mixed_number"
    unit = re.fullmatch(r"(.+?)\s+kg", body)
    if unit:
        answer, _ = parse_output(unit.group(1))
        if answer is not None:
            return answer, "posthoc_kg_suffix"
    return None, "invalid"


def audit(root, download):
    manifest_path = root / "manifests/c4_freegen_novel_seed42_20260905.json"
    manifest = json.loads(manifest_path.read_text())
    expected_hashes = {
        "manifest_sha256": sha(manifest_path),
        "adapter_sha256": manifest["adapter_sha256"],
        "evaluator_sha256": sha(root / "scripts/evaluate_c4_freegen_novel.py"),
        "protocol_sha256": sha(root / "scripts/c4_freegen_protocol.py"),
        "swap_helpers_sha256": sha(root / "scripts/state_swap_misconception_smoke.py"),
    }
    files = sorted(download.glob("*/freegen/records.jsonl"))
    assert len(files) == 3, f"Need three complete shards, found {len(files)}"
    records, fingerprints = [], []
    for file in files:
        fingerprint = json.loads(file.with_name("lineage.json").read_text())
        for key, value in expected_hashes.items():
            assert fingerprint[key] == value, (file, key)
        assert fingerprint["shards"] == 3
        assert fingerprint["max_new_tokens"] == manifest["max_new_tokens"]
        shard_records = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        assert len(shard_records) == 135
        assert all(r["group_idx"] % 3 == fingerprint["shard"] for r in shard_records)
        group_records = []
        for group_file in sorted(file.parent.glob("group_*.json")):
            saved = json.loads(group_file.read_text())
            assert saved["fingerprint"] == fingerprint
            group_records.extend(saved["records"])
        key = lambda r: (r["group_idx"], r["query_idx"], r["condition"])
        assert sorted(group_records, key=key) == sorted(shard_records, key=key)
        records.extend(shard_records)
        fingerprints.append(fingerprint)
    assert sorted(f["shard"] for f in fingerprints) == [0, 1, 2]
    assert len({json.dumps({k: v for k, v in f.items() if k != "shard"}, sort_keys=True)
                for f in fingerprints}) == 1, "Cross-shard lineage mismatch"
    queries = {(g["group_idx"], q["query_idx"]): (g, q)
               for g in manifest["groups"] for q in g["queries"]}
    by_query = defaultdict(dict)
    relaxed, format_changes, invalid = [], [], Counter()
    for record in records:
        group, query = queries[(record["group_idx"], record["query_idx"])]
        parsed, kind = parse_output(record["text"])
        assert (parsed, kind) == (record["parsed"], record["parse_type"])
        assert record["donor_id"] == group["ids"]["donor"]
        assert record["recipient_id"] == group["ids"]["recipient"]
        assert record["donor_rule"] == group["labels"]["donor"]
        for flag, target in (("matches_A", "donor_target"), ("matches_B", "recipient_target"),
                             ("matches_correct", "correct_target")):
            assert record[target] == query[target]
            assert record[flag] == (parsed == query[target])
        assert len(record["tokens"]) <= manifest["max_new_tokens"]
        by_query[(record["group_idx"], record["query_idx"])][record["condition"]] = record
        if parsed is None:
            invalid[record["text"]] += 1
        value, format_kind = format_sensitivity(record["text"])
        revised = dict(record, parsed=value,
                       matches_A=value == query["donor_target"],
                       matches_B=value == query["recipient_target"],
                       matches_correct=value == query["correct_target"])
        relaxed.append(revised)
        if value != parsed:
            format_changes.append({"group_idx": record["group_idx"], "query_idx": record["query_idx"],
                                   "condition": record["condition"], "text": record["text"],
                                   "parsed": value, "kind": format_kind,
                                   "matches_A": revised["matches_A"]})
    assert all(len({r["prompt_sha256"] for r in cs.values()}) == 1 for cs in by_query.values())
    frozen = summarize(manifest, records)
    noop = []
    examples = []
    for idx, cs in sorted(by_query.items()):
        group, query = queries[idx]
        sample = {"group_idx": idx[0], "query_idx": idx[1], "problem": query["problem_text"],
                  "donor_rule": group["labels"]["donor"], "donor_target": query["donor_target"],
                  "recipient_target": query["recipient_target"], "correct_target": query["correct_target"],
                  "outputs": {c: cs[c]["text"] for c in CONDITIONS}}
        if cs["none"]["text"] != cs["noop"]["text"]:
            noop.append(sample)
        if cs["swap"]["matches_A"] or cs["random_real"]["matches_A"]:
            examples.append(sample)
    arithmetic = [r for r in records if queries[(r["group_idx"], r["query_idx"])][1]["operation"] != "decimal_comparison"]
    arithmetic_manifest = dict(manifest, groups=[dict(g, queries=[q for q in g["queries"]
                                     if q["operation"] != "decimal_comparison"])
                                     for g in manifest["groups"] if any(q["operation"] != "decimal_comparison" for q in g["queries"])])
    return {
        "audit": "complete_three_shard_lineage_raw_text_and_flag_validation",
        "files": [{"path": str(f), "sha256": sha(f)} for f in files],
        "fingerprints": fingerprints,
        "frozen_primary": frozen,
        "raw_outcomes": {c: {"n": sum(r["condition"] == c for r in records),
                              **{flag: sum(r[flag] for r in records if r["condition"] == c)
                                 for flag in ("matches_A", "matches_B", "matches_correct")},
                              "token_limit": sum(r["stop"] == "token_limit" for r in records if r["condition"] == c)}
                         for c in CONDITIONS},
        "noop_disagreements": noop,
        "donor_target_hits_all_conditions": examples,
        "invalid_outputs": [{"text": text, "count": count} for text, count in invalid.most_common()],
        "posthoc_format_sensitivity": {"label": "post hoc; mixed numbers and single kg suffix only; not primary",
                                       "changed_records": format_changes, "summary": summarize(manifest, relaxed)},
        "posthoc_arithmetic_only": {"label": "post hoc; excludes decimal_comparison queries, whose answer can be an operand",
                                    "summary": summarize(arithmetic_manifest, arithmetic)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.download)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("audit", "frozen_primary", "raw_outcomes")}, indent=2))
