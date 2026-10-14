"""Frozen 2x2 prompt-support diagnostic on the existing 81-query C4 panel."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

from c4_freegen_protocol import CONDITIONS, parse_output, sha, summarize

FORMATS = {"both": (True, True), "candidates_only": (True, False),
           "correct_only": (False, True), "neither": (False, False)}
MODES = ("candidate_score", "freegen")
SEED = 20260905


def render(group, query, fmt):
    candidates, correct = FORMATS[fmt]
    context = group["contexts"]["recipient"]
    question = "<query>\nProblem: " + query["problem_text"] + "\n"
    if correct:
        question += "Correct answer: " + query["correct_target"] + "\n"
    if candidates:
        question += "Valid outputs:\n" + "\n".join(query["candidate_targets"][a] for a in query["candidate_labels"]) + "\n"
    return context["seg_instruction"] + context["seg_history"] + "<STUDENT_STATE>\n" + question + "</query>\nAnswer:"


def build(source):
    original = json.loads(source.read_text())
    assert original["n_groups"] == 27 and original["n_queries"] == 81
    result = copy.deepcopy(original)
    result.update(protocol="c4_format_diagnostic_v1", parent_manifest_sha256=sha(source),
                  formats=FORMATS, modes=MODES, n_expected_records=81 * 4 * 2 * 5,
                  selection="all 81 frozen queries, no new filtering or model-output selection",
                  score="sum log probability of complete tagged candidate, as in original C4; not open generation",
                  hypothesis="paired candidate/correct-answer visibility effects and score-vs-generation differences; exploratory diagnosis, no new replication gate")
    for group in result["groups"]:
        for query in group["queries"]:
            labels = [query[k] for k in ("donor_target", "recipient_target", "correct_target")]
            assert len(set(labels)) == 3
            random.Random(f"{SEED}:{group['group_idx']}:{query['query_idx']}").shuffle(labels)
            query["candidate_labels"] = labels
            query["candidate_targets"] = {a: f"<predicted_response>{a}</predicted_response>" for a in labels}
            query["prompt_sha256_by_format"] = {f: hashlib.sha256(render(group, query, f).encode()).hexdigest() for f in FORMATS}
    return result


def key(row):
    return tuple(row[k] for k in ("group_idx", "query_idx", "format", "mode", "condition"))


def expected(manifest, shard=None, shards=3):
    return {(g["group_idx"], q["query_idx"], f, m, c)
            for g in manifest["groups"] if shard is None or g["group_idx"] % shards == shard
            for q in g["queries"] for f in FORMATS for m in MODES for c in CONDITIONS}


def paired_ci(values, bootstrap=10000):
    """Resample complete donor clusters, preserving all paired conditions."""
    units = list(values.values())
    rng = random.Random(SEED)
    draws = []
    for _ in range(bootstrap):
        sampled = [units[rng.randrange(len(units))] for _ in units]
        draws.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    draws.sort()
    return {"estimate": sum(map(sum, units)) / sum(map(len, units)),
            "donor_cluster_bootstrap_95ci": [draws[int(.025 * bootstrap)], draws[min(bootstrap - 1, int(.975 * bootstrap))]],
            "n_donors": len(units), "n_queries": sum(map(len, units))}


def aggregate(manifest, records, bootstrap=10000):
    keys = [key(r) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected(manifest):
        raise ValueError("Incomplete, duplicate, or unexpected factorial records")
    queries = {(g["group_idx"], q["query_idx"]): (g, q) for g in manifest["groups"] for q in g["queries"]}
    for r in records:
        g, q = queries[(r["group_idx"], r["query_idx"])]
        assert r["donor_id"] == g["ids"]["donor"]
        assert r["recipient_id"] == g["ids"]["recipient"]
        assert r["prompt_sha256"] == q["prompt_sha256_by_format"][r["format"]]
        if r["mode"] == "freegen":
            assert parse_output(r["text"])[0] == r["parsed"]
        else:
            scores = r["scores"]
            assert set(scores) == set(q["candidate_labels"])
            assert r["parsed"] == max(q["candidate_labels"], key=scores.get)
        for flag, target in (("matches_A", "donor_target"), ("matches_B", "recipient_target"), ("matches_correct", "correct_target")):
            assert r[flag] == (r["parsed"] == q[target])
    cells = {}
    indexed = {key(r): r for r in records}
    for f in FORMATS:
        cells[f] = {}
        for m in MODES:
            rs = [r for r in records if r["format"] == f and r["mode"] == m]
            cell = summarize(manifest, rs, bootstrap)
            cell["outcomes"] = {c: {"n": 81, "A": sum(r["matches_A"] for r in rs if r["condition"] == c),
                                   "B": sum(r["matches_B"] for r in rs if r["condition"] == c),
                                   "correct": sum(r["matches_correct"] for r in rs if r["condition"] == c)} for c in CONDITIONS}
            cells[f][m] = cell

    def compare(left, right, endpoint):
        values = defaultdict(list)
        for (gi, qi), (g, _) in queries.items():
            def measure(cell):
                fmt, mode = cell
                if endpoint == "native_B":
                    return int(indexed[(gi, qi, fmt, mode, "none")]["matches_B"])
                return (int(indexed[(gi, qi, fmt, mode, "swap")]["matches_A"])
                        - int(indexed[(gi, qi, fmt, mode, "random_real")]["matches_A"]))
            values[g["ids"]["donor"]].append(measure(left) - measure(right))
        return paired_ci(values, bootstrap)

    contrasts = {}
    for mode in MODES:
        for left, right in (("both", "candidates_only"), ("both", "correct_only"),
                            ("candidates_only", "neither"), ("correct_only", "neither")):
            contrasts[f"{mode}:{left}-minus-{right}"] = {e: compare((left, mode), (right, mode), e) for e in ("native_B", "D_specific")}
    for fmt in FORMATS:
        contrasts[f"{fmt}:score-minus-freegen"] = {e: compare((fmt, "candidate_score"), (fmt, "freegen"), e) for e in ("native_B", "D_specific")}
    return {"status": "complete", "n_records": len(records), "n_queries": 81, "n_donors": 27,
            "cells": cells, "paired_contrasts": contrasts,
            "interpretation": "Exploratory paired diagnostic; marginal CIs are not multiplicity-adjusted. Candidate scoring always uses a fixed 3-answer evaluation set, even when hidden from prompt. No claim of open execution from scoring."}


def collect(manifest_path, download, bootstrap=10000):
    manifest = json.loads(manifest_path.read_text())
    files = sorted(download.rglob("format_diagnostic/records.jsonl"))
    if len(files) != 3:
        raise ValueError(f"Expected 3 shards, found {len(files)}")
    records, fingerprints, numerical, state_hashes = [], [], [], {}
    for path in files:
        lineage = json.loads(path.with_name("lineage.json").read_text())
        assert lineage["manifest_sha256"] == sha(manifest_path)
        assert lineage["shards"] == 3
        shard_records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len(shard_records) == len({key(r) for r in shard_records})
        assert {key(r) for r in shard_records} == expected(manifest, lineage["shard"], 3)
        by_group = []
        for group_file in sorted(path.parent.glob("group_*.json")):
            group_data = json.loads(group_file.read_text())
            assert group_data["fingerprint"] == lineage
            by_group.extend(group_data["records"])
            numerical.extend(group_data.get("numerical_diagnostics", []))
            state_hashes[group_file.stem] = group_data.get("state_hashes", {})
        assert sorted(by_group, key=key) == sorted(shard_records, key=key)
        fingerprints.append(lineage)
        records.extend(shard_records)
    assert {p["shard"] for p in fingerprints} == {0, 1, 2}
    stable = [{k: v for k, v in p.items() if k != "shard"} for p in fingerprints]
    assert stable[0] == stable[1] == stable[2]
    result = aggregate(manifest, records, bootstrap)
    result.update(files=[{"path": str(p), "sha256": sha(p)} for p in files], fingerprints=fingerprints,
                  numerical_diagnostics=numerical, state_hashes=state_hashes)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--download", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source) if args.action == "build" else collect(args.manifest, args.download)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("n_queries",) if k in result}, indent=2))
