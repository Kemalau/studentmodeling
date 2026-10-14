from pathlib import Path
import copy
import hashlib
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from c4_format_diagnostic import FORMATS, MODES, aggregate, build, expected, render
from c4_freegen_protocol import CONDITIONS, render_prompt

ROOT = Path(__file__).resolve().parents[1]


class FormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build(ROOT / "manifests/c4_freegen_novel_seed42_20260905.json")

    def test_panel_unchanged_and_deterministic(self):
        original = json.loads((ROOT / "manifests/c4_freegen_novel_seed42_20260905.json").read_text())
        self.assertEqual(self.manifest, build(ROOT / "manifests/c4_freegen_novel_seed42_20260905.json"))
        for g, old in zip(self.manifest["groups"], original["groups"]):
            self.assertEqual(g["contexts"], old["contexts"])
            for q, oq in zip(g["queries"], old["queries"]):
                for k, v in oq.items():
                    self.assertEqual(q[k], v)
                self.assertEqual(set(q["candidate_labels"]), {q["donor_target"], q["recipient_target"], q["correct_target"]})

    def test_factorial_visibility_and_neither_exactly_original(self):
        for g in self.manifest["groups"]:
            for q in g["queries"]:
                for fmt, (candidates, correct) in FORMATS.items():
                    prompt = render(g, q, fmt)
                    query = prompt.split("<query>")[1]
                    self.assertEqual("Valid outputs:" in query, candidates)
                    self.assertEqual("Correct answer:" in query, correct)
                    self.assertEqual(prompt.count("<STUDENT_STATE>"), 1)
                    self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), q["prompt_sha256_by_format"][fmt])
                self.assertEqual(render(g, q, "neither"), render_prompt(g["contexts"]["recipient"], q["problem_text"]))

    def test_shards_complete_disjoint(self):
        shards = [expected(self.manifest, i) for i in range(3)]
        self.assertEqual([len(s) for s in shards], [1080] * 3)
        self.assertEqual(len(expected(self.manifest)), 3240)
        self.assertFalse(shards[0] & shards[1] or shards[0] & shards[2] or shards[1] & shards[2])
        self.assertEqual(set.union(*shards), expected(self.manifest))

    def test_summary_audit_contrasts_and_reject_missing(self):
        records = []
        for g in self.manifest["groups"]:
            for q in g["queries"]:
                for fmt in FORMATS:
                    for mode in MODES:
                        for c in CONDITIONS:
                            value = q["donor_target"] if c == "swap" and fmt == "both" else q["recipient_target"]
                            records.append({"group_idx": g["group_idx"], "query_idx": q["query_idx"], "format": fmt,
                                            "mode": mode, "condition": c, "donor_id": g["ids"]["donor"],
                                            "recipient_id": g["ids"]["recipient"], "prompt_sha256": q["prompt_sha256_by_format"][fmt],
                                            "text": q["candidate_targets"][value], "parsed": value,
                                            "matches_A": value == q["donor_target"], "matches_B": value == q["recipient_target"],
                                            "matches_correct": False, "scores": {a: -1. if a == value else -2. for a in q["candidate_labels"]}})
        r = aggregate(self.manifest, records, bootstrap=20)
        self.assertEqual(r["cells"]["both"]["freegen"]["all_queries"]["directed_minus_real_random"], 1)
        self.assertEqual(r["paired_contrasts"]["freegen:both-minus-candidates_only"]["D_specific"]["estimate"], 1)
        with self.assertRaises(ValueError):
            aggregate(self.manifest, records[:-1], bootstrap=20)
        with self.assertRaises(ValueError):
            aggregate(self.manifest, records + records[:1], bootstrap=20)
        bad = copy.deepcopy(records)
        bad[0]["prompt_sha256"] = "wrong"
        with self.assertRaises(AssertionError):
            aggregate(self.manifest, bad, bootstrap=20)


if __name__ == "__main__":
    unittest.main()
