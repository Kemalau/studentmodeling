from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_paired_state_swap_manifest import build_pairs  # noqa: E402


def row(
    example_id: str,
    rule: str,
    question_id: str,
    *,
    answer_for_r1: str | None = None,
) -> dict:
    answers = {} if answer_for_r1 is None else {"r1": answer_for_r1}
    return {
        "example_id": example_id,
        "task": "malrule_next_response",
        "target_label": "B",
        "candidate_labels": ["A", "B", "C"],
        "query": {"correct_answer": "C", "problem_text": question_id},
        "seg_question": f"<query>{question_id}</query>",
        "metadata": {
            "family": "toy",
            "malrule_id": rule,
            "question_id": question_id,
            "query_answers_by_malrule": answers,
        },
    }


class PairedStateSwapManifestTest(unittest.TestCase):
    def test_multi_query_curriculum_reuses_each_donor_across_distinct_queries(self) -> None:
        rows = [row("donor", "r1", "q-donor")]
        rows.extend(
            row(f"recipient-{index}", "r2", f"q-{index}", answer_for_r1="A")
            for index in range(5)
        )
        # This duplicate student row must not count as a fifth distinct query.
        rows.append(row("recipient-duplicate", "r3", "q-0", answer_for_r1="A"))

        pairs, stats = build_pairs(rows, seed=17, copies_per_donor=4)
        donor_pairs = [pair for pair in pairs if pair["donor_id"] == "donor"]

        self.assertEqual(len(donor_pairs), 4)
        self.assertEqual(len({pair["recipient_query_key"] for pair in donor_pairs}), 4)
        self.assertEqual(stats["copies_per_donor"], 4)
        self.assertGreaterEqual(stats["max_distinct_queries_per_paired_donor"], 4)

    def test_default_keeps_one_pair_per_donor(self) -> None:
        rows = [row("donor", "r1", "q-donor")]
        rows.extend(
            row(f"recipient-{index}", "r2", f"q-{index}", answer_for_r1="A")
            for index in range(4)
        )
        pairs, stats = build_pairs(rows, seed=19)
        counts = Counter(pair["donor_id"] for pair in pairs)
        self.assertEqual(counts["donor"], 1)
        self.assertEqual(stats["copies_per_donor"], 1)

    def test_full_set_filter_excludes_partial_donors(self) -> None:
        rows = [row("donor", "r1", "q-donor")]
        rows.extend(
            row(f"recipient-{index}", "r2", f"q-{index}", answer_for_r1="A")
            for index in range(3)
        )
        pairs, stats = build_pairs(
            rows, seed=23, copies_per_donor=4, require_full_set=True
        )
        self.assertFalse(any(pair["donor_id"] == "donor" for pair in pairs))
        self.assertTrue(stats["require_full_set"])

    def test_minimum_query_filter_keeps_multi_query_partial_donors(self) -> None:
        rows = [row("donor", "r1", "q-donor")]
        rows.extend(
            row(f"recipient-{index}", "r2", f"q-{index}", answer_for_r1="A")
            for index in range(2)
        )
        pairs, stats = build_pairs(
            rows,
            seed=29,
            copies_per_donor=4,
            min_copies_per_donor=2,
        )
        donor_pairs = [pair for pair in pairs if pair["donor_id"] == "donor"]
        self.assertEqual(len(donor_pairs), 2)
        self.assertEqual(stats["min_copies_per_donor"], 2)

    def test_invalid_minimum_query_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between one"):
            build_pairs([], seed=31, copies_per_donor=4, min_copies_per_donor=5)

    def test_donor_label_cap_builds_a_balanced_diagnostic(self) -> None:
        rows = []
        for rule in ("r1", "r2"):
            for donor_index in range(3):
                rows.append(row(f"{rule}-donor-{donor_index}", rule, f"{rule}-q-donor-{donor_index}"))
        for query_index in range(4):
            recipient = row(
                f"recipient-{query_index}",
                "r3",
                f"q-{query_index}",
                answer_for_r1="A",
            )
            recipient["metadata"]["query_answers_by_malrule"]["r2"] = "A"
            rows.append(recipient)
        pairs, stats = build_pairs(
            rows,
            seed=37,
            copies_per_donor=2,
            min_copies_per_donor=2,
            max_donors_per_label=1,
        )
        counts = Counter(pair["donor_label"] for pair in pairs)
        self.assertEqual(counts["r1"], 2)
        self.assertEqual(counts["r2"], 2)
        self.assertEqual(stats["max_donors_per_label"], 1)

    def test_rejects_invalid_copy_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_pairs([], seed=1, copies_per_donor=0)


if __name__ == "__main__":
    unittest.main()
