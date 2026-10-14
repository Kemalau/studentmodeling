import unittest

from scripts.summarize_external_state_teacher_diagnostic import select


def row(weight, passed, t3, c4=0.2, t6=0.9):
    return {
        "consistency_weight": weight,
        "all_gates_pass": passed,
        "T3": {"semantic_accuracy": t3},
        "C4": {"D_specific": c4},
        "T6": {"retention_ratio": t6},
    }


class ExternalTeacherSelectionTest(unittest.TestCase):
    def test_never_selects_a_failing_arm(self):
        value = select([row(0.25, True, 0.3), row(1.0, False, 0.9)])
        self.assertEqual(value["selected_candidate"]["consistency_weight"], 0.25)
        self.assertEqual(value["pass_count"], 1)

    def test_reports_no_selection_when_joint_gate_fails(self):
        value = select([row(0.25, False, 0.4), row(1.0, False, 0.5)])
        self.assertFalse(value["any_all_gates_pass"])
        self.assertIsNone(value["selected_candidate"])


if __name__ == "__main__":
    unittest.main()
