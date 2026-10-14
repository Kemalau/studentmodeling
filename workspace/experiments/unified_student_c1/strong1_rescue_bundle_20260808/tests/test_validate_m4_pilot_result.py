from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_m4_pilot_result import validate  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def formal(groups: dict) -> dict:
    return {
        "information_route": "sequential_updates_then_continuous_state_plus_current_query",
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "metrics_by_group": groups,
    }


class M4PilotResultTest(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        write(
            root / "eval/t1_t6/metrics.json",
            formal(
                {
                    "predict_correctness/source/protocol": {
                        "n": 500,
                        "accuracy": 0.60,
                        "macro_f1": 0.55,
                    },
                    "predict_error_persistence/source/protocol": {
                        "n": 100,
                        "macro_f1": 0.50,
                        "auroc": 0.60,
                    },
                }
            ),
        )
        write(
            root / "eval/t2/metrics.json",
            formal(
                {
                    "predict_response/source/protocol": {
                        "n": 300,
                        "math_equivalence": 0.40,
                        "format_valid_rate": 0.95,
                    }
                }
            ),
        )
        write(
            root / "eval/t3/metrics.json",
            formal(
                {
                    "diagnose_misconception/source/protocol": {
                        "n": 300,
                        "format_valid_rate": 0.90,
                    }
                }
            ),
        )
        write(
            root / "eval/t5/metrics.json",
            formal(
                {
                    "predict_prospective_error/source/protocol": {
                        "n": 300,
                        "joint_math_equivalence": 0.50,
                        "format_valid_rate": 0.90,
                    }
                }
            ),
        )
        c4 = {
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "intervention_target": "final_continuous_recurrent_state",
            "summary": {
                "clean_n": 100,
                "D_specific": 0.20,
                "D_damage": 0.01,
                "scientific_gate_pass": True,
                "clean": {
                    "noop": {"flip_to_A_rate": 0.01},
                    "samemal": {"flip_to_A_rate": 0.05},
                },
            },
        }
        write(root / "eval/c4/recurrent_causal_gate.json", c4)
        return temporary, root

    def test_valid_strict_pilot_passes(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        self.assertTrue(validate(root)["pass"])

    def test_history_leak_fails(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        path = root / "eval/t2/metrics.json"
        payload = json.loads(path.read_text())
        payload["raw_history_visible_to_decoder"] = True
        write(path, payload)
        with self.assertRaisesRegex(ValueError, "Raw-history leak"):
            validate(root)

    def test_c4_specificity_and_task_collapse_fail(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        c4_path = root / "eval/c4/recurrent_causal_gate.json"
        c4 = json.loads(c4_path.read_text())
        c4["summary"]["D_specific"] = 0.02
        c4["summary"]["scientific_gate_pass"] = False
        write(c4_path, c4)
        t2_path = root / "eval/t2/metrics.json"
        t2 = json.loads(t2_path.read_text())
        next(iter(t2["metrics_by_group"].values()))["math_equivalence"] = 0.01
        write(t2_path, t2)
        with self.assertRaisesRegex(ValueError, "T2_noncollapsed.*C4_specificity"):
            validate(root)


if __name__ == "__main__":
    unittest.main()
