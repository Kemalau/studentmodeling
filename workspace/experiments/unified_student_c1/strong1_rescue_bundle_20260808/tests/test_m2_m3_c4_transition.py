from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m2_m3_c4_transition import CONDITIONS, load, render  # noqa: E402
from validate_m2_m3_c4_transition_amlt import validate  # noqa: E402


class TransitionAuditTest(unittest.TestCase):
    def test_rendered_config_is_fail_closed(self):
        value = render(load(ROOT / "amlt/feeds_recurrent_student_m4_eval_20260730.yaml"))
        validate(value)
        self.assertEqual(len(value["jobs"]), 3)
        by_name = {job["name"]: job for job in value["jobs"]}
        self.assertEqual(set(by_name), {spec["job"] for spec in CONDITIONS.values()})
        for condition, spec in CONDITIONS.items():
            job = by_name[spec["job"]]
            self.assertEqual(job["submit_args"]["env"]["CONDITION"], condition)
            self.assertNotIn("export RECEIPT=", job["command"])
            self.assertIn("export BASE_DATA=/mnt/amulet/studentmodeling/", job["command"])
            self.assertIn('mkdir -p "$$HF_HUB_CACHE"', job["command"])

    def test_explicit_recovery_subset_is_validated(self):
        value = render(load(ROOT / "amlt/feeds_recurrent_student_m4_eval_20260730.yaml"))
        selected = "m2-cf1-native-c4-seed42"
        value["jobs"] = [job for job in value["jobs"] if job["name"] == selected]
        validate(value, {selected})


if __name__ == "__main__":
    unittest.main()
