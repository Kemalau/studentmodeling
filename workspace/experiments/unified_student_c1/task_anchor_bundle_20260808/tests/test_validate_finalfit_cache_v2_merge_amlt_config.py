from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_finalfit_cache_v2_merge_amlt_config import load, validate  # noqa: E402


CONFIG = ROOT / "amlt/feeds_finalfit_cache_v2_merge_20260731.yaml"


class FinalFitCacheV2MergeConfigTest(unittest.TestCase):
    def test_config_passes(self) -> None:
        validate(load(CONFIG))

    def test_mount_or_command_drift_fails(self) -> None:
        config = load(CONFIG)
        mount = copy.deepcopy(config)
        mount["storage"]["student_store"]["mount_dir"] = "/tmp"
        with self.assertRaisesRegex(ValueError, "Storage drift"):
            validate(mount)
        command = copy.deepcopy(config)
        command["jobs"][0]["command"] = command["jobs"][0]["command"].replace(
            "--expected-rows 259845", "--expected-rows 1"
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(command)


if __name__ == "__main__":
    unittest.main()
