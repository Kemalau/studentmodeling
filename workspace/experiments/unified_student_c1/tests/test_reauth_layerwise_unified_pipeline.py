from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReauthPipelineTest(unittest.TestCase):
    def test_reauth_is_isolated_and_does_not_mutate_remote_jobs(self) -> None:
        text = (ROOT / "scripts/reauth_layerwise_unified_pipeline.sh").read_text()
        self.assertIn("use_studentmodeling_feeds_auth.sh", text)
        helper = (ROOT / "scripts/use_studentmodeling_feeds_auth.sh").read_text()
        self.assertIn(".azure-studentmodeling-feeds", helper)
        self.assertIn("env -u AZURE_CONFIG_DIR", helper)
        self.assertIn("--use-device-code", text)
        self.assertIn("timeout --signal=TERM --kill-after=10", text)
        self.assertIn("REAUTH_DEVICE_TIMEOUT_SECONDS", text)
        self.assertIn("az account get-access-token", text)
        self.assertIn("env -u AZURE_CONFIG_DIR az account get-access-token", helper)
        self.assertIn("az storage blob exists", text)
        self.assertIn("select_a100_80gb", text)
        self.assertIn("auth_ready", text)
        self.assertNotIn("amlt cancel", text)
        self.assertNotIn("amlt run", text)
        self.assertNotIn("az storage blob delete", text)


if __name__ == "__main__":
    unittest.main()
