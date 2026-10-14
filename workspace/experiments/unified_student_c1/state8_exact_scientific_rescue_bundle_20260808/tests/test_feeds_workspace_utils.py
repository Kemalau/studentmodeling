from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from feeds_workspace_utils import workspace_row_exists  # noqa: E402


class FeedsWorkspaceUtilsTest(unittest.TestCase):
    def test_accepts_amlt_separate_workspace_and_resource_group_columns(self) -> None:
        table = """
NAME                           SERVICE  SUBSCRIPTION  RESOURCE GROUP                  SOURCE
CS-NewsAndFeeds-Singularity    sing     subscription  rg-cs-newsandfeeds-singularity  manual
"""
        self.assertTrue(
            workspace_row_exists(
                table,
                "CS-NewsAndFeeds-Singularity",
                "rg-cs-newsandfeeds-singularity",
            )
        )

    def test_rejects_nearby_workspace_or_resource_group(self) -> None:
        table = """
CS-NewsAndFeeds-Singularity-x  sing  subscription  rg-cs-newsandfeeds-singularity  manual
CS-NewsAndFeeds-Singularity    sing  subscription  rg-cs-newsandfeeds-other        manual
"""
        self.assertFalse(
            workspace_row_exists(
                table,
                "CS-NewsAndFeeds-Singularity",
                "rg-cs-newsandfeeds-singularity",
            )
        )


if __name__ == "__main__":
    unittest.main()
