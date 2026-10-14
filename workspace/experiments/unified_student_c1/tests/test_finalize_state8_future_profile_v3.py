from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_state8_future_profile_v3 import job_rows, require, set_f1


def test_set_f1() -> None:
    assert set_f1(["a", "b"], ["a", "c"]) == 0.5
    assert set_f1([], []) == 1.0
    assert set_f1(["a"], []) == 0.0


def test_require_fails_closed() -> None:
    with pytest.raises(ValueError, match="broken"):
        require(False, "broken")


def test_job_rows_keeps_lineage_identity(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            [
                {
                    "experiment": "exp",
                    "job_id": "job",
                    "job_name": ":name",
                    "status": "pass",
                    "duration": "22h",
                    "cluster": "Feeds",
                    "sku": "80G1-A100",
                    "ignored": "value",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert job_rows([status]) == [
        {
            "experiment": "exp",
            "job_id": "job",
            "job_name": ":name",
            "status": "pass",
            "duration": "22h",
            "cluster": "Feeds",
            "sku": "80G1-A100",
        }
    ]
