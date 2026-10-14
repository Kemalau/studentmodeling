from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_formal_specialist_provenance import audit  # noqa: E402


def test_current_formal_specialists_have_frozen_full_support_provenance() -> None:
    value = audit(ROOT / "results/formal_specialist_v1")
    assert value["status"] == "complete"
    assert value["frozen_test_denominators"] == {
        "T1": 10913,
        "T2": 3693,
        "T3": 1494,
        "T4": 4600,
        "T5": 976,
    }
    assert all(value["checks"].values())
    assert all(len(value) == 64 for value in value["artifact_sha256"].values())
