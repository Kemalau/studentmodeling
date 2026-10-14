from pathlib import Path

from scripts.verify_xes_profile_confirmatory_freeze import verify


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_xes_confirmatory_manifest_passes() -> None:
    result = verify(ROOT / "manifests/xes_state8_profile_confirmatory_v1")
    assert result["status"] == "pass"
    assert result["rows"] == {"train": 800, "valid": 200, "test": 466}
    assert result["test_metrics_inspected_by_this_verifier"] is False
