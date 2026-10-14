from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_local_state8_lrfix_v4_smoke_recovery_audit.sh"


def test_local_fallback_is_bound_to_immutable_lrfix_v4_smoke() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_seed8801/checkpoint-200" in text
    assert "f23b0dc3a4c840ff3e0ef9192dd1d33700950328d7f3beb992cff4d3687ffd43" in text
    assert "24c4832dc61cbf53deb2d9ad750ec1edc18226522fdfe35f5d584ca9ff88cd78" in text
    assert "manifests/dbe_student_profile_v1/test.jsonl" in text
    assert "--permutation-replicates 2000" in text
    assert "profile_ownership_feature_normalization']=='layernorm'" in text


def test_local_fallback_does_not_retrain_or_touch_xes() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "train_state8" not in text
    assert "amlt run" not in text
    assert "xes" not in text.casefold()
    assert "device-code" not in text
