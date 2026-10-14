from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_auth_probe_has_bounded_latency() -> None:
    source = (ROOT / "scripts/use_studentmodeling_feeds_auth.sh").read_text(encoding="utf-8")
    assert source.count("timeout 30 az account get-access-token") == 2
    assert "studentmodeling-feeds-auth-choice.lock" in source
    assert 'mktemp "${studentmodeling_auth_state}.partial.XXXXXX"' in source
    assert "studentmodeling_now - studentmodeling_saved_at >= 60" in source
    assert "STUDENTMODELING_FEEDS_AUTH_HEALTH" in source
