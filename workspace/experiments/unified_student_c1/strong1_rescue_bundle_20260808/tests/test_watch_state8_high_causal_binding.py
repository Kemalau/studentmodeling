from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_binding_controller_fails_closed_on_candidate_attention() -> None:
    text = (ROOT / "scripts/watch_state8_high_causal_binding.sh").read_text()
    assert '"$state/mid" "$state/eedi" "$taxonomy"' in text
    assert 'candidate controller needs attention' in text
    assert '|| fail' in text
    assert "--taxonomy \"$taxonomy\"" in text


def test_binding_controller_prefers_a_live_azure_profile() -> None:
    text = (ROOT / "scripts/watch_state8_high_causal_binding.sh").read_text()
    assert ".azure az account get-access-token" in text
    assert ".azure-keyflip-feeds" in text
