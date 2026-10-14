from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_highcausal_promotion_emits_verified_final_fit_gate() -> None:
    text = (ROOT / "scripts/watch_state8_highcausal_promotion.sh").read_text()
    assert "state8_qualified_unified_final_fit_v1" in text
    assert "state8_canonical_final_checkpoint_publish_v1" in text
    assert 'touch "$state/scientific_gate_pass" "$state/complete"' in text
    assert text.index("state8_qualified_unified_final_fit_v1") < text.index(
        'touch "$state/scientific_gate_pass"'
    )


def test_highcausal_promotion_prefers_a_live_azure_profile() -> None:
    text = (ROOT / "scripts/watch_state8_highcausal_promotion.sh").read_text()
    assert ".azure az account get-access-token" in text
    assert ".azure-keyflip-feeds" in text
