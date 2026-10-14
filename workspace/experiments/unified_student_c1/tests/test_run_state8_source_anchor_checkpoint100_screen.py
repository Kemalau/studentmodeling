from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint100_screen_is_protocol_matched_and_hash_bound() -> None:
    text = (ROOT / "scripts/run_state8_source_anchor_checkpoint100_screen.sh").read_text()
    assert "757e58405b9f96cd18d73ef840eb779d2848f7dbc51eb664e661e01ef1fea4ab" in text
    assert 'replace("checkpoint-20", "checkpoint-100", 1)' in text
    assert '--arm \\\"$SOURCE_ANCHOR_ARM\\\" --step 100' in text
    assert "checkpoint100_screen_v1" in text
    assert "'checkpoint_step':100" in text
    assert 'bash "$rendered"' in text
