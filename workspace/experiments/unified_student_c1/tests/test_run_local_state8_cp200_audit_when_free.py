from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_cp200_audit_uses_downloaded_adapter_and_clean_cli() -> None:
    source = (ROOT / "scripts/run_local_state8_cp200_audit_when_free.sh").read_text()
    assert '--adapter "$state/adapter"' in source
    assert '--checkpoint "$state/recurrent_state.pt"' in source
    assert "--state-token-alias '<|vision_pad|>'" in source
    assert "checkpoint_step']==200" in source
    assert ".py +" not in source
    assert ' +  --' not in source
