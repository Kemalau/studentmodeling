from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_guard_preserves_only_non_accumulation_boundary_receipts() -> None:
    source = (
        ROOT / "scripts/watch_state8_single_ga4_checkpoint_boundaries.sh"
    ).read_text()
    assert "step % 4 == 0" in source
    assert "checkpoint_completion.nonresumable_ga4.json" in source
    assert 'az storage blob delete' in source
    assert 'name "$visible"' in source
    assert 'adapter_model.safetensors' not in source
    assert 'recurrent_state.pt' not in source
    assert 'cmp "$source" "$verify"' in source
