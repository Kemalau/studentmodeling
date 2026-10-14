from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/materialize_recurrent_index_manifest.py").read_text()


def test_materializer_is_hash_order_and_leakage_bound() -> None:
    assert "source_train_sha256" in SOURCE
    assert "formal_eval_sha256" in SOURCE
    assert "selected_example_ids_sha256" in SOURCE
    assert "len(indices) != len(set(indices))" in SOURCE
    assert "Materialized train/test overlap" in SOURCE
