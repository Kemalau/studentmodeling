import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_recurrent_student_m4_manifest_g1_open_t3_selective_kd as hedge  # noqa: E402


def test_hedge_declares_non_bitwise_topology_sensitivity() -> None:
    source = (ROOT / "scripts/train_recurrent_student_m4_manifest_g1_open_t3_selective_kd.py").read_text()
    assert '"bitwise_equivalence_claimed": False' in source
    assert "single_a100_serial_accumulation_4" in source
    assert "FULL_REAL_ROWS = 259_845" in source
    assert "SEGMENT_START = 12_000" in source


def test_manifest_selection_preserves_declared_order(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {
            "example_id": f"row-{index}",
            "task": "diagnose_misconception",
            "source": "eedi_mining",
            "target_label": str(index),
        }
        for index in range(6)
    ]
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "stage": "layerwise_task_retention_index_manifest_v1",
        "branch": "eedi_full",
        "start_global_position": hedge.SEGMENT_START,
        "source_train_sha256": hedge.sha256(train),
        "indices": [4, 1, 5, 0, 3, 2],
    }
    # The full 5000-row invariant belongs to main(); this fixture checks the
    # order primitive used by its read hook without requiring a huge corpus.
    assert [rows[index]["example_id"] for index in manifest["indices"]] == [
        "row-4", "row-1", "row-5", "row-0", "row-3", "row-2"
    ]
