import hashlib
import json
from pathlib import Path

import pytest

from scripts import state8_curriculum_data_cache as cache


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def materialize(root: Path, monkeypatch) -> None:
    pinned = {}
    for relative in cache.required_files():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("train5000.jsonl.receipt.json"):
            variant = relative.split("/", 1)[0]
            write_json(
                path,
                {
                    "stage": "materialized_recurrent_index_manifest_v1",
                    "rows": 5000,
                    "manifest_sha256": cache.MANIFEST_HASHES[variant],
                    "output_sha256": "pending",
                    "formal_example_id_overlap": 0,
                },
            )
        elif relative.endswith("augmentation_receipt.json"):
            write_json(
                path,
                {
                    "stage": "recurrent_causal_same_rule_augmentation_v1",
                    "status": "passed",
                },
            )
        elif relative.endswith("train.jsonl.receipt.json"):
            write_json(
                path,
                {
                    "pairs": 250,
                    "twins": 125,
                    "role_counts": {
                        "recipient_native": 125,
                        "same_rule_native": 125,
                    },
                },
            )
        else:
            path.write_text(relative + "\n", encoding="utf-8")
    for variant in cache.VARIANTS:
        output = root / variant / "train5000.jsonl"
        subset = root / variant / "train5000.jsonl.receipt.json"
        value = json.loads(subset.read_text())
        value["output_sha256"] = digest(output)
        write_json(subset, value)
    for relative in cache.PINNED_HASHES:
        pinned[relative] = digest(root / relative)
    monkeypatch.setattr(cache, "PINNED_HASHES", pinned)


def test_cache_is_receipt_last_and_detects_tampering(tmp_path, monkeypatch) -> None:
    materialize(tmp_path, monkeypatch)
    receipt = cache.create_receipt(tmp_path)
    assert receipt["receipt_published_last"] is True
    paths = cache.verify(tmp_path, "balanced")
    assert paths["subset_jsonl"].endswith("balanced/train5000.jsonl")
    (tmp_path / "common/valid.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="drift"):
        cache.verify(tmp_path, "balanced")


def test_cache_requires_completion_receipt(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="completion receipt"):
        cache.verify(tmp_path, "t3_heavy")
