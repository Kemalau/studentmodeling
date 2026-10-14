#!/usr/bin/env python3
"""Merge disjoint teacher-cache partitions without copying their shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(parts_root: Path, output_dir: Path, expected_rows: int) -> dict:
    if parts_root.resolve() != output_dir.resolve():
        raise ValueError("Partition root and merged cache root must be identical")
    part_dirs = sorted(path for path in parts_root.glob("rank-*") if path.is_dir())
    if not part_dirs:
        raise ValueError("No teacher-cache partitions found")
    manifests = []
    partition_indices = set()
    merged_index = []
    seen = set()
    merged_shard_hashes = {}
    for part in part_dirs:
        manifest = json.loads((part / "manifest.json").read_text(encoding="utf-8"))
        integrity = json.loads((part / "integrity.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or integrity.get("status") != "verified":
            raise ValueError(f"Incomplete cache partition: {part}")
        try:
            directory_index = int(part.name.removeprefix("rank-"))
            manifest_index = int(manifest["partition_index"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid cache partition identity: {part}") from error
        if manifest_index != directory_index or manifest_index in partition_indices:
            raise ValueError(f"Cache partition identity mismatch: {part}")
        partition_indices.add(manifest_index)
        if sha256(part / "manifest.json") != integrity.get("manifest_sha256"):
            raise ValueError(f"Partition manifest hash mismatch: {part}")
        if sha256(part / "index.json") != integrity.get("index_sha256"):
            raise ValueError(f"Partition index hash mismatch: {part}")
        index = json.loads((part / "index.json").read_text(encoding="utf-8"))
        if len(index) != int(manifest["rows"]):
            raise ValueError(f"Partition index mismatch: {part}")
        part_shards = integrity.get("shards") or {}
        for filename, expected_hash in part_shards.items():
            relative = str(part.relative_to(output_dir) / filename)
            if sha256(output_dir / relative) != expected_hash:
                raise ValueError(f"Partition shard hash mismatch: {relative}")
            merged_shard_hashes[relative] = expected_hash
        for row in index:
            example_id = str(row["example_id"])
            if example_id in seen:
                raise ValueError(f"Duplicate cache example: {example_id}")
            seen.add(example_id)
            item = dict(row)
            item["file"] = str(part.relative_to(output_dir) / str(row["file"]))
            if str(row["file"]) not in part_shards:
                raise ValueError(f"Partition index references unknown shard: {item['file']}")
            merged_index.append(item)
        manifests.append(manifest)
    if len(merged_index) != expected_rows:
        raise ValueError(
            f"Merged cache rows={len(merged_index)} expected={expected_rows}"
        )
    source_hashes = {str(value["source_sha256"]) for value in manifests}
    models = {str(value["model"]) for value in manifests}
    adapters = {str(value["teacher_adapter"]) for value in manifests}
    adapter_hashes = {
        value.get("teacher_adapter_sha256") for value in manifests
    }
    topks = {int(value["top_k"]) for value in manifests}
    state_token_aliases = {str(value["state_token_alias"]) for value in manifests}
    max_lengths = {int(value["max_length"]) for value in manifests}
    dtypes = {str(value["dtype"]) for value in manifests}
    shard_sizes = {int(value["shard_size"]) for value in manifests}
    scopes = {str(value["scope"]) for value in manifests}
    batch_sizes = {int(value.get("batch_size", 1)) for value in manifests}
    attention_implementations = {
        str(value.get("attn_implementation", "eager")) for value in manifests
    }
    cache_roles = {value.get("cache_role") for value in manifests}
    prompt_fields = {value.get("prompt_field") for value in manifests}
    state_token_counts = {int(value.get("state_token_count", 1)) for value in manifests}
    state_token_rendered_aliases = {
        value.get("state_token_rendered_alias") for value in manifests
    }
    state8_natural_views = {
        bool(value.get("state8_natural_view", False)) for value in manifests
    }
    state8_natural_view_hashes = {
        value.get("state8_natural_view_sha256") for value in manifests
    }
    layerwise_online_kd_values = {
        bool(value.get("layerwise_online_kd", False)) for value in manifests
    }
    source_row_counts = {int(value.get("source_rows", expected_rows)) for value in manifests}
    partition_counts = {int(value.get("partition_count", 1)) for value in manifests}
    order_seeds = {value.get("order_seed") for value in manifests}
    invariant_sets = (
        source_hashes,
        models,
        adapters,
        adapter_hashes,
        topks,
        state_token_aliases,
        max_lengths,
        dtypes,
        shard_sizes,
        scopes,
        batch_sizes,
        attention_implementations,
        cache_roles,
        prompt_fields,
        state_token_counts,
        state_token_rendered_aliases,
        state8_natural_views,
        state8_natural_view_hashes,
        layerwise_online_kd_values,
        source_row_counts,
        partition_counts,
        order_seeds,
    )
    if any(len(values) != 1 for values in invariant_sets):
        raise ValueError("Teacher-cache partition metadata drift")
    if partition_counts != {len(part_dirs)}:
        raise ValueError("Teacher-cache partition count mismatch")
    if partition_indices != set(range(len(part_dirs))):
        raise ValueError("Teacher-cache partition index coverage mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(merged_index, indent=2) + "\n", encoding="utf-8")
    first = manifests[0]
    payload = {
        "version": 2,
        "status": "complete",
        "model": next(iter(models)),
        "teacher_adapter": next(iter(adapters)),
        "teacher_adapter_sha256": next(iter(adapter_hashes)),
        "source_jsonl": first["source_jsonl"],
        "source_sha256": next(iter(source_hashes)),
        "rows": len(merged_index),
        "source_rows": next(iter(source_row_counts)),
        "state_token_alias": next(iter(state_token_aliases)),
        "state_token_count": next(iter(state_token_counts)),
        "state_token_rendered_alias": next(iter(state_token_rendered_aliases)),
        "prompt_field": next(iter(prompt_fields)),
        "state8_natural_view": next(iter(state8_natural_views)),
        "state8_natural_view_sha256": next(iter(state8_natural_view_hashes)),
        "cache_role": next(iter(cache_roles)),
        "layerwise_online_kd": next(iter(layerwise_online_kd_values)),
        "max_length": next(iter(max_lengths)),
        "dtype": next(iter(dtypes)),
        "top_k": next(iter(topks)),
        "shard_size": next(iter(shard_sizes)),
        "batch_size": next(iter(batch_sizes)),
        "attn_implementation": next(iter(attention_implementations)),
        "shards": sum(int(value["shards"]) for value in manifests),
        "scope": next(iter(scopes)),
        "index_sha256": sha256(index_path),
        "order_seed": next(iter(order_seeds)),
        "partition_count": len(part_dirs),
        "partition_manifests": [
            {
                "path": str(part / "manifest.json"),
                "sha256": sha256(part / "manifest.json"),
                "rows": int(manifest["rows"]),
            }
            for part, manifest in zip(part_dirs, manifests, strict=True)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    integrity = {
        "version": 2,
        "status": "verified",
        "manifest_sha256": sha256(manifest_path),
        "index_sha256": sha256(index_path),
        "rows": len(merged_index),
        "shards": dict(sorted(merged_shard_hashes.items())),
        "verification": "composition_of_verified_disjoint_partitions",
    }
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.parts_root, args.output_dir, args.expected_rows), indent=2))


if __name__ == "__main__":
    main()
