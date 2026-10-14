"""Load and validate immutable behavioral-distillation teacher caches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherCache:
    def __init__(
        self,
        root: Path,
        expected_source_sha256: str | None = None,
        *,
        verify_shards: bool = True,
        expected_manifest: dict[str, object] | None = None,
    ) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        index_path = root / "index.json"
        integrity_path = root / "integrity.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        self.declared_unused = (
            os.environ.get("STUDENTMODELING_TEACHER_CACHE_UNUSED", "0") == "1"
        )
        if self.manifest.get("status") != "complete":
            raise ValueError("Teacher cache is incomplete")
        if integrity.get("status") != "verified":
            raise ValueError("Teacher cache integrity receipt is incomplete")
        if sha256(manifest_path) != integrity.get("manifest_sha256"):
            raise ValueError("Teacher cache manifest SHA256 mismatch")
        if sha256(index_path) != self.manifest.get("index_sha256"):
            raise ValueError("Teacher cache index SHA256 mismatch")
        if expected_source_sha256 and self.manifest.get("source_sha256") != expected_source_sha256:
            raise ValueError("Teacher cache source SHA256 mismatch")
        drift = {
            key: {"expected": expected, "observed": self.manifest.get(key)}
            for key, expected in (expected_manifest or {}).items()
            if self.manifest.get(key) != expected
        }
        if drift:
            raise ValueError(
                "Teacher cache scientific manifest drift: "
                + json.dumps(drift, sort_keys=True, default=str)
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if len(index) != int(self.manifest["rows"]):
            raise ValueError("Teacher cache row count mismatch")
        if int(integrity.get("rows", -1)) != len(index):
            raise ValueError("Teacher cache integrity row count mismatch")
        if verify_shards and not self.declared_unused:
            for filename, expected_hash in (integrity.get("shards") or {}).items():
                if sha256(root / filename) != expected_hash:
                    raise ValueError(f"Teacher cache shard SHA256 mismatch: {filename}")
        self.shards_verified = bool(verify_shards and not self.declared_unused)
        self.by_example = {str(row["example_id"]): row for row in index}
        if len(self.by_example) != len(index):
            raise ValueError("Duplicate example IDs in teacher cache")
        self._loaded_shard_name: str | None = None
        self._loaded_shard: dict | None = None

    def load(self, example_id: str) -> dict:
        if self.declared_unused:
            raise RuntimeError(
                "Teacher cache was declared unused; shard payload access is forbidden"
            )
        row = self.by_example[str(example_id)]
        if self._loaded_shard_name != row["file"]:
            self._loaded_shard = torch.load(
                self.root / row["file"], map_location="cpu", weights_only=True
            )
            self._loaded_shard_name = row["file"]
        payload = self._loaded_shard[str(example_id)]
        if str(payload["example_id"]) != str(example_id):
            raise ValueError("Teacher cache example ID mismatch")
        if len(payload["target_token_ids"]) != int(row["target_tokens"]):
            raise ValueError("Teacher cache target length mismatch")
        ids = payload["teacher_topk_ids"]
        logprobs = payload["teacher_topk_logprobs"]
        if ids.shape != logprobs.shape or ids.shape[0] != int(row["target_tokens"]):
            raise ValueError("Teacher cache top-k tensor shape mismatch")
        return payload
