from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from teacher_cache import TeacherCache  # noqa: E402
from build_teacher_cache import (  # noqa: E402
    batched_target_logprobs,
    bind_build_spec,
    ordered_partition_indices,
    prepare_cache_row,
    prompt_for_cache,
    resumable_shard_prefix,
)
from merge_teacher_cache_partitions import merge  # noqa: E402


class TeacherCacheTest(unittest.TestCase):
    def test_state8_cache_view_rewrites_natural_target_and_direct_prompt(self) -> None:
        row = {
            "example_id": "x",
            "task": "diagnose_misconception",
            "source": "malrule",
            "target": "<misconception>fractions.denominator_comparison_error</misconception>",
            "target_label": "fractions.denominator_comparison_error",
            "direct_text": "Return exactly one <misconception> tag. <STUDENT_STATE>",
            "decode_text": "Return exactly one <misconception> tag. <STUDENT_STATE>",
            "update_texts": [],
        }
        viewed = prepare_cache_row(row, state8_natural_view=True)
        self.assertIn("<label>fractions.denominator_comparison_error</label>", viewed["target"])
        self.assertIn("<diagnosis>", viewed["target"])
        self.assertIn("<evidence>", viewed["target"])
        self.assertIn("<label>", prompt_for_cache(viewed, "direct_text"))
        self.assertEqual(row["target"], "<misconception>fractions.denominator_comparison_error</misconception>")

    def test_resumable_cache_is_bound_to_complete_build_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bind_build_spec(root, {"teacher_adapter_sha256": "a" * 64, "top_k": 256})
            bind_build_spec(root, {"teacher_adapter_sha256": "a" * 64, "top_k": 256})
            with self.assertRaisesRegex(ValueError, "build-spec drift"):
                bind_build_spec(
                    root, {"teacher_adapter_sha256": "b" * 64, "top_k": 256}
                )

    def test_batched_target_logprobs_preserve_per_example_alignment(self) -> None:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask, **kwargs):
                del attention_mask, kwargs
                vocab = 11
                logits = torch.zeros((*input_ids.shape, vocab), device=input_ids.device)
                for token in range(vocab):
                    logits[..., token] = -(input_ids.float() - token).abs()
                return type("Output", (), {"logits": logits})()

        rows = [
            {
                "input_ids": torch.tensor([[2, 3, 4, 5]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "labels": torch.tensor([[-100, -100, 4, 5]]),
            },
            {
                "input_ids": torch.tensor([[7, 8, 9]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "labels": torch.tensor([[-100, 8, 9]]),
            },
        ]
        outputs = batched_target_logprobs(
            FakeModel(), rows, 3, pad_token_id=0
        )
        self.assertEqual([row[0] for row in outputs], [[4, 5], [8, 9]])
        self.assertEqual([tuple(row[1].shape) for row in outputs], [(2, 3), (2, 3)])
        self.assertTrue(all(torch.isfinite(row[2].float()).all() for row in outputs))
        for encoded, batched in zip(rows, outputs, strict=True):
            single = batched_target_logprobs(
                FakeModel(), [encoded], 3, pad_token_id=0
            )[0]
            self.assertEqual(batched[0], single[0])
            self.assertTrue(torch.equal(batched[1], single[1]))
            self.assertTrue(torch.equal(batched[2], single[2]))

    def test_online_kd_cache_matches_detached_topk_definition(self) -> None:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask, **kwargs):
                del attention_mask, kwargs
                logits = torch.arange(13, dtype=torch.float32).view(1, 1, 13)
                logits = logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
                return type("Output", (), {"logits": logits})()

        encoded = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, 2, 3]]),
        }
        target, ids, values = batched_target_logprobs(
            FakeModel(),
            [encoded],
            4,
            pad_token_id=0,
            logprob_dtype=torch.float32,
            renormalize_topk=True,
        )[0]
        expected_logits = torch.arange(13, dtype=torch.float32).topk(4).values
        self.assertEqual(target, [2, 3])
        self.assertTrue(torch.equal(ids[0], torch.tensor([12, 11, 10, 9], dtype=torch.int32)))
        torch.testing.assert_close(
            values,
            torch.log_softmax(expected_logits, dim=-1).expand(2, -1),
            rtol=0,
            atol=0,
        )

    def test_reconstructs_valid_atomic_shard_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [
                ("x0", [4, 5]),
                ("x1", [6]),
                ("x2", [7, 8, 9]),
            ]
            for shard_id, rows in enumerate((expected[:2], expected[2:])):
                payload = {}
                for example_id, tokens in rows:
                    payload[example_id] = {
                        "example_id": example_id,
                        "target_token_ids": tokens,
                        "teacher_topk_ids": torch.zeros((len(tokens), 2), dtype=torch.int32),
                        "teacher_topk_logprobs": torch.zeros(
                            (len(tokens), 2), dtype=torch.float16
                        ),
                    }
                torch.save(payload, root / f"shard-{shard_id:05d}.pt")
            index, next_shard = resumable_shard_prefix(
                root,
                selected_indices=[10, 11, 12],
                shard_size=2,
                top_k=2,
                expected_record=lambda position, row_index: expected[position],
            )
            self.assertEqual(next_shard, 2)
            self.assertEqual([row["example_id"] for row in index], ["x0", "x1", "x2"])
            self.assertEqual([row["source_row_index"] for row in index], [10, 11, 12])

    def test_resume_rejects_noncontiguous_or_mismatched_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "wrong": {
                    "example_id": "wrong",
                    "target_token_ids": [4],
                    "teacher_topk_ids": torch.zeros((1, 2), dtype=torch.int32),
                    "teacher_topk_logprobs": torch.zeros((1, 2), dtype=torch.float16),
                }
            }
            torch.save(payload, root / "shard-00001.pt")
            with self.assertRaisesRegex(ValueError, "contiguous prefix"):
                resumable_shard_prefix(
                    root,
                    selected_indices=[10, 11],
                    shard_size=1,
                    top_k=2,
                    expected_record=lambda position, row_index: (f"x{position}", [4]),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "wrong": {
                    "example_id": "wrong",
                    "target_token_ids": [4],
                    "teacher_topk_ids": torch.zeros((1, 2), dtype=torch.int32),
                    "teacher_topk_logprobs": torch.zeros((1, 2), dtype=torch.float16),
                }
            }
            torch.save(payload, root / "shard-00000.pt")
            with self.assertRaisesRegex(ValueError, "example mismatch"):
                resumable_shard_prefix(
                    root,
                    selected_indices=[10],
                    shard_size=1,
                    top_k=2,
                    expected_record=lambda position, row_index: ("x0", [4]),
                )

    def test_ordered_partitions_reconstruct_one_seeded_epoch(self) -> None:
        partitions = [
            ordered_partition_indices(
                17, order_seed=42, partition_index=rank, partition_count=4
            )
            for rank in range(4)
        ]
        self.assertEqual(sorted(index for part in partitions for index in part), list(range(17)))
        self.assertEqual(
            partitions,
            [
                ordered_partition_indices(
                    17, order_seed=42, partition_index=rank, partition_count=4
                )
                for rank in range(4)
            ],
        )

    def test_merges_disjoint_partition_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            for rank in range(2):
                part = root / f"rank-{rank}"
                part.mkdir(parents=True)
                example_id = f"x{rank}"
                shard = part / "shard-00000.pt"
                torch.save(
                    {
                        example_id: {
                            "example_id": example_id,
                            "target_token_ids": [4],
                            "teacher_topk_ids": torch.tensor([[1, 2]]),
                            "teacher_topk_logprobs": torch.tensor([[-0.1, -2.4]]),
                        }
                    },
                    shard,
                )
                index = [{"example_id": example_id, "file": shard.name, "target_tokens": 1}]
                index_path = part / "index.json"
                index_path.write_text(json.dumps(index))
                manifest = {
                    "status": "complete",
                    "model": "model",
                    "teacher_adapter": "adapter",
                    "source_jsonl": "source.jsonl",
                    "source_sha256": "source",
                    "rows": 1,
                    "source_rows": 2,
                    "state_token_alias": "<state>",
                    "max_length": 16,
                    "dtype": "float16",
                    "top_k": 2,
                    "shard_size": 1,
                    "shards": 1,
                    "scope": "target",
                    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    "order_seed": 42,
                    "partition_index": rank,
                    "partition_count": 2,
                    "batch_size": 1,
                    "attn_implementation": "eager",
                }
                manifest_path = part / "manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                (part / "integrity.json").write_text(
                    json.dumps(
                        {
                            "status": "verified",
                            "rows": 1,
                            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            "index_sha256": manifest["index_sha256"],
                            "shards": {shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()},
                        }
                    )
                )
            payload = merge(root, root, expected_rows=2)
            self.assertEqual(payload["rows"], 2)
            self.assertEqual(payload["batch_size"], 1)
            self.assertEqual(payload["attn_implementation"], "eager")
            merged = TeacherCache(root, expected_source_sha256="source")
            self.assertEqual(set(merged.by_example), {"x0", "x1"})
            torch.testing.assert_close(
                merged.load("x1")["teacher_topk_ids"], torch.tensor([[1, 2]])
            )

    def test_validates_index_and_loads_example(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "example_id": "x",
                "target_token_ids": [4, 5],
                "teacher_topk_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "teacher_topk_logprobs": torch.randn(2, 3),
            }
            torch.save({"x": payload}, root / "shard-00000.pt")
            index = [{"example_id": "x", "file": "shard-00000.pt", "target_tokens": 2}]
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "rows": 1,
                        "index_sha256": digest,
                        "source_sha256": "source",
                    }
                ),
                encoding="utf-8",
            )
            manifest_digest = hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest()
            shard_digest = hashlib.sha256(
                (root / "shard-00000.pt").read_bytes()
            ).hexdigest()
            (root / "integrity.json").write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "rows": 1,
                        "manifest_sha256": manifest_digest,
                        "index_sha256": digest,
                        "shards": {"shard-00000.pt": shard_digest},
                    }
                ),
                encoding="utf-8",
            )
            cache = TeacherCache(root, expected_source_sha256="source")
            loaded = cache.load("x")
            torch.testing.assert_close(
                loaded["teacher_topk_logprobs"], payload["teacher_topk_logprobs"]
            )
            cache = TeacherCache(
                root,
                expected_source_sha256="source",
                expected_manifest={"rows": 1},
            )
            self.assertIn("x", cache.by_example)
            with self.assertRaisesRegex(ValueError, "scientific manifest drift"):
                TeacherCache(root, expected_manifest={"rows": 2})

    def test_rejects_corrupt_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.json").write_text("[]", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {"status": "complete", "rows": 0, "index_sha256": "wrong"}
                ),
                encoding="utf-8",
            )
            (root / "integrity.json").write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "rows": 0,
                        "manifest_sha256": hashlib.sha256(
                            (root / "manifest.json").read_bytes()
                        ).hexdigest(),
                        "index_sha256": "wrong",
                        "shards": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                TeacherCache(root)

    def test_nonverifying_worker_still_checks_manifest_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard.pt"
            torch.save(
                {
                    "x": {
                        "example_id": "x",
                        "target_token_ids": [1],
                        "teacher_topk_ids": torch.tensor([[1]]),
                        "teacher_topk_logprobs": torch.tensor([[0.0]]),
                    }
                },
                shard,
            )
            index = root / "index.json"
            index.write_text(json.dumps([{"example_id": "x", "file": "shard.pt", "target_tokens": 1}]))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "rows": 1,
                        "source_sha256": "source",
                        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    }
                )
            )
            (root / "integrity.json").write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "rows": 1,
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                        "shards": {"shard.pt": "intentionally-not-checked"},
                    }
                )
            )
            cache = TeacherCache(root, expected_source_sha256="source", verify_shards=False)
            self.assertEqual(set(cache.by_example), {"x"})

    def test_declared_unused_skips_shard_hash_but_forbids_payload_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard.pt"
            shard.write_bytes(b"payload-is-never-read")
            index = root / "index.json"
            index.write_text(json.dumps([
                {"example_id": "x", "file": "shard.pt", "target_tokens": 1}
            ]))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "status": "complete",
                "rows": 1,
                "source_sha256": "source",
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }))
            (root / "integrity.json").write_text(json.dumps({
                "status": "verified",
                "rows": 1,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                "shards": {"shard.pt": "deliberately-not-the-payload-hash"},
            }))
            with mock.patch.dict(
                os.environ, {"STUDENTMODELING_TEACHER_CACHE_UNUSED": "1"}
            ):
                cache = TeacherCache(root, expected_source_sha256="source")
                self.assertTrue(cache.declared_unused)
                self.assertFalse(cache.shards_verified)
                with self.assertRaisesRegex(RuntimeError, "declared unused"):
                    cache.load("x")


if __name__ == "__main__":
    unittest.main()
