from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_recurrent_formal_protocols import (  # noqa: E402
    activate_decode_adapter,
    closed_decode_text,
    evaluation_metadata,
    evaluation_shard,
    generate_batch,
    generate_one,
    pad_supervised_batches,
    prepare_recurrent_modules_for_inference,
    recurrent_states_batched,
    sequence_logprob_at,
    update_texts_for_row,
)
from evaluate_formal_protocols import sample_rows
import torch
from torch import nn


class RecurrentFormalProtocolTest(unittest.TestCase):
    def test_evaluation_shards_are_stable_disjoint_and_complete(self) -> None:
        rows = [{"example_id": f"row-{index}"} for index in range(17)]
        reordered = list(reversed(rows))
        shards = [evaluation_shard(rows, index, 4) for index in range(4)]
        reordered_shards = [
            evaluation_shard(reordered, index, 4) for index in range(4)
        ]
        ids = [[row["example_id"] for row in shard] for shard in shards]
        self.assertEqual(
            ids,
            [[row["example_id"] for row in shard] for shard in reordered_shards],
        )
        flattened = [example_id for shard in ids for example_id in shard]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["example_id"] for row in rows})
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_evaluation_shard_rejects_partial_or_invalid_coordinates(self) -> None:
        rows = [{"example_id": "only"}]
        with self.assertRaisesRegex(ValueError, "provided together"):
            evaluation_shard(rows, 0, None)
        with self.assertRaisesRegex(ValueError, "0 <= index"):
            evaluation_shard(rows, 2, 2)

    def test_layerwise_loader_restores_transition_artifact(self) -> None:
        source = (SCRIPT_DIR / "evaluate_recurrent_formal_protocols.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('config.get("layerwise_transition_rank", 0)', source)
        self.assertIn('payload["transition_in"]', source)
        self.assertIn('payload["transition_out"]', source)
        self.assertIn("Layerwise state artifact lacks transition weights", source)

    def test_sampling_preserves_original_protocol_groups(self) -> None:
        rows = [
            {
                "example_id": f"{protocol}-{index}",
                "task": "predict_correctness",
                "target_label": "correct" if index % 2 else "incorrect",
                "original_metadata": {
                    "eval_source": "malrule",
                    "eval_protocol": protocol,
                },
            }
            for protocol in ("same", "cross")
            for index in range(6)
        ]
        selected = sample_rows(rows, maximum=4, seed=42)
        counts = {}
        for row in selected:
            protocol = row["original_metadata"]["eval_protocol"]
            counts[protocol] = counts.get(protocol, 0) + 1
        self.assertEqual(counts, {"same": 4, "cross": 4})

    def test_recurrent_projection_stays_fp32_with_bf16_backbone(self) -> None:
        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(3, 4).to(dtype=torch.bfloat16)

            def get_input_embeddings(self):
                return self.embedding

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.state_norm = nn.LayerNorm(4, elementwise_affine=False)
                self.state_in_proj = nn.Linear(4, 4, bias=False)
                self.state_out_proj = nn.Linear(4, 4, bias=False)

        backbone = Backbone()
        wrapper = Wrapper()
        self.assertEqual(wrapper.state_in_proj.weight.dtype, torch.float32)
        prepare_recurrent_modules_for_inference(wrapper, backbone)
        self.assertEqual(wrapper.state_in_proj.weight.dtype, torch.float32)
        self.assertEqual(wrapper.state_out_proj.weight.dtype, torch.float32)

    def test_layerwise_wrapper_needs_no_scalar_projection_modules(self) -> None:
        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(3, 4).to(dtype=torch.bfloat16)

            def get_input_embeddings(self):
                return self.embedding

        wrapper = nn.Module()
        prepare_recurrent_modules_for_inference(wrapper, Backbone())

    def test_same_decode_adapter_is_a_noop(self) -> None:
        wrapper = type("Wrapper", (), {"backbone": object()})()
        self.assertEqual(
            activate_decode_adapter(wrapper, "same", "same"), "default"
        )

    def test_layerwise_cold_start_remains_natural_none_state(self) -> None:
        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def initial_state(self, batch_size, device=None):
                return None

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

        states = recurrent_states_batched(
            Wrapper(),
            Tokenizer(),
            [{"example_id": "cold", "update_texts": []}],
            "<state>",
            32,
            4,
        )
        self.assertEqual(states, [None])

    def test_manifest_update_texts_take_precedence_over_raw_history(self) -> None:
        row = {
            "update_texts": ["<STUDENT_STATE> audited <STUDENT_STATE>"],
            "history": [{"question_text": "raw"}],
        }
        self.assertEqual(
            update_texts_for_row(row), ["<STUDENT_STATE> audited <STUDENT_STATE>"]
        )

    def test_manifest_decode_and_original_metadata_are_used(self) -> None:
        row = {
            "decode_text": "instruction <STUDENT_STATE> query",
            "source": "fallback",
            "original_metadata": {
                "eval_source": "dbe_kt22",
                "eval_protocol": "temporal_warm_item_binary",
                "prospective_delta": 3,
            },
        }
        self.assertEqual(closed_decode_text(row), row["decode_text"])
        self.assertEqual(
            evaluation_metadata(row),
            {
                "source": "dbe_kt22",
                "protocol": "temporal_warm_item_binary",
                "delta": 3,
            },
        )

    def test_closed_decoder_rejects_history_leak(self) -> None:
        with self.assertRaisesRegex(AssertionError, "leaked raw history"):
            closed_decode_text(
                {
                    "decode_text": (
                        "<student_history>secret</student_history>"
                        "<STUDENT_STATE>query"
                    )
                }
            )

    def test_candidate_padding_is_excluded_from_logprob(self) -> None:
        batches = [
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "labels": torch.tensor([[-100, 2, 3]]),
                "prompt_ends": torch.tensor([1]),
            },
            {
                "input_ids": torch.tensor([[1, 4]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "labels": torch.tensor([[-100, 4]]),
                "prompt_ends": torch.tensor([1]),
            },
        ]
        combined = pad_supervised_batches(batches, 0)
        logits = torch.zeros(2, 3, 5)
        score = sequence_logprob_at(
            logits,
            combined["input_ids"],
            prompt_end=1,
            valid_length=2,
            row_index=1,
        )
        self.assertAlmostEqual(score, -torch.log(torch.tensor(5.0)).item())

    def test_equal_length_greedy_batch_matches_one_by_one_decode(self) -> None:
        class Tokenizer:
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                # Both prompts have equal length but select different first
                # generated tokens through their final character.
                return {"input_ids": [7, 1 if text.endswith("a") else 2]}

            def decode(self, values, skip_special_tokens=True):
                del skip_special_tokens
                return ",".join(map(str, values))

        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, **kwargs):
                del kwargs
                batch = input_ids.shape[0]
                logits = torch.full((batch, 1, 8), -100.0)
                for index, value in enumerate(input_ids[:, -1].tolist()):
                    logits[index, 0, 0 if value >= 4 else value + 2] = 0.0
                return SimpleNamespace(logits=logits, past_key_values=())

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = Backbone()

            def decode_from_state(self, state, input_ids, **kwargs):
                del state, kwargs
                batch, length = input_ids.shape
                logits = torch.full((batch, length, 8), -100.0)
                for index, value in enumerate(input_ids[:, -1].tolist()):
                    logits[index, -1, value + 2] = 0.0
                return SimpleNamespace(logits=logits, past_key_values=())

        wrapper = Wrapper()
        tokenizer = Tokenizer()
        texts = ["a", "b"]
        states = [torch.tensor([1.0]), torch.tensor([2.0])]
        expected = [
            generate_one(wrapper, tokenizer, state.unsqueeze(0), text, "", 16, 6)
            for state, text in zip(states, texts, strict=True)
        ]
        self.assertEqual(
            generate_batch(wrapper, tokenizer, states, texts, "", 16, 6),
            expected,
        )

    def test_generation_batch_rejects_padding_or_state_route_changes(self) -> None:
        wrapper = nn.Linear(1, 1)
        tokenizer = type("Tokenizer", (), {"eos_token_id": 0})()
        tokenizer.__class__.__call__ = lambda self, text, add_special_tokens=False: {
            "input_ids": [1] * len(text)
        }
        with self.assertRaisesRegex(ValueError, "equal prompt lengths"):
            generate_batch(wrapper, tokenizer, [None, None], ["a", "bb"], "", 8, 2)
        with self.assertRaisesRegex(ValueError, "mixed cold and populated"):
            generate_batch(
                wrapper,
                tokenizer,
                [None, torch.zeros(1)],
                ["a", "b"],
                "",
                8,
                2,
            )


if __name__ == "__main__":
    unittest.main()
