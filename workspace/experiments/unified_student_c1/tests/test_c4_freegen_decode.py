"""Tiny random transformer test of patch scope and cached generation masks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from evaluate_c4_freegen_novel import greedy_tokens


class DecodeTest(unittest.TestCase):
    def test_cached_matches_full_recomputation_with_patches(self):
        torch.manual_seed(31)
        config = Qwen3Config(vocab_size=47, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=128)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        vectors = {i: torch.randn(32) for i in range(3)}
        for patch in (None, vectors):
            cached = greedy_tokens(model, [2, 3, 4, 5, 6, 7, 8], 1, 4, patch, 7, set(), cached=True)
            full = greedy_tokens(model, [2, 3, 4, 5, 6, 7, 8], 1, 4, patch, 7, set(), cached=False)
            self.assertEqual(cached, full)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))


if __name__ == "__main__":
    unittest.main()
