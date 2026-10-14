"""Architecture diagnostic: final block OUTPUT at a past token has no readout path.

This is a tiny random-model unit test, not a behavioral result for trained C4.
"""
from pathlib import Path
import sys
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_c4_freegen_novel import patch_hooks
from state_swap_misconception_smoke import build_bottleneck_attention_mask


def test_final_block_output_patch_does_not_reach_later_positions():
    torch.manual_seed(46)
    config=Qwen3Config(vocab_size=47,hidden_size=32,intermediate_size=64,
                      num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
                      head_dim=8,max_position_embeddings=128)
    config._attn_implementation="eager"
    model=Qwen3ForCausalLM(config).eval()
    ids=torch.tensor([[2,3,4,5,6,7,8,9,10]])
    state_pos=4
    mask=build_bottleneck_attention_mask(torch.ones_like(ids),torch.tensor([1]),
                                        torch.tensor([state_pos]),torch.tensor([7]),"late_fusion")
    with torch.inference_mode():
        base=model(input_ids=ids,attention_mask=mask,use_cache=True)
        final=patch_hooks(model,state_pos,{2:torch.randn(32)*10})
        try:
            changed=model(input_ids=ids,attention_mask=mask,use_cache=True)
        finally:
            for h in final: h.remove()
        assert not torch.equal(base.logits[:,state_pos],changed.logits[:,state_pos])
        assert torch.equal(base.logits[:,state_pos+1:],changed.logits[:,state_pos+1:])
        for a,b in zip(base.past_key_values.layers,changed.past_key_values.layers):
            assert torch.equal(a.keys,b.keys)
            assert torch.equal(a.values,b.values)
        early=patch_hooks(model,state_pos,{1:torch.randn(32)*10})
        try:
            earlier=model(input_ids=ids,attention_mask=mask,use_cache=False)
        finally:
            for h in early: h.remove()
        assert not torch.equal(base.logits[:,-1],earlier.logits[:,-1])
