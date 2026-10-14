from pathlib import Path

import pytest
import torch

from scripts.train_c1_lora import non_tensor_parallel_peft_resume


def test_init_adapter_is_fail_closed_and_separate_from_resume() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/train_c1_lora.py"
    ).read_text(encoding="utf-8")
    assert '"--init-adapter"' in source
    assert "is_trainable=True" in source
    assert "model.enable_input_require_grads()" in source
    assert "initialize the exact" in source
    assert "with non_tensor_parallel_peft_resume(model):" in source
    assert "trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)" in source
    assert '"init_adapter": str(args.init_adapter)' in source
    assert '"--num-train-epochs"' in source
    assert "--num-train-epochs requires a positive value and --max-steps -1" in source
    assert '"seed": args.seed' in source


def test_seed_is_set_before_model_and_lora_initialization() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/train_c1_lora.py"
    ).read_text(encoding="utf-8")
    seed_pos = source.index("set_seed(args.seed)")
    model_pos = source.index("AutoModelForCausalLM.from_pretrained(")
    lora_pos = source.index("get_peft_model(model, lora_cfg)")
    assert seed_pos < model_pos < lora_pos

    paired_source = (
        Path(__file__).resolve().parents[1] / "scripts/train_c1_paired_state_swap.py"
    ).read_text(encoding="utf-8")
    paired_seed_pos = paired_source.index("set_seed(args.seed)")
    paired_model_pos = paired_source.index("AutoModelForCausalLM.from_pretrained(")
    paired_lora_pos = paired_source.index("model = get_peft_model(")
    assert paired_seed_pos < paired_model_pos < paired_lora_pos


def test_non_tensor_parallel_peft_resume_restores_hook() -> None:
    import peft.utils.save_and_load as peft_save_and_load

    model = torch.nn.Linear(2, 2)
    original = peft_save_and_load._maybe_shard_state_dict_for_tp
    with non_tensor_parallel_peft_resume(model):
        assert peft_save_and_load._maybe_shard_state_dict_for_tp is not original
        peft_save_and_load._maybe_shard_state_dict_for_tp(model, {}, "default")
    assert peft_save_and_load._maybe_shard_state_dict_for_tp is original


def test_non_tensor_parallel_peft_resume_rejects_active_tp_mesh() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model[0]._hf_tp_plan = "colwise"
    model[0]._hf_device_mesh = object()
    with pytest.raises(RuntimeError, match="real tensor-parallel model"):
        with non_tensor_parallel_peft_resume(model):
            pass
