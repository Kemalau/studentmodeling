from pathlib import Path


def test_init_adapter_is_fail_closed_and_separate_from_resume() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/train_c1_lora.py"
    ).read_text(encoding="utf-8")
    assert '"--init-adapter"' in source
    assert "is_trainable=True" in source
    assert "model.enable_input_require_grads()" in source
    assert "initialize the exact" in source
    assert "trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)" in source
    assert '"init_adapter": str(args.init_adapter)' in source
    assert '"--num-train-epochs"' in source
    assert "--num-train-epochs requires a positive value and --max-steps -1" in source
