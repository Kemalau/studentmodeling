from pathlib import Path

from scripts.validate_state8_interleaved_retention_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_interleaved_retention_config_is_hash_bound_and_remote() -> None:
    validate(ROOT / "amlt/feeds_state8_interleaved_retention_20260813.yaml")


def test_interleaved_runner_restores_render_only_environment_before_execution() -> None:
    text = (ROOT / "scripts/run_state8_interleaved_retention_branch.sh").read_text(
        encoding="utf-8"
    )
    execution = 'bash "$rendered"'
    assert text.count(execution) == 1
    required_bindings = (
        'export TRAIN_SEED="$REPAIR_SEED"',
        "export REQUESTED_WORLD_SIZE=1",
        "export GRADIENT_ACCUMULATION_STEPS=4",
        "export STATE8_REPAIR_LABEL_WEIGHT=1.0",
        "export STATE8_CAUSAL_LABEL_WEIGHT=1.0",
        "export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=1.0",
        "export STATE8_SOURCE_ANCHOR_ENTRYPOINT_SHA256=",
        "export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=",
        "export STATE8_DIAGNOSIS_FIRST_C4_SHA256=",
        "export REPAIR_ARM=taskanchor",
        'export STUDENT_ROOT="$SOURCE_CHECKPOINT_ROOT"',
        'export STUDENT_MODEL_DIR="$STUDENT_ROOT/adapter"',
        'export STUDENT_STATE_PATH="$STUDENT_ROOT/recurrent_state.pt"',
        'export STUDENT_RECEIPT="$STUDENT_ROOT/checkpoint_completion.json"',
    )
    for binding in required_bindings:
        assert text.count(binding) == 1
        assert text.index(binding) < text.index(execution)
