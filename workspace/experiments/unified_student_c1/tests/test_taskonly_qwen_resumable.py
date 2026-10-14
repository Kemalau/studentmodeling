from __future__ import annotations

import json
import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_taskonly_qwen_checkpoint import select, validate  # noqa: E402
from validate_taskonly_qwen_resumable_amlt import load, validate as validate_amlt  # noqa: E402
from stage_taskonly_qwen_checkpoint import stage  # noqa: E402


def make_checkpoint(root: Path, step: int, max_steps: int = 1000) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": max_steps}), encoding="utf-8"
    )
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 16,
                "peft_type": "LORA",
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
                ],
            }
        ),
        encoding="utf-8",
    )
    sizes = {
        "adapter_model.safetensors": 250_000_001,
        "optimizer.pt": 500_000_001,
        "scheduler.pt": 1,
        "rng_state.pth": 1,
        "training_args.bin": 1,
    }
    for name, size in sizes.items():
        with (path / name).open("wb") as handle:
            handle.truncate(size)
    return path


def test_checkpoint_validation_and_latest_selection(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 100)
    make_checkpoint(tmp_path, 500)
    value = select(tmp_path, expected_max_steps=1000)
    assert value["global_step"] == 500
    assert value["remaining_steps"] == 500


def test_checkpoint_rejects_missing_optimizer(tmp_path: Path) -> None:
    path = make_checkpoint(tmp_path, 100)
    (path / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        validate(path, expected_max_steps=1000)


def test_resumable_runner_preserves_one_epoch_contract() -> None:
    source = (ROOT / "scripts/run_taskonly_qwen_resumable.sh").read_text(encoding="utf-8")
    for value in (
        "--max-steps -1 --num-train-epochs 1",
        "--per-device-train-batch-size 1 --gradient-accumulation-steps 4",
        "--save-steps 250 --save-total-limit 2",
        '--init-adapter "$staged_checkpoint" --resume-from-checkpoint "$staged_checkpoint"',
        "stage_taskonly_qwen_checkpoint.py",
        'test ! -e "$RUN_DIR/completion.json"',
    ):
        assert value in source


def test_checkpoint_cadence_migration_changes_only_trainer_state(tmp_path: Path) -> None:
    source = make_checkpoint(tmp_path / "source", 500, max_steps=1000)
    state_path = source / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"save_steps": 50, "stateful_callbacks": {"TrainerControl": {"args": {"should_save": True}, "attributes": {}}}})
    state_path.write_text(json.dumps(state), encoding="utf-8")
    destination = tmp_path / "staged"
    value = stage(source, destination, expected_step=500, save_steps=250)
    migrated = json.loads((destination / "trainer_state.json").read_text(encoding="utf-8"))
    assert migrated["global_step"] == 500
    assert migrated["save_steps"] == 250
    assert migrated["stateful_callbacks"]["TrainerControl"]["args"]["should_save"] is False
    assert value["only_trainer_state_changed"] is True
    for name in value["source_hashes"]:
        if name != "trainer_state.json":
            assert value["source_hashes"][name] == value["staged_hashes"][name]


def test_resumable_amlt_is_hash_bound_and_compliant() -> None:
    validate_amlt(ROOT / "amlt/feeds_taskonly_qwen_resumable_recovery_20260814.yaml")


def test_resumable_amlt_accepts_only_requested_rendered_job(tmp_path: Path) -> None:
    source = load(ROOT / "amlt/feeds_taskonly_qwen_resumable_recovery_20260814.yaml")
    selected = "qwen3-32b-taskonly-t1-resumable-seed42"
    rendered = copy.deepcopy(source)
    rendered["target"] = {
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
    }
    rendered["code"] = {
        "storage_id": "default",
        "ignore": source["code"]["ignore"],
    }
    rendered["jobs"] = [job for job in source["jobs"] if job["name"] == selected]
    path = tmp_path / "rendered.yaml"
    import yaml

    path.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    validate_amlt(path, {selected})
    with pytest.raises(ValueError, match="Job matrix drift"):
        validate_amlt(path, {"qwen3-32b-taskonly-t2-resumable-seed42"})
