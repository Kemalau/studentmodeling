from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_state8_final_eval import FROZEN_GROUP_SUPPORT, REFERENCES
from scripts.validate_state8_taskanchor_confirmation import validate


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path: Path, *, t6_positive: float = 0.5):
    artifacts = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    groups = {}
    for name, n in FROZEN_GROUP_SUPPORT.items():
        task = name.split("/", 1)[0]
        row = {"n": n}
        if task in REFERENCES:
            metric, reference, _ = REFERENCES[task]
            row[metric] = reference
        if task == "predict_error_persistence":
            row["positive_recall"] = t6_positive
        groups[name] = row
    formal = {
        "sampled_rows": 21676,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "artifact_sha256": artifacts,
        "metrics_by_group": groups,
    }
    c4_rows = [
        {"pair_seed": seed, "D_specific": 0.2, "random": 0.1, "noop": 0.0, "same_rule": 0.05, "scientific_gate_pass": True}
        for seed in (13, 42, 97)
    ]
    c4 = {"stage": "state8_promotion_c4_multiseed_v1", "replication_gate_pass": True, "mean_D_specific": 0.2, "rows": c4_rows}
    semantic = {"stage": "final_T3_semantic_judge", "pass": True, "support": 1494, "judge_backend": "trapi"}
    diagnosis = {
        "stage": "state8_closed_diagnosis_c4_semantic_multiseed_v1",
        "semantic_judge_authoritative": True,
        "exact_label_role": "auxiliary_only_not_used_for_gate",
        "pass": True,
        "seed_passes": 3,
        "mean_semantic_D_specific": 0.3,
        "mean_swap_to_donor": 0.7,
        "mean_random_matches_either": 0.1,
        "rows": [{"generation": {"artifact_sha256": artifacts}} for _ in (13, 42, 97)],
    }
    return tuple(dump(tmp_path / f"{name}.json", value) for name, value in (
        ("formal", formal), ("c4", c4), ("semantic", semantic), ("diagnosis", diagnosis)
    ))


def test_full_confirmation_passes_only_complete_joint_gate(tmp_path: Path) -> None:
    value = validate(*fixture(tmp_path), 400, "a" * 64, "b" * 64)
    assert value["pass"] is True
    assert value["checkpoint_step"] == 400
    assert value["requires_exact_one_epoch_final_fit"] is True


def test_full_confirmation_rejects_t6_collapse(tmp_path: Path) -> None:
    value = validate(*fixture(tmp_path, t6_positive=0.0), 600, "a" * 64, "b" * 64)
    assert value["checks"]["T6_positive_recall"] is False
    assert value["pass"] is False

