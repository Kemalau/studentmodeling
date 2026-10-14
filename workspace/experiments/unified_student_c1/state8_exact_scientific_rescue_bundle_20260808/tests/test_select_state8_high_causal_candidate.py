from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arm(
    root: Path,
    d_specific: float,
    cross: float = 0.5,
    eedi: float = 0.2,
    c4_pass: bool = True,
) -> None:
    (root / "c4_semantic").mkdir(parents=True)
    (root / "t3_semantic").mkdir()
    (root / "c4_semantic/summary.json").write_text(
        json.dumps(
            {
                "semantic_judge_authoritative": True,
                "semantic_D_specific": d_specific,
                "same_rule_recipient_retention": 0.4,
                "random_matches_either_stable_target_rate": 0.1,
                "pass": c4_pass,
                "generation": {
                    "artifact_sha256": {
                        "adapter_model": "a" * 64,
                        "recurrent_state": "b" * 64,
                    }
                },
            }
        )
    )
    (root / "t3_semantic/metrics.json").write_text(
        json.dumps(
            {
                "judge_backend": "trapi",
                "judge_errors": 0,
                "generation_rows": 300,
                "judged_rows": 300,
                "metrics_by_group": {
                    "diagnose_misconception/eedi_mining/open_taxonomy_generation": {"llm_judge_accuracy": eedi, "llm_judge_exact_shortcuts": 0},
                    "diagnose_misconception/malrule/synthetic_cross_template_open_generation": {"llm_judge_accuracy": cross, "llm_judge_exact_shortcuts": 0},
                    "diagnose_misconception/malrule/synthetic_same_template_open_generation": {"llm_judge_accuracy": 0.4, "llm_judge_exact_shortcuts": 0},
                }
            }
        )
    )
    (root / "publish_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": "state8_stateonly_repair_pilot_publish_v1",
                "state_shape": [64, 8, 5120],
                "raw_history_visible_to_decoder": False,
                "train_entrypoint": (
                    "scripts/train_state8_taxonomy_causal_repair.py"
                    if root.name == "taxonomy"
                    else "scripts/train_state8_semantic_causal_repair.py"
                ),
                "artifact_sha256": {
                    "adapter_model": "a" * 64,
                    "recurrent_state": "b" * 64,
                },
            }
        )
    )
    if root.name == "taxonomy":
        (root / "taxonomy_causal_repair.json").write_text(
            json.dumps(
                {
                    "stage": "state8_taxonomy_causal_repair_v1",
                    "status": "complete",
                    "contract": "source_aware_no_answer_labels_v1",
                    "candidate_taxonomy_visible_to_decoder": False,
                    "raw_history_visible_to_decoder": False,
                }
            )
        )


def test_mid_arm_can_win_three_way_selection(tmp_path: Path) -> None:
    for name, value in (("joint", 0.16), ("updater", 0.18), ("mid", 0.30)):
        arm(tmp_path / name, value)
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_state8_high_causal_candidate.py"),
            "--joint",
            str(tmp_path / "joint"),
            "--updater",
            str(tmp_path / "updater"),
            "--mid",
            str(tmp_path / "mid"),
            "--output",
            str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text())
    assert value["selected"] == "mid"
    assert set(value["arms"]) == {"joint", "updater", "mid"}


def test_zero_eedi_arm_cannot_be_promoted(tmp_path: Path) -> None:
    for name, value in (("joint", 0.30), ("updater", 0.25), ("mid", 0.20)):
        arm(tmp_path / name, value, eedi=0.0)
    arm(tmp_path / "eedi", 0.18, eedi=0.20)
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_state8_high_causal_candidate.py"),
            "--joint", str(tmp_path / "joint"),
            "--updater", str(tmp_path / "updater"),
            "--mid", str(tmp_path / "mid"),
            "--eedi", str(tmp_path / "eedi"),
            "--output", str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text())
    assert value["selected"] == "eedi"
    assert value["arms"]["joint"]["eligible"] is False


def test_failed_authoritative_c4_gate_cannot_be_promoted(tmp_path: Path) -> None:
    arm(tmp_path / "joint", 0.50, eedi=0.30, c4_pass=False)
    arm(tmp_path / "updater", 0.18, eedi=0.20)
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_state8_high_causal_candidate.py"),
            "--joint", str(tmp_path / "joint"),
            "--updater", str(tmp_path / "updater"),
            "--output", str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text())
    assert value["selected"] == "updater"
    assert value["arms"]["joint"]["eligible"] is False


def test_taxonomy_arm_can_win_five_way_selection(tmp_path: Path) -> None:
    for name, value in (
        ("joint", 0.16),
        ("updater", 0.17),
        ("mid", 0.18),
        ("eedi", 0.19),
        ("taxonomy", 0.35),
    ):
        arm(tmp_path / name, value)
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_state8_high_causal_candidate.py"),
            "--joint", str(tmp_path / "joint"),
            "--updater", str(tmp_path / "updater"),
            "--mid", str(tmp_path / "mid"),
            "--eedi", str(tmp_path / "eedi"),
            "--taxonomy", str(tmp_path / "taxonomy"),
            "--output", str(output),
        ],
        check=True,
    )
    value = json.loads(output.read_text())
    assert value["selected"] == "taxonomy"
    assert set(value["arms"]) == {"joint", "updater", "mid", "eedi", "taxonomy"}
