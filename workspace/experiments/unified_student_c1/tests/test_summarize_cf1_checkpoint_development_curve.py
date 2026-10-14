from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_cf1_checkpoint_development_curve import checkpoint_row, render_markdown


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_checkpoint_row_combines_task_probe_causal_and_loss(tmp_path: Path) -> None:
    curve = tmp_path / "curve"
    run = tmp_path / "run"
    step_dir = curve / "step_250"
    step_dir.mkdir(parents=True)
    (step_dir / "checkpoint_adapter.sha256").write_text("abc123\n", encoding="utf-8")
    dump(
        step_dir / "response_cross_masked.json",
        {"model": {"metrics": {"malrule_next_response": {"accuracy": 0.7, "macro_f1": 0.6}}}},
    )
    dump(
        step_dir / "state_probe_masked.json",
        {
            "metrics": {
                "state": {
                    "8": {"cross": {"accuracy": 0.8, "macro_f1": 0.7}},
                    "16": {"cross": {"accuracy": 0.9, "macro_f1": 0.85}},
                }
            }
        },
    )
    records = []
    categories = {
        0: {"none": "stay_B", "swap": "flip_to_A", "random": "stay_B", "samemal": "stay_B", "noop": "stay_B"},
        1: {"none": "stay_B", "swap": "stay_B", "random": "flip_to_A", "samemal": "stay_B", "noop": "stay_B"},
    }
    for pair_idx, conditions in categories.items():
        for condition, category in conditions.items():
            records.append(
                {
                    "pair_idx": pair_idx,
                    "condition": condition,
                    "category": category,
                    "pred": category,
                    "p_A": 0.5,
                    "p_B": 0.5,
                    "score_A": 0.0,
                    "score_B": 0.0,
                }
            )
    dump(step_dir / "state_swap_masked.json", {"records": records, "summary": {}})
    dump(
        run / "checkpoint-250" / "trainer_state.json",
        {"epoch": 0.2, "log_history": [{"step": 200, "loss": 0.4}, {"step": 250, "loss": 0.3}]},
    )

    row = checkpoint_row(curve, run, 250)
    assert row["response_cross_accuracy"] == 0.7
    assert row["state_probe_best_cross_accuracy"] == 0.9
    assert row["state_probe_best_layer"] == 16
    assert row["train_loss_latest"] == 0.3
    assert row["swap_flip_to_A"] == 0.5
    assert row["random_flip_to_A"] == 0.5
    assert row["D_specific"] == 0.0
    assert row["checkpoint_adapter_sha256"] == "abc123"
    assert "| 250 | 0.300 | 0.700 | 0.900 |" in render_markdown([row])
