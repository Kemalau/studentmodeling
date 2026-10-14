#!/usr/bin/env python3
"""Mechanism-only T3 evaluation through training-time direct teacher states.

The raw observed history is visible only while constructing the layerwise
teacher state.  The decoder receives that state plus the closed current query.
This distinguishes transition failure from state-to-text readout failure and
is explicitly ineligible for formal/final gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import evaluate_recurrent_formal_protocols as evaluator
from layerwise_recurrent_student_model import LayerwiseRecurrentStudentModel
from train_recurrent_student_diagnostic import direct_supervised_batch


@torch.inference_mode()
def direct_teacher_states_batched(
    wrapper,
    tokenizer,
    rows: list[dict],
    alias: str,
    max_length: int,
    batch_size: int,
) -> list[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("State batch size must be positive")
    if not isinstance(wrapper, LayerwiseRecurrentStudentModel):
        raise ValueError("Direct teacher state diagnosis requires layerwise state")
    device = next(wrapper.parameters()).device
    output: list[torch.Tensor] = []
    for row in rows:
        batch = direct_supervised_batch(
            tokenizer,
            str(row["direct_text"]),
            str(row["target"]),
            alias,
            max_length,
            device,
        )
        batch.pop("prompt_ends")
        captured = wrapper.capture_direct_state(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).state
        output.extend(state.detach().cpu() for state in captured)
        print(json.dumps({"direct_teacher_states": len(output), "total": len(rows)}), flush=True)
    return output


def argument_value(name: str) -> str:
    if name not in sys.argv:
        raise ValueError(f"Missing required argument {name}")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for {name}")
    return sys.argv[index + 1]


def main() -> None:
    output_dir = Path(argument_value("--output-dir"))
    evaluator.recurrent_states_batched = direct_teacher_states_batched
    evaluator.main()
    metrics_path = output_dir / "metrics.json"
    value = json.loads(metrics_path.read_text(encoding="utf-8"))
    value.update(
        {
            "state_source": "direct_teacher",
            "diagnostic_only": True,
            "eligible_for_final_gate": False,
            "information_route": "raw_history_to_direct_teacher_state_then_state_plus_current_query",
            "raw_history_visible_to_state_encoder": True,
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
        }
    )
    temporary = metrics_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(metrics_path)


if __name__ == "__main__":
    main()
