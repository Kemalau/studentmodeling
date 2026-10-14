#!/usr/bin/env python3
"""Render an isolated one-A100 scheduling sensitivity for the selected bootstrap arm.

The G1 job preserves the first 2,000 global examples, 500 optimizer updates,
global batch four, seed, data order, and causal curriculum of the G4 diagnostic.
It is deliberately written to a separate remote run directory and is not
presented as bitwise equivalent to four-rank data parallel training.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "m4-causal-bootstrap-sum-w1-round0to500-seed42"
TARGET_JOB = "m4-causal-bootstrap-sum-w1-g1-sensitivity-seed42"
G1_BOOTSTRAP_FRACTION = "0.25"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}, got {text.count(old)}")
    return text.replace(old, new, 1)


def render(base: dict) -> dict:
    value = copy.deepcopy(base)
    jobs = [job for job in value.get("jobs") or [] if job.get("name") == SOURCE_JOB]
    if len(jobs) != 1:
        raise ValueError("Selected G4 bootstrap source job is missing or duplicated")
    job = jobs[0]
    command = str(job["command"])
    command = replace_once(
        command,
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_$${BRANCH}_seed42_round500_20260731"',
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_bootstrap_sum_w1_g1_sensitivity_seed42_20260731"',
    )
    start = command.index('  resume_args=()\n')
    end = command.index('  python -u scripts/train_recurrent_student_m4.py \\\n', start)
    resume = '''  resume_args=()
  latest_checkpoint=""
  mapfile -t sensitivity_checkpoints < <(find "$$RUN_DIR" -mindepth 1 -maxdepth 1 \\
    -type d -name 'checkpoint-*' | sort -V)
  for ((i=$${#sensitivity_checkpoints[@]}-1; i>=0; i--)); do
    candidate="$${sensitivity_checkpoints[$$i]}"
    if [[ -f "$$candidate/checkpoint_completion.json" \\
          && -f "$$candidate/adapter/adapter_model.safetensors" \\
          && -f "$$candidate/recurrent_state.pt" ]]; then
      latest_checkpoint="$$candidate"
      break
    fi
  done
  if [[ -n "$$latest_checkpoint" ]]; then
    python - "$$latest_checkpoint" "$$CAUSAL_WEIGHT" <<'PY'
import hashlib, json, sys, torch
root = sys.argv[1]
receipt = json.load(open(root + '/checkpoint_completion.json', encoding='utf-8'))
payload = torch.load(root + '/recurrent_state.pt', map_location='cpu', weights_only=False)
digest = lambda path: hashlib.sha256(open(path, 'rb').read()).hexdigest()
assert receipt.get('status') == 'complete', receipt
assert 0 < int(receipt.get('step', -1)) <= 2000, receipt
assert int(receipt['step']) % 500 == 0, receipt
assert receipt['sha256']['adapter_model'] == digest(root + '/adapter/adapter_model.safetensors')
assert receipt['sha256']['recurrent_state'] == digest(root + '/recurrent_state.pt')
assert int(payload.get('step', -1)) == int(receipt['step']), payload
config = payload['config']
assert config.get('distributed') is False, config
assert int(config.get('gradient_accumulation_steps', -1)) == 4, config
assert config.get('causal_preference_reduction') == 'sum', config
assert config.get('causal_objective_mode') == 'bootstrap_then_symmetric', config
assert abs(float(config.get('causal_bootstrap_fraction')) - 0.25) < 1e-12, config
assert float(config['causal_weight']) == float(sys.argv[2]), config
PY
    resume_args=(--resume-from-checkpoint "$$latest_checkpoint/recurrent_state.pt")
  fi
'''
    command = command[:start] + resume + command[end:]
    command = replace_once(command, "--gradient-accumulation-steps 1", "--gradient-accumulation-steps 4")
    command = replace_once(command, "--save-steps 250", "--save-steps 500")
    command = replace_once(
        command,
        "--causal-bootstrap-fraction 0.041666666666666664",
        f"--causal-bootstrap-fraction {G1_BOOTSTRAP_FRACTION}",
    )
    command = replace_once(
        command,
        "    --distributed-mpi \\\n    --distributed-world-size 4 \\\n    --diagnostic-stop-after-round 500 \\\n",
        "    --diagnostic-max-micro-steps 2000 \\\n",
    )
    command = replace_once(
        command,
        '''if [[ "$${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then
  exit 0
fi
''',
        "",
    )
    command = replace_once(
        command,
        "assert config.get('diagnostic_stop_after_round') == 500, config\n",
        "assert config.get('diagnostic_max_micro_steps') == 2000, config\n"
        "assert config.get('distributed') is False, config\n"
        "assert int(config.get('gradient_accumulation_steps')) == 4, config\n"
        "assert int(completion.get('optimizer_steps')) == 500, completion\n",
    )
    command = replace_once(
        command,
        "    'branch': branch,\n",
        "    'branch': 'bootstrap_sum_w1_g1_sensitivity',\n"
        "    'source_branch': branch,\n"
        "    'topology': 'single_a100_accum4_sensitivity',\n",
    )
    job["name"] = TARGET_JOB
    job["sku"] = "80G1-A100"
    job["mpi"] = False
    job["process_count_per_node"] = 1
    job["command"] = command
    job["submit_args"]["env"]["BRANCH"] = "bootstrap_sum_w1_g1_sensitivity"
    job["submit_args"]["env"]["CAUSAL_WEIGHT"] = "1.0"
    value["description"] = (
        "Isolated one-A100 scheduling sensitivity for selected sequence-sum "
        "bootstrap curriculum; 2000 examples, accumulation 4, 500 updates"
    )
    value["jobs"] = [job]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(yaml.safe_dump(render(load(args.base)), sort_keys=False))
    print(f"rendered={args.output}")


if __name__ == "__main__":
    main()
