#!/usr/bin/env python3
"""Render a one-A100 topology replicate of the CF4 donor-directional screen.

The scientific data, first 1,000-example schedule, optimizer-update count, and
frozen C4 gate match the four-GPU screen.  Four serial micro-batches replace
one four-rank distributed round, allowing Feeds to schedule the diagnostic on
an otherwise stranded single A100 without changing its effective batch size.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


SOURCE_JOB = "m4-causal-cf4-donor-directional-w1-round0to250-seed42"
JOB = "m4-causal-cf4-donor-directional-w1-single-gb4-step0to1000-seed42"
BRANCH = "donor_directional_cf4_w1_single_gb4"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one source fragment, found {text.count(old)}: {old!r}")
    return text.replace(old, new, 1)


def render(base: dict) -> dict:
    jobs = {str(job.get("name")): job for job in base.get("jobs") or []}
    if SOURCE_JOB not in jobs:
        raise ValueError(f"Missing source job: {SOURCE_JOB}")
    job = copy.deepcopy(jobs[SOURCE_JOB])
    command = str(job["command"])
    command = replace_once(
        command,
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_cf4_$${BRANCH}_seed42_round250_20260731"',
        'export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m4_causal_cf4_$${BRANCH}_seed42_step1000_20260731"',
    )

    block_start = "  resume_args=()\n  latest_checkpoint=\"\"\n"
    block_end = "  python -u scripts/train_recurrent_student_m4.py \\\n"
    if command.count(block_start) != 1 or command.count(block_end) != 1:
        raise ValueError("Could not locate CF4 resume block")
    before, tail = command.split(block_start, 1)
    _old_resume, after = tail.split(block_end, 1)
    resume = r'''  resume_args=()
  latest_checkpoint=""
  mapfile -t branch_checkpoints < <(find "$$RUN_DIR" -mindepth 1 -maxdepth 1 \
    -type d -name 'checkpoint-*' ! -name 'checkpoint-round-*' | sort -V)
  for ((i=$${#branch_checkpoints[@]}-1; i>=0; i--)); do
    candidate="$${branch_checkpoints[$$i]}"
    if [[ -f "$$candidate/checkpoint_completion.json" \
          && -f "$$candidate/adapter/adapter_model.safetensors" \
          && -f "$$candidate/recurrent_state.pt" ]]; then
      latest_checkpoint="$$candidate"
      break
    fi
  done
  if [[ -n "$$latest_checkpoint" ]]; then
    python - "$$latest_checkpoint" "$$BRANCH" "$$CAUSAL_WEIGHT" <<'PY'
import hashlib, json, sys, torch
root, branch = sys.argv[1], sys.argv[2]
receipt = json.load(open(root + '/checkpoint_completion.json', encoding='utf-8'))
payload = torch.load(root + '/recurrent_state.pt', map_location='cpu', weights_only=False)
digest = lambda path: hashlib.sha256(open(path, 'rb').read()).hexdigest()
assert receipt.get('status') == 'complete', receipt
assert 0 < int(receipt.get('step', -1)) <= 1000, receipt
assert int(receipt['step']) % 50 == 0, receipt
assert receipt['sha256']['adapter_model'] == digest(root + '/adapter/adapter_model.safetensors')
assert receipt['sha256']['recurrent_state'] == digest(root + '/recurrent_state.pt')
config = payload['config']
assert int(payload.get('step', -1)) == int(receipt['step']), payload
assert config.get('causal_repeat_fraction') == 1.0, config
assert float(config['causal_weight']) == float(sys.argv[3]), config
assert config.get('causal_preference_reduction') == 'sum', config
assert config.get('causal_objective_mode') == 'donor_directional', config
assert config.get('diagnostic_max_micro_steps') == 1000, config
assert config.get('distributed') is False, config
assert int(config.get('gradient_accumulation_steps')) == 4, config
assert branch == 'donor_directional_cf4_w1_single_gb4', branch
PY
    resume_args=(--resume-from-checkpoint "$$latest_checkpoint/recurrent_state.pt")
  fi
  python -u scripts/train_recurrent_student_m4.py \
'''
    command = before + resume + after
    command = replace_once(
        command,
        "--gradient-accumulation-steps 1",
        "--gradient-accumulation-steps 4",
    )
    command = replace_once(
        command,
        "    --distributed-mpi \\\n    --distributed-world-size 4 \\\n    --diagnostic-stop-after-round 250 \\\n",
        "    --diagnostic-max-micro-steps 1000 \\\n",
    )
    command = replace_once(
        command,
        'if [[ "$${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then\n  exit 0\nfi\n',
        "",
    )
    command = replace_once(
        command,
        "assert config.get('diagnostic_stop_after_round') == 250, config",
        "assert config.get('diagnostic_max_micro_steps') == 1000, config\n"
        "assert config.get('distributed') is False, config\n"
        "assert int(config.get('gradient_accumulation_steps')) == 4, config\n"
        "assert completion.get('optimizer_steps') == 250, completion",
    )
    command = replace_once(command, "'round_end': 500", "'round_end': 1000")
    command = replace_once(
        command,
        "sum(float(x['causal_margin_loss']) for x in causal[:1000]) / 1000",
        "sum(float(x['causal_margin_loss']) for x in causal[:500]) / 500",
    )
    command = replace_once(
        command,
        "sum(float(x['causal_margin_loss']) for x in causal[1000:]) / 1000",
        "sum(float(x['causal_margin_loss']) for x in causal[500:]) / 500",
    )

    job["name"] = JOB
    job["sku"] = "80G1-A100"
    job["process_count_per_node"] = 1
    job["mpi"] = False
    job["command"] = command
    env = job["submit_args"]["env"]
    env["BRANCH"] = BRANCH
    env["CAUSAL_WEIGHT"] = "1.0"

    result = copy.deepcopy(base)
    result["description"] = (
        "CF4 donor-directional single-A100 queue hedge: same first 1000 examples, "
        "250 optimizer updates, effective global batch 4, and frozen 150-pair C4"
    )
    result["jobs"] = [job]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("amlt/feeds_m4_cf4_diagnostic_20260731.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(json.dumps({"job": JOB, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
