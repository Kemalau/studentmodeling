#!/usr/bin/env python3
"""Render dense sequence-sum bootstrap curricula from the verified rehearsal YAML."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


OLD_TRAIN_SHA256 = "f03c363cd3614965b495cab73629c8ab79eebce4d8dc72a873e205cc1212f4bc"
TRAIN_SHA256 = "865c12219deb23cf9db145c797244969721fce80323050dcd987b33d96cbfb1b"
BOOTSTRAP_FRACTION = "0.041666666666666664"
SPECS = {
    "bootstrap_sum_w1": {
        "source": "m4-causal-repeat100-sum-w1-round0to500-seed42",
        "job": "m4-causal-bootstrap-sum-w1-round0to500-seed42",
        "weight": "1.0",
        "native_weight": "0.0",
        "objective_mode": "bootstrap_then_symmetric",
    },
    "bootstrap_sum_w4": {
        "source": "m4-causal-repeat100-sum-w4-round0to500-seed42",
        "job": "m4-causal-bootstrap-sum-w4-round0to500-seed42",
        "weight": "4.0",
        "native_weight": "0.0",
        "objective_mode": "bootstrap_then_symmetric",
    },
    "bootstrap_native25_sum_w1": {
        "source": "m4-causal-repeat100-sum-w1-round0to500-seed42",
        "job": "m4-causal-bootstrap-native25-sum-w1-round0to500-seed42",
        "weight": "1.0",
        "native_weight": "0.25",
        "objective_mode": "bootstrap_then_symmetric",
    },
    "bootstrap_native25_sum_w4": {
        "source": "m4-causal-repeat100-sum-w4-round0to500-seed42",
        "job": "m4-causal-bootstrap-native25-sum-w4-round0to500-seed42",
        "weight": "4.0",
        "native_weight": "0.25",
        "objective_mode": "bootstrap_then_symmetric",
    },
    "bootstrap_balanced_sum_w1": {
        "source": "m4-causal-repeat100-sum-w1-round0to500-seed42",
        "job": "m4-causal-bootstrap-balanced-sum-w1-round0to500-seed42",
        "weight": "1.0",
        "native_weight": "0.0",
        "objective_mode": "bootstrap_then_directional",
    },
    "bootstrap_balanced_sum_w4": {
        "source": "m4-causal-repeat100-sum-w4-round0to500-seed42",
        "job": "m4-causal-bootstrap-balanced-sum-w4-round0to500-seed42",
        "weight": "4.0",
        "native_weight": "0.0",
        "objective_mode": "bootstrap_then_directional",
    },
    "bootstrap_native25_balanced_sum_w1": {
        "source": "m4-causal-repeat100-sum-w1-round0to500-seed42",
        "job": "m4-causal-bootstrap-native25-balanced-sum-w1-round0to500-seed42",
        "weight": "1.0",
        "native_weight": "0.25",
        "objective_mode": "bootstrap_then_directional",
    },
    "bootstrap_native25_balanced_sum_w4": {
        "source": "m4-causal-repeat100-sum-w4-round0to500-seed42",
        "job": "m4-causal-bootstrap-native25-balanced-sum-w4-round0to500-seed42",
        "weight": "4.0",
        "native_weight": "0.25",
        "objective_mode": "bootstrap_then_directional",
    },
    "donor_directional_sum_w1": {
        "source": "m4-causal-repeat100-sum-w1-round0to500-seed42",
        "job": "m4-causal-donor-directional-sum-w1-round0to500-seed42",
        "weight": "1.0",
        "native_weight": "0.0",
        "objective_mode": "donor_directional",
    },
    "donor_directional_sum_w4": {
        "source": "m4-causal-repeat100-sum-w4-round0to500-seed42",
        "job": "m4-causal-donor-directional-sum-w4-round0to500-seed42",
        "weight": "4.0",
        "native_weight": "0.0",
        "objective_mode": "donor_directional",
    },
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one template fragment, found {text.count(old)}")
    return text.replace(old, new, 1)


def bootstrap_command(source: str, native_weight: str, objective_mode: str) -> str:
    command = replace_once(source, OLD_TRAIN_SHA256, TRAIN_SHA256)
    command = replace_once(
        command,
        "    --causal-preference-reduction sum \\\n"
        "    --causal-repeat-fraction 1.0 \\\n",
        "    --causal-preference-reduction sum \\\n"
        f"    --causal-objective-mode {objective_mode} \\\n"
        f"    --causal-bootstrap-fraction {BOOTSTRAP_FRACTION} \\\n"
        f"    --causal-bootstrap-native-weight {native_weight} \\\n"
        "    --causal-bootstrap-donor-weight 1.0 \\\n"
        "    --causal-repeat-fraction 1.0 \\\n",
    )
    anchor = "assert config.get('planned_causal_exposures') == 12000, config\n"
    command = replace_once(
        command,
        anchor,
        anchor
        + f"assert config.get('causal_objective_mode') == '{objective_mode}', config\n"
        + f"assert abs(float(config.get('causal_bootstrap_fraction')) - {BOOTSTRAP_FRACTION}) < 1e-12, config\n",
    )
    command = replace_once(
        command,
        "    'causal_rows_total': len(causal),\n",
        "    'causal_rows_total': len(causal),\n"
        "    'objective_mode': config['causal_objective_mode'],\n"
        "    'objective_phase_counts': {\n"
        "        phase: sum(x.get('causal_objective_phase') == phase for x in causal)\n"
        "        for phase in sorted({x.get('causal_objective_phase') for x in causal})\n"
        "    },\n"
        "    'native_advantage_first_500': sum(float(x['causal_native_advantage']) for x in causal[:500]) / 500,\n"
        "    'donor_advantage_first_500': sum(float(x['causal_donor_advantage']) for x in causal[:500]) / 500,\n"
        "    'preference_last_500': sum(float(x['causal_preference']) for x in causal[-500:]) / 500,\n"
        "    'state_relative_delta_first_500': sum(float(x['causal_state_relative_delta']) for x in causal[:500]) / 500,\n"
        "    'state_relative_delta_last_500': sum(float(x['causal_state_relative_delta']) for x in causal[-500:]) / 500,\n",
    )
    resume_start = "  resume_args=()\n"
    training_start = "  python -u scripts/train_recurrent_student_m4.py \\\n"
    start = command.find(resume_start)
    end = command.find(training_start, start)
    if start < 0 or end < 0 or command.find(resume_start, start + 1) >= 0:
        raise ValueError("Could not locate unique dense-rehearsal resume block")
    robust_resume = f'''  resume_args=()
  latest_checkpoint=""
  mapfile -t branch_checkpoints < <(find "$$RUN_DIR" -mindepth 1 -maxdepth 1 \\
    -type d -name 'checkpoint-round-*' | sort -V)
  for ((i=$${{#branch_checkpoints[@]}}-1; i>=0; i--)); do
    candidate="$${{branch_checkpoints[$$i]}}"
    if [[ -f "$$candidate/checkpoint_completion.json" \\
          && -f "$$candidate/adapter/adapter_model.safetensors" \\
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
assert receipt.get('world_size') == 4, receipt
assert 0 < int(receipt.get('round', -1)) <= 500, receipt
assert int(receipt['round']) % 250 == 0, receipt
assert receipt['sha256']['adapter_model'] == digest(root + '/adapter/adapter_model.safetensors')
assert receipt['sha256']['recurrent_state'] == digest(root + '/recurrent_state.pt')
config = payload['config']
assert int(payload.get('round', -1)) == int(receipt['round']), payload
assert config.get('causal_repeat_fraction') == 1.0, config
assert float(config['causal_weight']) == float(sys.argv[3]), config
assert config.get('causal_preference_reduction') == 'sum', config
assert config.get('causal_objective_mode') == '{objective_mode}', config
assert abs(float(config.get('causal_bootstrap_fraction')) - {BOOTSTRAP_FRACTION}) < 1e-12, config
assert branch.startswith(('bootstrap_', 'donor_directional_')), branch
PY
    resume_args=(--resume-from-checkpoint "$$latest_checkpoint/recurrent_state.pt")
  fi
'''
    command = command[:start] + robust_resume + command[end:]
    return command


def render(base: dict, branches: list[str] | None = None) -> dict:
    selected = branches or ["bootstrap_sum_w1", "bootstrap_sum_w4"]
    by_name = {str(job.get("name")): job for job in base.get("jobs") or []}
    result = copy.deepcopy(base)
    result["description"] = (
        "Dense sequence-sum recurrent causal curriculum: asymmetric first 500 "
        "global examples, controlled tail objective, weights 1 and 4"
    )
    result["jobs"] = []
    for branch in selected:
        spec = SPECS[branch]
        if spec["source"] not in by_name:
            raise ValueError(f"Missing source job: {spec['source']}")
        job = copy.deepcopy(by_name[spec["source"]])
        job["name"] = spec["job"]
        job["command"] = bootstrap_command(
            str(job["command"]), spec["native_weight"], spec["objective_mode"]
        )
        env = job["submit_args"]["env"]
        env["BRANCH"] = branch
        env["CAUSAL_WEIGHT"] = spec["weight"]
        result["jobs"].append(job)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branches", nargs="*", choices=tuple(SPECS))
    args = parser.parse_args()
    value = render(load(args.base), args.branches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(json.dumps({"branches": args.branches or list(SPECS), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
