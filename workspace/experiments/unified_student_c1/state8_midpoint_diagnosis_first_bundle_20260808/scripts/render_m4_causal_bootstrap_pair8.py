#!/usr/bin/env python3
"""Render an isolated pair-8 donor-bootstrap diagnostic from the AMLT template."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


OLD_TRAIN_SHA256 = "f03c363cd3614965b495cab73629c8ab79eebce4d8dc72a873e205cc1212f4bc"
TRAIN_SHA256 = "865c12219deb23cf9db145c797244969721fce80323050dcd987b33d96cbfb1b"
SOURCE_JOB = "m4-pair8-overfit-sum-seed42"
JOB = "m4-pair8-overfit-sum-bootstrap-seed42"
SPECS = {
    "donor_only": {
        "job": JOB,
        "branch": "pair8_sum_bootstrap",
        "native_weight": "0.0",
        "description": "donor-only first quarter",
        "objective_mode": "bootstrap_then_symmetric",
    },
    "native25": {
        "job": "m4-pair8-overfit-sum-bootstrap-native25-seed42",
        "branch": "pair8_sum_bootstrap_native25",
        "native_weight": "0.25",
        "description": "native-anchor 0.25 and donor 1.0 first quarter",
        "objective_mode": "bootstrap_then_symmetric",
    },
    "balanced": {
        "job": "m4-pair8-overfit-sum-bootstrap-balanced-seed42",
        "branch": "pair8_sum_bootstrap_balanced",
        "native_weight": "0.0",
        "description": "donor-only first quarter, balanced directional tail",
        "objective_mode": "bootstrap_then_directional",
    },
    "native25_balanced": {
        "job": "m4-pair8-overfit-sum-bootstrap-native25-balanced-seed42",
        "branch": "pair8_sum_bootstrap_native25_balanced",
        "native_weight": "0.25",
        "description": "native-anchor 0.25 first quarter, balanced directional tail",
        "objective_mode": "bootstrap_then_directional",
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


def render(base: dict, variant: str = "donor_only") -> dict:
    spec = SPECS[variant]
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == SOURCE_JOB),
        None,
    )
    if source is None:
        raise ValueError(f"Missing source job: {SOURCE_JOB}")
    result = copy.deepcopy(base)
    result["description"] = (
        "Eight-pair recurrent causal bootstrap diagnostic: sequence-sum "
        f"preference, {spec['description']}"
    )
    job = copy.deepcopy(source)
    job["name"] = spec["job"]
    command = str(job["command"])
    command = replace_once(command, OLD_TRAIN_SHA256, TRAIN_SHA256)
    command = replace_once(
        command,
        '    --causal-preference-reduction "$$PREFERENCE_REDUCTION" \\\n'
        "    --causal-repeat-fraction 1.0 \\\n",
        '    --causal-preference-reduction "$$PREFERENCE_REDUCTION" \\\n'
        '    --causal-objective-mode "$$OBJECTIVE_MODE" \\\n'
        '    --causal-bootstrap-fraction "$$BOOTSTRAP_FRACTION" \\\n'
        '    --causal-bootstrap-native-weight "$$BOOTSTRAP_NATIVE_WEIGHT" \\\n'
        '    --causal-bootstrap-donor-weight "$$BOOTSTRAP_DONOR_WEIGHT" \\\n'
        "    --causal-repeat-fraction 1.0 \\\n",
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
        "    'state_relative_delta_first_50': sum(float(x['causal_state_relative_delta']) for x in causal[:50]) / 50,\n"
        "    'state_relative_delta_last_50': sum(float(x['causal_state_relative_delta']) for x in causal[-50:]) / 50,\n",
    )
    job["command"] = command
    env = job["submit_args"]["env"]
    env.update(
        {
            "BRANCH": spec["branch"],
            "PREFERENCE_REDUCTION": "sum",
            "OBJECTIVE_MODE": spec["objective_mode"],
            "BOOTSTRAP_FRACTION": "0.25",
            "BOOTSTRAP_NATIVE_WEIGHT": spec["native_weight"],
            "BOOTSTRAP_DONOR_WEIGHT": "1.0",
        }
    )
    result["jobs"] = [job]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(SPECS), default="donor_only")
    args = parser.parse_args()
    value = render(load(args.base), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(json.dumps({"job": SPECS[args.variant]["job"], "variant": args.variant, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
