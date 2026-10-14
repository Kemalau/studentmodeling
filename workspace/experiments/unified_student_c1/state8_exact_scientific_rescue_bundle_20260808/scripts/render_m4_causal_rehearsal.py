#!/usr/bin/env python3
"""Render dense causal-rehearsal M4 diagnostics from the generated branch template."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


SPECS = {
    "repeat100_w1": {
        "job": "m4-causal-repeat100-w1-round0to500-seed42",
        "weight": "1.0",
    },
    "repeat100_w4": {
        "job": "m4-causal-repeat100-w4-round0to500-seed42",
        "weight": "4.0",
        "preference_reduction": "mean",
    },
    "repeat100_sum_w1": {
        "job": "m4-causal-repeat100-sum-w1-round0to500-seed42",
        "weight": "1.0",
        "preference_reduction": "sum",
    },
    "repeat100_sum_w4": {
        "job": "m4-causal-repeat100-sum-w4-round0to500-seed42",
        "weight": "4.0",
        "preference_reduction": "sum",
    },
}

# Preserve the historical default so rendering the original dense config remains
# byte-for-protocol compatible. Sequence-aligned branches are requested explicitly.
SPECS["repeat100_w1"]["preference_reduction"] = "mean"

TRAIN_SHA256 = "f03c363cd3614965b495cab73629c8ab79eebce4d8dc72a873e205cc1212f4bc"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one template fragment, found {text.count(old)}: {old[:80]!r}")
    return text.replace(old, new, 1)


def rehearsal_command(source: str, branches: set[str], preference_reduction: str) -> str:
    command = source
    command = replace_once(
        command,
        "export MAIN_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/"
        "world_model_v1_m4_generation_kd_trainable_seed42_fixedu4_v2_20260730\n"
        "export CHECKPOINT=\"$$MAIN_RUN_DIR/checkpoint-round-250\"\n",
        "",
    )
    for fragment in (
        'test -f "$$CHECKPOINT/adapter/adapter_model.safetensors"\n',
        'test -f "$$CHECKPOINT/recurrent_state.pt"\n',
        'test -f "$$CHECKPOINT/checkpoint_completion.json"\n',
    ):
        command = replace_once(command, fragment, "")
    command = replace_once(
        command,
        "24f066362c951edb932caf2b98c4cf909b7a211cc090a7a3623946d659bfc1d2",
        TRAIN_SHA256,
    )
    checkpoint_audit_start = 'python - "$$CHECKPOINT" <<\'PY\'\n'
    checkpoint_audit_end = "PY\n\nexec 7>/tmp/m4-causal-branch-data.lock\n"
    if command.count(checkpoint_audit_start) != 1 or command.count(checkpoint_audit_end) != 1:
        raise ValueError("Could not locate inherited round-250 checkpoint audit")
    before, tail = command.split(checkpoint_audit_start, 1)
    _, after = tail.split(checkpoint_audit_end, 1)
    command = before + "exec 7>/tmp/m4-causal-branch-data.lock\n" + after

    start = command.index("normalize_args=()")
    end = command.index('if [[ "$${OMPI_COMM_WORLD_RANK:-0}" != 0 ]]; then')
    training = r'''if [[ ! -f "$$RUN_DIR/completion.json" ]]; then
  resume_args=()
  if [[ -f "$$RUN_DIR/checkpoint-round-500/checkpoint_completion.json" \
        && -f "$$RUN_DIR/checkpoint-round-500/adapter/adapter_model.safetensors" \
        && -f "$$RUN_DIR/checkpoint-round-500/recurrent_state.pt" ]]; then
    python - "$$RUN_DIR/checkpoint-round-500" "$$BRANCH" "$$CAUSAL_WEIGHT" <<'PY'
import hashlib, json, sys, torch
root, branch = sys.argv[1], sys.argv[2]
receipt = json.load(open(root + '/checkpoint_completion.json', encoding='utf-8'))
payload = torch.load(root + '/recurrent_state.pt', map_location='cpu', weights_only=False)
digest = lambda path: hashlib.sha256(open(path, 'rb').read()).hexdigest()
assert receipt.get('status') == 'complete' and receipt.get('round') == 500, receipt
assert receipt.get('world_size') == 4, receipt
assert receipt['sha256']['adapter_model'] == digest(root + '/adapter/adapter_model.safetensors')
assert receipt['sha256']['recurrent_state'] == digest(root + '/recurrent_state.pt')
config = payload['config']
assert config.get('causal_repeat_fraction') == 1.0, config
assert float(config['causal_weight']) == float(sys.argv[3]), config
assert branch in __ALLOWED_BRANCHES__, branch
PY
    resume_args=(--resume-from-checkpoint "$$RUN_DIR/checkpoint-round-500/recurrent_state.pt")
  fi
  python -u scripts/train_recurrent_student_m4.py \
    --model "$$LOCAL_MODEL" \
    --init-adapter "$$M3_RUN_DIR/final_adapter" \
    --init-recurrent-state "$$M3_RUN_DIR/recurrent_state.pt" \
    --train-jsonl "$$UNIFIED_DATA/train.jsonl" \
    --valid-jsonl "$$UNIFIED_DATA/valid.jsonl" \
    --causal-train-jsonl "$$CAUSAL_DATA/train.jsonl" \
    --causal-valid-jsonl "$$CAUSAL_DATA/valid.jsonl" \
    --teacher-cache "$$CACHE_DIR" \
    --output-dir "$$RUN_DIR" \
    --state-token-alias '<|vision_pad|>' \
    --max-length 1536 \
    --learning-rate 1e-5 \
    --generation-weight 1.0 \
    --behavior-kd-weight 0.5 \
    --behavior-kd-temperature 1.0 \
    --consistency-weight 0.25 \
    --direct-replay-weight 0.25 \
    --causal-weight "$$CAUSAL_WEIGHT" \
    --causal-margin 0.5 \
    --causal-preference-reduction __PREFERENCE_REDUCTION__ \
    --causal-repeat-fraction 1.0 \
    --gradient-accumulation-steps 1 \
    --trainable-projections \
    --state-projection-rank 64 \
    --eval-limit 300 \
    --causal-eval-limit 165 \
    --save-steps 250 \
    --distributed-mpi \
    --distributed-world-size 4 \
    --diagnostic-stop-after-round 500 \
    --seed 42 \
    "$${resume_args[@]}"
fi
'''
    allowed_branches = "{" + ", ".join(repr(value) for value in sorted(branches)) + "}"
    training = training.replace(
        "__ALLOWED_BRANCHES__", allowed_branches
    ).replace("__PREFERENCE_REDUCTION__", preference_reduction)
    command = command[:start] + training + command[end:]

    old_summary = r'''assert completion.get('micro_steps') == 2000, completion
assert completion.get('causal_pairs_seen') == 288, completion
assert config.get('diagnostic_stop_after_round') == 500, config
assert float(config.get('causal_weight')) == float(sys.argv[3]), config
assert bool(config.get('normalize_causal_exposure')) == bool(int(sys.argv[4])), config
causal = [row for row in history if row.get('causal_pair_id')]
payload = {
    'branch': branch,
    'round_start': 250,
    'round_end': 500,
    'causal_weight': float(config['causal_weight']),
    'causal_exposure_multiplier': float(config['causal_exposure_multiplier']),
    'effective_causal_weight': float(config['causal_weight']) * float(config['causal_exposure_multiplier']),
    'causal_rows_total': len(causal),
    'margin_loss_first_half': sum(float(x['causal_margin_loss']) for x in causal[:144]) / 144,
    'margin_loss_second_half': sum(float(x['causal_margin_loss']) for x in causal[144:]) / 144,
}'''
    new_summary = r'''assert completion.get('micro_steps') == 2000, completion
assert completion.get('causal_pairs_seen') == 2000, completion
assert completion.get('unique_causal_pairs_seen') == 1726, completion
assert config.get('diagnostic_stop_after_round') == 500, config
assert config.get('causal_repeat_fraction') == 1.0, config
assert config.get('planned_causal_exposures') == 12000, config
assert float(config.get('causal_weight')) == float(sys.argv[3]), config
causal = [row for row in history if row.get('causal_pair_id')]
payload = {
    'branch': branch,
    'round_start': 0,
    'round_end': 500,
    'causal_weight': float(config['causal_weight']),
    'causal_exposure_multiplier': float(config['causal_exposure_multiplier']),
    'effective_causal_weight': float(config['causal_weight']) * float(config['causal_exposure_multiplier']),
    'causal_rows_total': len(causal),
    'margin_loss_first_half': sum(float(x['causal_margin_loss']) for x in causal[:1000]) / 1000,
    'margin_loss_second_half': sum(float(x['causal_margin_loss']) for x in causal[1000:]) / 1000,
}'''
    command = replace_once(command, old_summary, new_summary)
    command = replace_once(
        command,
        'python - "$$RUN_DIR" "$$BRANCH" "$$CAUSAL_WEIGHT" "$$NORMALIZE_CAUSAL" <<\'PY\'',
        'python - "$$RUN_DIR" "$$BRANCH" "$$CAUSAL_WEIGHT" <<\'PY\'',
    )
    return command


def render(base: dict, branches: list[str]) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == "m4-causal-raw4-round250-to500-seed42"),
        None,
    )
    if source is None:
        raise ValueError("Raw4 source job is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "Dense recurrent causal-rehearsal diagnostics from M3 through round 500; "
        "every real-data position receives a matched causal pair"
    )
    result["jobs"] = []
    for branch in branches:
        spec = SPECS[branch]
        job = copy.deepcopy(source)
        job["name"] = spec["job"]
        job["command"] = rehearsal_command(
            str(job["command"]), set(branches), spec["preference_reduction"]
        )
        env = job["submit_args"]["env"]
        env["BRANCH"] = branch
        env["CAUSAL_WEIGHT"] = spec["weight"]
        env.pop("NORMALIZE_CAUSAL", None)
        result["jobs"].append(job)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branches", nargs="*", choices=tuple(SPECS))
    args = parser.parse_args()
    branches = args.branches or ["repeat100_w1", "repeat100_w4"]
    value = render(load(args.base), branches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(json.dumps({"branches": branches, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
