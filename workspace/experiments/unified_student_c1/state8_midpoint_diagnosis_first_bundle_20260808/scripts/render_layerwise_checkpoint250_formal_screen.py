#!/usr/bin/env python3
"""Render a small five-task screen of the immutable final-fit round-1000 snapshot."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SNAPSHOT = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "layerwise_unified_finalfit_seed42_20260801_snapshot"
)
SOURCE_JOBS = {
    "layerwise-final-eval-t1-t6-seed42": "layerwise-checkpoint1000-screen-t1-t6-seed42",
    "layerwise-final-eval-t2-seed42": "layerwise-checkpoint1000-screen-t2-seed42",
    "layerwise-final-eval-t3-seed42": "layerwise-checkpoint1000-screen-t3-seed42",
    "layerwise-final-eval-t5-seed42": "layerwise-checkpoint1000-screen-t5-seed42",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of source block, found {text.count(old)}")
    return text.replace(old, new)


def screen_command(command: str) -> str:
    receipt_block = '''export RECEIPT=/mnt/amulet/studentmodeling/layerwise_world_model_v1/final_fit/seed42.json
mkdir -p "$$HF_HUB_CACHE" "$$(dirname "$$RECURRENT_EVAL")" "$$AMLT_OUTPUT_DIR/eval/$$EVAL_GROUP"
test -f "$$RAW_EVAL"; test -f "$$RECEIPT"
export RUN_DIR="$$(python - "$$RECEIPT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='final_fit_complete' and x['stage']=='layerwise_unified_final_fit',x
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,5120],x
assert x['real_examples']==259845 and x['one_epoch_exact'] is True,x
print(x['run_dir'])
PY
)"
test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$$RUN_DIR/recurrent_state.pt"'''
    snapshot_block = f'''export RUN_DIR={SNAPSHOT}
export CHECKPOINT="$$RUN_DIR/checkpoint-round-1000"
mkdir -p "$$HF_HUB_CACHE" "$$(dirname "$$RECURRENT_EVAL")" "$$AMLT_OUTPUT_DIR/eval/$$EVAL_GROUP"
test -f "$$RAW_EVAL"
test -f "$$CHECKPOINT/adapter/adapter_config.json"
test -f "$$CHECKPOINT/adapter/adapter_model.safetensors"
test -f "$$CHECKPOINT/recurrent_state.pt"
test -f "$$CHECKPOINT/checkpoint_completion.json"
python - "$$CHECKPOINT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); receipt=json.load(open(root/'checkpoint_completion.json',encoding='utf-8'))
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): value.update(chunk)
    return value.hexdigest()
assert receipt.get('status')=='complete' and receipt.get('round')==1000,receipt
assert receipt.get('world_size')==16,receipt
assert receipt['sha256']['adapter_model']==digest(root/'adapter/adapter_model.safetensors')
assert receipt['sha256']['recurrent_state']==digest(root/'recurrent_state.pt')
PY'''
    command = replace_once(command, receipt_block, snapshot_block)
    command = command.replace('"$$RUN_DIR/final_adapter"', '"$$CHECKPOINT/adapter"')
    command = command.replace('"$$RUN_DIR/recurrent_state.pt"', '"$$CHECKPOINT/recurrent_state.pt"')
    command = replace_once(
        command,
        "--state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 128 \\",
        "--state-token-alias '<|vision_pad|>' --max-samples-per-group 25 \\\n        --max-length 1536 --max-new-tokens 128 \\",
    )
    command = command.replace(
        "stage=layerwise_unified_final_eval",
        "stage=layerwise_finalfit_checkpoint1000_formal_screen",
    )
    if "final_fit/seed42.json" in command or "final_adapter" in command:
        raise ValueError("Final-fit artifact reference survived checkpoint rendering")
    return command


def render(base: dict) -> dict:
    value = copy.deepcopy(base)
    value["description"] = (
        "Diagnostic-only five-task performance screen of the immutable layerwise "
        "unified final-fit round-1000 snapshot"
    )
    by_name = {str(job.get("name")): job for job in value.get("jobs") or []}
    jobs = []
    for source_name, target_name in SOURCE_JOBS.items():
        source = by_name.get(source_name)
        if source is None:
            raise ValueError(f"Missing source job: {source_name}")
        job = copy.deepcopy(source)
        job["name"] = target_name
        job["command"] = screen_command(str(job["command"]))
        submit = job.setdefault("submit_args", {})
        submit["max_attempts"] = 2
        submit["max_run_duration_seconds"] = 172800
        jobs.append(job)
    value["jobs"] = jobs
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output} jobs={len(value['jobs'])}")


if __name__ == "__main__":
    main()
