#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "TASKONLY_QWEN_ONEEPOCH_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${TASKONLY_TASK:?}" "${TASKONLY_ROWS:?}" "${TASKONLY_DATA_SHA256:?}" "${TASKONLY_RUN_TAG:?}"
case "$TASKONLY_TASK" in
  predict_correctness)
    expected_rows=110525
    expected_sha=a98d0f24d47fa0ef1a06fed7946935f6f569ef7eced8a35e2f2289f02ec76046
    expected_tag=t1_correctness ;;
  predict_response)
    expected_rows=55569
    expected_sha=e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e
    expected_tag=t2_response ;;
  predict_prospective_error)
    expected_rows=78600
    expected_sha=240dc2f1a02bbb446e4d73b0301ee4da7f6716c088746f7ec17bf82934c2aa1e
    expected_tag=t4_prospective ;;
  *) echo "Unsupported TASKONLY_TASK=$TASKONLY_TASK" >&2; exit 2 ;;
esac
test "$TASKONLY_ROWS" = "$expected_rows"
test "$TASKONLY_DATA_SHA256" = "$expected_sha"
test "$TASKONLY_RUN_TAG" = "$expected_tag"

export PATH="/tmp/studentmodeling-taskonly-qwen-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_MODEL=/tmp/Qwen3-32B
export TRAIN_DATA="/mnt/amulet/studentmodeling/taskonly_qwen_public_t1_t5_v1/${TASKONLY_TASK}.jsonl"
export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/taskonly_qwen_${TASKONLY_RUN_TAG}_oneepoch_seed42_v1"

test "$(sha256sum scripts/train_c1_lora.py | cut -d' ' -f1)" = 7c8b8b0a32ecf5695b7200f2c5b64fb0989598b88dff1da59da7cbfe7752d591
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(sha256sum "$TRAIN_DATA" | cut -d' ' -f1)" = "$TASKONLY_DATA_SHA256"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test ! -e "$RUN_DIR/completion.json"
mkdir -p "$RUN_DIR"

python -u scripts/train_c1_lora.py \
  --model "$LOCAL_MODEL" --train-jsonl "$TRAIN_DATA" --output-dir "$RUN_DIR" \
  --task-filter "$TASKONLY_TASK" --max-length 1536 --max-train-samples "$TASKONLY_ROWS" \
  --max-steps -1 --num-train-epochs 1 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --lora-rank 8 --lora-alpha 16 --load-4bit \
  --state-token-alias '<|vision_pad|>' --supervise-eos \
  --logging-steps 20 --save-final-only --seed 42

python - <<'PY'
import hashlib, json, os, pathlib
root=pathlib.Path(os.environ['RUN_DIR']); cfg=json.load(open(root/'run_config.json'))
task=os.environ['TASKONLY_TASK']; rows=int(os.environ['TASKONLY_ROWS'])
assert cfg['task_filter']==task and cfg['num_train_epochs']==1.0,cfg
assert cfg['max_train_samples']==rows and cfg['max_steps']==-1,cfg
assert cfg['state_token_alias']=='<|vision_pad|>' and cfg['save_final_only'] is True,cfg
path=root/'final_adapter/adapter_model.safetensors'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
receipt={'status':'complete','task':task,'train_rows':rows,'epochs':1,'global_batch':4,'adapter_sha256':digest}
(root/'completion.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
PY

cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/run_config.json" "$RUN_DIR/completion.json" "$AMLT_OUTPUT_DIR/"
