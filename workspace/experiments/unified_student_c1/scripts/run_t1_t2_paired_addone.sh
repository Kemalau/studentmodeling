#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "T1_T2_PAIRED_ADDONE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export PATH="/tmp/studentmodeling-t1t2-addone-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export DATA_ROOT=/mnt/amulet/studentmodeling/taskonly_qwen_public_t1_t5_v1
export T1_DATA="$DATA_ROOT/predict_correctness.jsonl"
export T2_DATA="$DATA_ROOT/predict_response.jsonl"
export TRAIN_DATA=/tmp/t1_t2_paired_addone_seed42.jsonl
export DATA_RECEIPT=/tmp/t1_t2_paired_addone_seed42.receipt.json
export EVAL_DATA=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/qwen3_32b_t1_t2_paired_addone_oneepoch_seed42_v1
export EVAL_DIR="$AMLT_OUTPUT_DIR/eval_t1"

test "$(sha256sum "$T1_DATA" | cut -d' ' -f1)" = a98d0f24d47fa0ef1a06fed7946935f6f569ef7eced8a35e2f2289f02ec76046
test "$(sha256sum "$T2_DATA" | cut -d' ' -f1)" = e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e
test "$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(df --output=avail /tmp | tail -1)" -ge 125829120

python scripts/build_t1_t2_paired_addone.py \
  --t1 "$T1_DATA" --t2 "$T2_DATA" --output "$TRAIN_DATA" --receipt "$DATA_RECEIPT"
python - "$DATA_RECEIPT" <<'PY'
import json, sys
x=json.load(open(sys.argv[1])); assert x["rows"]==162765 and x["t1_rows"]==110525 and x["paired_t2_rows"]==52240,x
assert x["t1_exposure_matches_taskonly"] is True and x["total_update_matched"] is False,x
PY

python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -s "$LOCAL_MODEL/config.json"
mkdir -p "$RUN_DIR" "$EVAL_DIR" "$AMLT_OUTPUT_DIR/model"

if [[ ! -s "$RUN_DIR/completion.json" ]]; then
  # The initial attempts were interrupted by the Feeds wall-time/platform signal.
  # Resume the latest complete Trainer state (adapter, optimizer, scheduler, and
  # RNG) instead of replaying completed updates. Fail closed on horizon drift or
  # a partial checkpoint. Feeds-LLM has preempted this workload before the next
  # 5k boundary.  Recovery wave 10 was then preempted again before reaching the
  # next 1k boundary, so save every 200 steps to keep future replay bounded.
  SOURCE_RESUME_CHECKPOINT="$(python - "$RUN_DIR" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
required=('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
          'rng_state.pth','trainer_state.json','training_args.bin')
candidates=[]
for path in root.glob('checkpoint-*'):
    try: step=int(path.name.split('-')[-1])
    except ValueError: continue
    if all((path/name).is_file() and (path/name).stat().st_size > 0 for name in required):
        state=json.load(open(path/'trainer_state.json', encoding='utf-8'))
        if int(state['global_step']) != step or int(state['max_steps']) != 40692:
            continue
        if int(state['num_train_epochs']) != 1 or not 0 < step < 40692:
            continue
        candidates.append((step,path))
if not candidates:
    raise SystemExit('no complete compatible checkpoint found')
print(max(candidates)[1])
PY
)"
  RESUME_STEP="$(python - "$SOURCE_RESUME_CHECKPOINT/trainer_state.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1],encoding='utf-8'))['global_step']))
PY
)"
  test "$RESUME_STEP" -ge 20000
  RESUME_CHECKPOINT="/tmp/t1_t2_resume_checkpoint-$RESUME_STEP"
  CADENCE_RECEIPT="/tmp/t1_t2_resume_cadence-$RESUME_STEP.json"
  python scripts/stage_t1_t2_resume_checkpoint.py \
    --source "$SOURCE_RESUME_CHECKPOINT" --destination "$RESUME_CHECKPOINT" \
    --save-steps 200 --max-steps 40692 --receipt "$CADENCE_RECEIPT"
  export RESUME_CHECKPOINT RESUME_STEP SOURCE_RESUME_CHECKPOINT

  python -u scripts/train_c1_lora.py \
    --model "$LOCAL_MODEL" --train-jsonl "$TRAIN_DATA" --output-dir "$RUN_DIR" \
    --max-length 1536 --max-train-samples 162765 --max-steps -1 --num-train-epochs 1 \
    --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 --lora-rank 8 --lora-alpha 16 --load-4bit \
    --state-token-alias '<|vision_pad|>' --supervise-eos \
    --logging-steps 20 --save-steps 200 --save-total-limit 2 --seed 42 \
    --resume-from-checkpoint "$RESUME_CHECKPOINT"

  python - "$RUN_DIR" "$DATA_RECEIPT" "$RESUME_STEP" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); data=json.load(open(sys.argv[2])); resume_step=int(sys.argv[3]); cfg=json.load(open(root/'run_config.json'))
adapter=root/'final_adapter/adapter_model.safetensors'
assert cfg['max_train_samples']==162765 and cfg['num_train_epochs']==1.0 and cfg['seed']==42,cfg
assert cfg['resume_from_checkpoint']==f'/tmp/t1_t2_resume_checkpoint-{resume_step}',cfg
receipt={'stage':'t1_t2_paired_addone_train_v1','status':'complete','seed':42,'train_rows':162765,
         't1_rows':110525,'paired_t2_rows':52240,'optimizer_steps':40692,'global_batch':4,
         'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'data_sha256':data['output_sha256'],
         'diagnostic_only_total_update_not_matched':True,'resumed_from_step':resume_step}
(root/'completion.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
else
  # A prior wave may have finished training but been interrupted during formal
  # evaluation. Validate and reuse that immutable final adapter.
  python - "$RUN_DIR" "$DATA_RECEIPT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); data=json.load(open(sys.argv[2])); receipt=json.load(open(root/'completion.json'))
adapter=root/'final_adapter/adapter_model.safetensors'
assert receipt['stage']=='t1_t2_paired_addone_train_v1' and receipt['status']=='complete',receipt
assert receipt['seed']==42 and receipt['train_rows']==162765 and receipt['optimizer_steps']==40692,receipt
assert receipt['data_sha256']==data['output_sha256'],receipt
assert hashlib.sha256(adapter.read_bytes()).hexdigest()==receipt['adapter_sha256'],receipt
PY
fi

cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/run_config.json" "$RUN_DIR/completion.json" "$DATA_RECEIPT" "$AMLT_OUTPUT_DIR/"
if [[ -n "${CADENCE_RECEIPT:-}" && -s "$CADENCE_RECEIPT" ]]; then
  cp "$CADENCE_RECEIPT" "$AMLT_OUTPUT_DIR/"
fi
python -u scripts/evaluate_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --eval-jsonl "$EVAL_DATA" --output-dir "$EVAL_DIR" \
  --tasks predict_correctness --max-length 1536 --max-new-tokens 16 \
  --classification-batch-size 8 --generation-batch-size 2 \
  --state-token-alias '<|vision_pad|>' --seed 42 --save-unredacted-predictions

python - "$RUN_DIR/completion.json" "$EVAL_DIR/metrics.json" "$EVAL_DIR/predictions.jsonl" "$EVAL_DIR/completion.json" <<'PY'
import hashlib,json,pathlib,sys
train_path,metrics_path,pred_path,out_path=map(pathlib.Path,sys.argv[1:])
train=json.load(open(train_path)); metrics=json.load(open(metrics_path)); rows=[json.loads(x) for x in open(pred_path) if x.strip()]
assert len(rows)==10913 and len({r['example_id'] for r in rows})==10913 and {r['task'] for r in rows}=={'predict_correctness'}
assert metrics['artifact_sha256']['adapter_model']==train['adapter_sha256']
out={'stage':'t1_t2_paired_addone_t1_eval_v1','status':'complete','evaluation_rows':10913,
     'adapter_sha256':train['adapter_sha256'],'predictions_sha256':hashlib.sha256(pred_path.read_bytes()).hexdigest()}
out_path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
