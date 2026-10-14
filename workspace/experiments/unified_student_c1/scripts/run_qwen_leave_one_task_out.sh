#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "QWEN_LOO_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${EXCLUDED_PUBLIC_TASK:?}" "${EXCLUDED_TASK:?}" "${EXPECTED_TRAIN_ROWS:?}" "${EXPECTED_OPTIMIZER_STEPS:?}"
case "$EXCLUDED_PUBLIC_TASK:$EXCLUDED_TASK" in
  T1:predict_correctness) expected_rows=149320; expected_steps=37330 ;;
  T2:predict_response) expected_rows=204276; expected_steps=51069 ;;
  T3:diagnose_misconception) expected_rows=253069; expected_steps=63268 ;;
  T4:predict_prospective_error) expected_rows=181245; expected_steps=45312 ;;
  T5:predict_error_persistence) expected_rows=251470; expected_steps=62868 ;;
  *) echo "Unsupported exclusion: $EXCLUDED_PUBLIC_TASK:$EXCLUDED_TASK" >&2; exit 2 ;;
esac
test "$EXPECTED_TRAIN_ROWS" = "$expected_rows"
test "$EXPECTED_OPTIMIZER_STEPS" = "$expected_steps"

export PATH="/tmp/studentmodeling-loo-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export TRAIN_DATA=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export EVAL_DATA=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/qwen3_32b_loo_${EXCLUDED_PUBLIC_TASK,,}_oneepoch_seed42_v1"
export EVAL_DIR="$AMLT_OUTPUT_DIR/eval_all_t1_t5"

test "$(sha256sum "$TRAIN_DATA" | cut -d' ' -f1)" = aeabd3b4dfef3c4d92ac68583879c91448ef90f58304cedf32e470b586ca4542
test "$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(df --output=avail /tmp | tail -1)" -ge 125829120

(
  flock -x 9
  if [[ ! -f /tmp/Qwen3-32B.copy_complete ]]; then
    rm -rf "$LOCAL_MODEL"
    python scripts/copy_model_parallel.py \
      --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
      --destination "$LOCAL_MODEL" --workers 16
    touch /tmp/Qwen3-32B.copy_complete
  fi
) 9>/tmp/qwen3-32b-copy.lock
test -s "$LOCAL_MODEL/config.json"

rank="${OMPI_COMM_WORLD_RANK:-${RANK:-0}}"
world="${OMPI_COMM_WORLD_SIZE:-${WORLD_SIZE:-1}}"
test "$world" = 1
if [[ "$rank" = 0 ]]; then
  test ! -e "$RUN_DIR/completion.json"
  mkdir -p "$RUN_DIR"
fi

python -u scripts/train_c1_lora.py \
  --model "$LOCAL_MODEL" --train-jsonl "$TRAIN_DATA" --output-dir "$RUN_DIR" \
  --exclude-task "$EXCLUDED_TASK" --max-length 1536 \
  --max-train-samples "$EXPECTED_TRAIN_ROWS" --max-steps -1 --num-train-epochs 1 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --lora-rank 8 --lora-alpha 16 --load-4bit \
  --state-token-alias '<|vision_pad|>' --supervise-eos \
  --logging-steps 20 --save-steps 5000 --save-total-limit 2 --seed 42

if [[ "$rank" != 0 ]]; then
  exit 0
fi

python - "$RUN_DIR" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
config = json.loads((root / "run_config.json").read_text())
adapter = root / "final_adapter" / "adapter_model.safetensors"
assert config["exclude_task"] == os.environ["EXCLUDED_TASK"], config
assert config["task_filter"] is None and config["num_train_epochs"] == 1.0, config
assert config["max_train_samples"] == int(os.environ["EXPECTED_TRAIN_ROWS"]), config
assert config["seed"] == 42 and config["state_token_alias"] == "<|vision_pad|>", config
receipt = {
    "stage": "qwen_leave_one_task_out_train_v1",
    "status": "complete",
    "excluded_public_task": os.environ["EXCLUDED_PUBLIC_TASK"],
    "excluded_task": os.environ["EXCLUDED_TASK"],
    "train_rows": int(os.environ["EXPECTED_TRAIN_ROWS"]),
    "optimizer_steps": int(os.environ["EXPECTED_OPTIMIZER_STEPS"]),
    "epochs": 1,
    "world_size": 1,
    "global_batch": 4,
    "seed": 42,
    "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
}
(root / "completion.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY

mkdir -p "$AMLT_OUTPUT_DIR/model" "$EVAL_DIR"
cp -r "$RUN_DIR/final_adapter/." "$AMLT_OUTPUT_DIR/model/"
cp "$RUN_DIR/run_config.json" "$RUN_DIR/completion.json" "$AMLT_OUTPUT_DIR/"

python -u scripts/evaluate_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --eval-jsonl "$EVAL_DATA" --output-dir "$EVAL_DIR" \
  --tasks predict_correctness predict_response diagnose_misconception \
          predict_prospective_error predict_error_persistence \
  --max-length 1536 --max-new-tokens 160 \
  --classification-batch-size 8 --generation-batch-size 2 \
  --state-token-alias '<|vision_pad|>' --seed 42 --save-unredacted-predictions

python - "$RUN_DIR/completion.json" "$EVAL_DIR/metrics.json" "$EVAL_DIR/predictions.jsonl" "$EVAL_DIR/completion.json" <<'PY'
import hashlib, json, pathlib, sys
train_path, metrics_path, predictions_path, output_path = map(pathlib.Path, sys.argv[1:])
train = json.loads(train_path.read_text())
metrics = json.loads(metrics_path.read_text())
records = [json.loads(line) for line in predictions_path.open() if line.strip()]
expected = {
    "predict_correctness": 10913,
    "predict_response": 3693,
    "diagnose_misconception": 1494,
    "predict_prospective_error": 4600,
    "predict_error_persistence": 976,
}
observed = {task: 0 for task in expected}
for row in records:
    observed[row["task"]] += 1
assert observed == expected, observed
assert len(records) == 21676 and len({row["example_id"] for row in records}) == 21676
assert metrics["artifact_sha256"]["adapter_model"] == train["adapter_sha256"]
receipt = {
    "stage": "qwen_leave_one_task_out_formal_eval_v1",
    "status": "complete",
    "excluded_public_task": train["excluded_public_task"],
    "evaluation_rows": len(records),
    "evaluation_rows_by_task": observed,
    "adapter_sha256": train["adapter_sha256"],
    "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    "t3_semantic_judge_pending": True,
}
output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY
