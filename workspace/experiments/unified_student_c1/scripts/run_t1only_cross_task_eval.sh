#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "T1ONLY_CROSS_EVAL_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${EVAL_PUBLIC_TASK:?}" "${EVAL_TASK:?}" "${EVAL_EXPECTED_ROWS:?}"
case "$EVAL_PUBLIC_TASK:$EVAL_TASK" in
  T2:predict_response) expected=3693 ;;
  T3:diagnose_misconception) expected=1494 ;;
  T4:predict_prospective_error) expected=4600 ;;
  T5:predict_error_persistence) expected=976 ;;
  *) echo "Unsupported evaluation binding: $EVAL_PUBLIC_TASK:$EVAL_TASK" >&2; exit 2 ;;
esac
test "$EVAL_EXPECTED_ROWS" = "$expected"

export PATH="/tmp/studentmodeling-t1only-cross-eval-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export EVAL_DATA=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export ADAPTER_ROOT=/mnt/amulet/studentmodeling/checkpoints/taskonly_qwen_t1_correctness_oneepoch_seed42_v1
export ADAPTER="$ADAPTER_ROOT/final_adapter"
export OUTPUT="$AMLT_OUTPUT_DIR/eval/$EVAL_PUBLIC_TASK"

test "$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -s "$ADAPTER_ROOT/completion.json"
test -s "$ADAPTER/adapter_model.safetensors"
python - "$ADAPTER_ROOT/completion.json" "$ADAPTER/adapter_model.safetensors" <<'PY'
import hashlib, json, pathlib, sys
receipt = json.load(open(sys.argv[1]))
adapter = pathlib.Path(sys.argv[2])
assert receipt == {
    "adapter_sha256": "4e415fb657898ff58fbc58c133069042536db692eda9e74ca7c18d226a56729b",
    "epochs": 1,
    "global_batch": 4,
    "optimizer_steps": 27632,
    "resumable": True,
    "status": "complete",
    "task": "predict_correctness",
    "train_rows": 110525,
}, receipt
assert hashlib.sha256(adapter.read_bytes()).hexdigest() == receipt["adapter_sha256"]
PY

python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
mkdir -p "$OUTPUT"
python -u scripts/evaluate_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$ADAPTER" \
  --eval-jsonl "$EVAL_DATA" --output-dir "$OUTPUT" \
  --tasks "$EVAL_TASK" --max-length 1536 --max-new-tokens 160 \
  --classification-batch-size 8 --generation-batch-size 2 \
  --state-token-alias '<|vision_pad|>' --seed 42 --save-unredacted-predictions

python - "$OUTPUT/metrics.json" "$OUTPUT/predictions.jsonl" "$OUTPUT/completion.json" <<'PY'
import hashlib, json, os, pathlib, sys
metrics_path, predictions_path, output_path = map(pathlib.Path, sys.argv[1:])
metrics = json.load(metrics_path.open())
rows = [json.loads(line) for line in predictions_path.open() if line.strip()]
expected = int(os.environ["EVAL_EXPECTED_ROWS"])
task = os.environ["EVAL_TASK"]
assert metrics["sampled_rows"] == expected and metrics["unsharded_filtered_rows"] == expected
assert metrics["artifact_sha256"]["adapter_model"] == "4e415fb657898ff58fbc58c133069042536db692eda9e74ca7c18d226a56729b"
assert len(rows) == expected and len({row["example_id"] for row in rows}) == expected
assert {row["task"] for row in rows} == {task}
receipt = {
    "stage": "t1only_cross_task_eval_v1",
    "status": "complete",
    "training_task": "T1",
    "evaluation_task": os.environ["EVAL_PUBLIC_TASK"],
    "evaluation_rows": expected,
    "adapter_sha256": metrics["artifact_sha256"]["adapter_model"],
    "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    "semantic_judge_pending": task == "diagnose_misconception",
}
output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY
