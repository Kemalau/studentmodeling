#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "TASKONLY_QWEN_FORMAL_EVAL_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${TASKONLY_PUBLIC_TASK:?}" "${TASKONLY_TASK:?}" "${TASKONLY_ADAPTER_TAG:?}"
: "${TASKONLY_EXPECTED_ROWS:?}" "${TASKONLY_TRAIN_ROWS:?}"

case "$TASKONLY_PUBLIC_TASK:$TASKONLY_TASK" in
  T1:predict_correctness)
    expected_rows=10913; expected_train_rows=110525 ;;
  T2:predict_response)
    expected_rows=3693; expected_train_rows=55569 ;;
  T3:diagnose_misconception)
    expected_rows=1494; expected_train_rows=6776 ;;
  T4:predict_prospective_error)
    expected_rows=4600; expected_train_rows=78600 ;;
  T5:predict_error_persistence)
    expected_rows=976; expected_train_rows=8375 ;;
  *)
    echo "Unsupported task binding: $TASKONLY_PUBLIC_TASK:$TASKONLY_TASK" >&2
    exit 2 ;;
esac
test "$TASKONLY_EXPECTED_ROWS" = "$expected_rows"
test "$TASKONLY_TRAIN_ROWS" = "$expected_train_rows"

export PATH="/tmp/studentmodeling-taskonly-formal-eval-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export EVAL_DATA=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export ADAPTER_ROOT="/mnt/amulet/studentmodeling/checkpoints/${TASKONLY_ADAPTER_TAG}"
export ADAPTER="$ADAPTER_ROOT/final_adapter"
export OUTPUT="$AMLT_OUTPUT_DIR/eval/$TASKONLY_PUBLIC_TASK"

test "$(sha256sum scripts/evaluate_formal_protocols.py | cut -d' ' -f1)" = b347ed4d2228725a325975c0ffab162ca32a2e0252583de841c00b3ab1e196de
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(sha256sum requirements-feeds.txt | cut -d' ' -f1)" = fcbe7c063774982c0f0c87a6be96c203352e00ff4ac653d949a8eb97347a3fb1
test "$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -s "$ADAPTER_ROOT/completion.json"
test -s "$ADAPTER/adapter_model.safetensors"

python - "$ADAPTER_ROOT/completion.json" "$ADAPTER/adapter_model.safetensors" <<'PY'
import hashlib, json, os, pathlib, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
adapter = pathlib.Path(sys.argv[2])
digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
assert receipt["status"] == "complete", receipt
assert receipt["task"] == os.environ["TASKONLY_TASK"], receipt
assert int(receipt["train_rows"]) == int(os.environ["TASKONLY_TRAIN_ROWS"]), receipt
assert int(receipt["epochs"]) == 1, receipt
assert receipt["adapter_sha256"] == digest, (receipt, digest)
PY

mkdir -p "$HF_HUB_CACHE" "$OUTPUT"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -s "$LOCAL_MODEL/config.json"

python -u scripts/evaluate_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$ADAPTER" \
  --eval-jsonl "$EVAL_DATA" --output-dir "$OUTPUT" \
  --tasks "$TASKONLY_TASK" --max-length 1536 --max-new-tokens 160 \
  --classification-batch-size 8 --generation-batch-size 2 \
  --state-token-alias '<|vision_pad|>' --seed 42 \
  --save-unredacted-predictions

python - "$OUTPUT/metrics.json" "$OUTPUT/predictions.jsonl" \
  "$ADAPTER_ROOT/completion.json" "$OUTPUT/completion.json" <<'PY'
import hashlib, json, os, pathlib, sys
metrics_path, predictions_path, train_receipt_path, output_path = map(pathlib.Path, sys.argv[1:])
metrics = json.load(metrics_path.open(encoding="utf-8"))
train = json.load(train_receipt_path.open(encoding="utf-8"))
records = [json.loads(line) for line in predictions_path.open(encoding="utf-8") if line.strip()]
expected = int(os.environ["TASKONLY_EXPECTED_ROWS"])
task = os.environ["TASKONLY_TASK"]
assert metrics["sampled_rows"] == expected, metrics
assert metrics["unsharded_filtered_rows"] == expected, metrics
assert metrics["artifact_sha256"]["evaluation_input"] == "bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8", metrics
assert metrics["artifact_sha256"]["adapter_model"] == train["adapter_sha256"], (metrics, train)
assert len(records) == expected and len({row["example_id"] for row in records}) == expected
assert {row["task"] for row in records} == {task}
receipt = {
    "stage": "taskonly_qwen_formal_eval_v1",
    "status": "complete",
    "public_task": os.environ["TASKONLY_PUBLIC_TASK"],
    "internal_task": task,
    "evaluation_rows": expected,
    "train_rows": int(os.environ["TASKONLY_TRAIN_ROWS"]),
    "adapter_sha256": train["adapter_sha256"],
    "evaluation_input_sha256": metrics["artifact_sha256"]["evaluation_input"],
    "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    "semantic_judge_pending": task == "diagnose_misconception",
}
output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY
