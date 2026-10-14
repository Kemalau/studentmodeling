#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "NEURAL_RESPONSE_SPECIALIST_FAILED line=%s rc=%s command=%s\n" "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR

: "${BASELINE_TASK:?}" "${BASELINE_PUBLIC_TASK:?}" "${BASELINE_TRAIN_DATA:?}"
: "${BASELINE_TRAIN_SHA256:?}" "${BASELINE_TRAIN_ROWS:?}" "${BASELINE_TEST_ROWS:?}"

case "$BASELINE_PUBLIC_TASK:$BASELINE_TASK" in
  T2:predict_response)
    expected_train_sha=e44efac7f183fdb7b5fa4dfe2075e5b10efaa5abf689dc33e0fd718256dfc62e
    expected_train_rows=55569
    expected_test_rows=3693
    expected_metric=math_equivalence ;;
  T4:predict_prospective_error)
    expected_train_sha=240dc2f1a02bbb446e4d73b0301ee4da7f6716c088746f7ec17bf82934c2aa1e
    expected_train_rows=78600
    expected_test_rows=4600
    expected_metric=joint_math_equivalence ;;
  *)
    echo "Unsupported baseline binding: $BASELINE_PUBLIC_TASK:$BASELINE_TASK" >&2
    exit 2 ;;
esac

test "$BASELINE_TRAIN_SHA256" = "$expected_train_sha"
test "$BASELINE_TRAIN_ROWS" = "$expected_train_rows"
test "$BASELINE_TEST_ROWS" = "$expected_test_rows"

export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TEST_DATA=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export OUTPUT="$AMLT_OUTPUT_DIR/eval/$BASELINE_PUBLIC_TASK"
mkdir -p "$OUTPUT"

test -s "$BASELINE_TRAIN_DATA"
test -s "$TEST_DATA"
test "$(sha256sum "$BASELINE_TRAIN_DATA" | cut -d' ' -f1)" = "$expected_train_sha"
test "$(sha256sum "$TEST_DATA" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8

python -u scripts/train_neural_response_specialist.py \
  --task "$BASELINE_TASK" \
  --train-jsonl "$BASELINE_TRAIN_DATA" \
  --test-jsonl "$TEST_DATA" \
  --output-dir "$OUTPUT" \
  --epochs 8 --batch-size 256 --eval-batch-size 512 \
  --embedding-dim 64 --hidden-dim 128 --dropout 0.2 \
  --learning-rate 1e-3 --weight-decay 1e-4 \
  --max-history 32 --seed 42 --device cuda

python - "$OUTPUT/metrics.json" "$OUTPUT/predictions.jsonl" "$OUTPUT/completion.json" <<'PY'
import hashlib,json,os,pathlib,sys
metrics_path,predictions_path,completion_path=map(pathlib.Path,sys.argv[1:])
value=json.load(metrics_path.open(encoding="utf-8"))
records=[json.loads(line) for line in predictions_path.open(encoding="utf-8") if line.strip()]
expected_test=int(os.environ["BASELINE_TEST_ROWS"])
expected_train=int(os.environ["BASELINE_TRAIN_ROWS"])
task=os.environ["BASELINE_TASK"]
public=os.environ["BASELINE_PUBLIC_TASK"]
metric="math_equivalence" if task=="predict_response" else "joint_math_equivalence"
assert value["status"]=="complete",value
assert value["task"]==task and value["public_task"]==public,value
assert value["support"]=={"train":expected_train-int(value["support"]["valid"]),"valid":int(value["support"]["valid"]),"test":expected_test},value
assert value["support"]["train"]+value["support"]["valid"]==expected_train,value
assert value["metrics"]["overall"]["n"]==expected_test,value
assert metric in value["metrics"]["overall"],value
assert len(records)==expected_test and len({row["example_id"] for row in records})==expected_test
assert {row["task"] for row in records}=={task}
assert value["artifacts"]["predictions_sha256"]==hashlib.sha256(predictions_path.read_bytes()).hexdigest()
receipt={
  "stage":"formal_neural_response_specialist_v1",
  "status":"complete",
  "public_task":public,
  "internal_task":task,
  "train_rows":expected_train,
  "test_rows":expected_test,
  "primary_metric":metric,
  "primary_value":float(value["metrics"]["overall"][metric]),
  "model_sha256":value["artifacts"]["model_sha256"],
  "predictions_sha256":value["artifacts"]["predictions_sha256"],
  "train_data_sha256":os.environ["BASELINE_TRAIN_SHA256"],
  "test_data_sha256":"bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8",
}
completion_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(receipt,sort_keys=True))
PY
