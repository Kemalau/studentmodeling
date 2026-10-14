#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/taskonly_qwen_formal_eval_20260813"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/summary.lock"; flock -n 9 || exit 0
echo $$ > "$state/summary.active.pid"; trap 'rm -f "$state/summary.active.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/summary.heartbeat"
  for task in T1 T2 T3 T4 T5; do
    [[ ! -f "$state/$task/needs_attention" ]] || {
      touch "$state/needs_attention"; exit 1;
    }
  done
  ready=1
  for task in T1 T2 T3 T4 T5; do [[ -f "$state/$task/complete" ]] || ready=0; done
  (( ready == 1 )) && break
  sleep 60
done

unified_t3_v2="$project/results/formal_specialist_v2/unrestricted_qwen_t3_v2"
while [[ ! -f "$unified_t3_v2/complete" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/summary.heartbeat"
  [[ ! -f "$unified_t3_v2/needs_attention" ]] || {
    touch "$state/needs_attention"; exit 1;
  }
  sleep 60
done

mapfile -t predictions < <(find "$state" -type f -path '*/download/*/eval/T?/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 5 ]] || { touch "$state/needs_attention"; exit 1; }
"$python_bin" scripts/audit_taskonly_target_exposure.py \
  --unified-jsonl data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl \
  --unified-manifest manifests/unified_recurrent_full259845_u4_v1_final_fit.json \
  --taskonly-root data/processed/taskonly_qwen_public_t1_t5_v1 \
  --taskonly-manifest data/processed/taskonly_qwen_public_t1_t5_v1/manifest.json \
  --output "$state/target_exposure_audit.json" \
  > "$state/target_exposure_audit.log" 2>&1 \
  || { touch "$state/needs_attention"; exit 1; }
"$python_bin" scripts/summarize_taskonly_qwen_formal.py \
  --predictions "${predictions[@]}" \
  --t3-semantic-metrics "$state/T3/t3_semantic/metrics.json" \
  --output-dir "$state/summary" > "$state/summary.log" 2>&1 \
  || { touch "$state/needs_attention"; exit 1; }
unified_classification="$(find results/formal_specialist_v1/qwen_amlt -type f \
  -path '*/adapter_classification/predictions_redacted.jsonl' | head -n 1)"
unified_t3="$(find results/formal_specialist_v2/unrestricted_qwen_t3_v2/adapter_diagnose_misconception \
  -type f -name predictions_redacted.jsonl | head -n 1)"
unified_generation_experiment=studentmodeling-qwen3-32b-formal-generation-recovery1-4a100-20260720
unified_generation_job=qwen3-32b-formal-generation-v1
unified_generation_download="$state/unified_generation_predictions"
while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/summary.heartbeat"
  unified_t2="$(find "$unified_generation_download" -type f \
    -path '*/eval/adapter_predict_response/predictions_redacted.jsonl' 2>/dev/null | head -n 1)"
  unified_t4="$(find "$unified_generation_download" -type f \
    -path '*/eval/adapter_predict_prospective_error/predictions_redacted.jsonl' 2>/dev/null | head -n 1)"
  [[ -s "$unified_t2" && -s "$unified_t4" ]] && break
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "$STUDENTMODELING_FEEDS_AUTH_HEALTH" == unhealthy ]]; then
    sleep 60
    continue
  fi
  mkdir -p "$unified_generation_download"
  timeout --foreground 600 amlt results download \
    "$unified_generation_experiment" ":$unified_generation_job" \
    --include 'predictions_redacted.jsonl' --output "$unified_generation_download" \
    </dev/null \
    >> "$state/unified_generation_download.log" 2>&1 || {
      sleep 60
      continue
    }
done
[[ -s "$unified_classification" && -s "$unified_t2" && -s "$unified_t3" \
   && -s "$unified_t4" ]] \
  || { touch "$state/needs_attention"; exit 1; }
mkdir -p "$state/paired"
for task in T1 T2 T3 T4 T5; do
  case "$task" in
    T1|T5) unified="$unified_classification" ;;
    T2) unified="$unified_t2" ;;
    T3) unified="$unified_t3" ;;
    T4) unified="$unified_t4" ;;
  esac
  taskonly="$(find "$state/$task/download" -type f -path "*/eval/$task/predictions.jsonl")"
  [[ -s "$taskonly" ]] || { touch "$state/needs_attention"; exit 1; }
  if [[ "$task" == T3 ]]; then
    taskonly="$state/$task/t3_semantic/predictions_redacted.jsonl"
  fi
  # The archived T2/T4 unified artifacts intentionally retain hashes and
  # aggregate metrics only.  They cannot support per-example paired scoring.
  # Never interpret two missing raw-output fields as a correct prediction.
  if [[ "$task" == T2 || "$task" == T4 ]]; then
    continue
  fi
  "$python_bin" scripts/compare_taskonly_unified_paired.py \
    --task "$task" --unified "$unified" --taskonly "$taskonly" \
    --output "$state/paired/$task.json" \
    > "$state/paired/$task.log" 2>&1 \
    || { touch "$state/needs_attention"; exit 1; }
done
"$python_bin" - "$state/paired" \
  results/formal_specialist_v2/formal_evaluation_matrix.json \
  "$state/summary/summary.json" <<'PY' \
  || { touch "$state/needs_attention"; exit 1; }
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
tasks = ["T1", "T2", "T3", "T4", "T5"]
values = {task: json.loads((root / f"{task}.json").read_text()) for task in ("T1", "T3", "T5")}
matrix = json.loads(pathlib.Path(sys.argv[2]).read_text())
taskonly = json.loads(pathlib.Path(sys.argv[3]).read_text())["cells"]
unified = next(row for row in matrix["rows"] if row["model"] == "Unified Qwen3-32B QLoRA")
for task in ("T2", "T4"):
    values[task] = {
        "public_task": task,
        "internal_task": {"T2": "predict_response", "T4": "predict_prospective_error"}[task],
        "n": int(taskonly[task]["n"]),
        "paired_available": False,
        "unified_minus_taskonly": float(unified[task]["value"]) - float(taskonly[task]["primary"]),
        "reason": "Archived unified predictions are redacted and lack raw_output/target; only aggregate comparison is valid.",
    }
assert all(values[task]["public_task"] == task for task in tasks), values
(root / "complete.json").write_text(json.dumps({
    "stage": "taskonly_unified_paired_full_v1",
    "status": "complete",
    "tasks": values,
}, indent=2, sort_keys=True) + "\n")
PY
"$python_bin" scripts/build_formal_evaluation_matrix.py \
  --classification-root results/formal_specialist_v1/qwen_amlt \
  --generation-root results/formal_specialist_v1/qwen_amlt \
  --t3-root results/formal_specialist_v2/unrestricted_qwen_t3_v2 \
  --specialist-root results/formal_specialist_v1 \
  --taskonly-summary "$state/summary/summary.json" \
  --paired-summary "$state/paired/complete.json" \
  --output-dir "$state/formal_specialist_v3" \
  > "$state/formal_matrix.log" 2>&1 \
  || { touch "$state/needs_attention"; exit 1; }
touch "$state/complete"
date -u +%Y-%m-%dT%H:%M:%SZ > "$state/summary.heartbeat"
