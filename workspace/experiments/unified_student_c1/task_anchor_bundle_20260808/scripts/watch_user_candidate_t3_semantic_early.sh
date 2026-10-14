#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/user_candidate_parallel_eval_20260804"
semantic_dir="$state_dir/t3_semantic"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"

mkdir -p "$semantic_dir/full1494"
cd "$project_dir"
exec 11>"$semantic_dir/judge.lock"
flock -x 11
[[ ! -f "$semantic_dir/complete" ]] || exit 0

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$semantic_dir/early_judge.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$semantic_dir/early_judge.controller.log"; }
fail(){ log "ERROR: $*"; touch "$semantic_dir/early_judge.needs_attention"; exit 1; }

test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "frozen T3 eval panel drift"
readarray -t predictions < <(find "$state_dir/partial_pass_download" -type f -path '*/eval/t3/predictions.jsonl' | sort)
[[ ${#predictions[@]} -eq 1 ]] || fail "expected one preserved full T3 prediction file"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"

heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!

attempt=0
while [[ ! -f "$semantic_dir/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$extra" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$semantic_dir/full1494" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$semantic_dir/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$semantic_dir/early_judge.full.log" 2>&1; then
    if "$python_bin" scripts/validate_t3_semantic_result.py \
        --metrics "$semantic_dir/full1494/metrics.json" \
        --output "$semantic_dir/t3_semantic_gate.json" \
        >> "$semantic_dir/early_judge.controller.log" 2>&1; then
      rm -f "$semantic_dir/scientific_gate_failed"
      touch "$semantic_dir/scientific_gate_pass"
    else
      rc=$?
      (( rc == 2 )) || fail "candidate T3 semantic validator failed rc=$rc"
      rm -f "$semantic_dir/scientific_gate_pass"
      touch "$semantic_dir/scientific_gate_failed"
    fi
    touch "$semantic_dir/complete"
    log "candidate full-1494 T3 semantic judgment completed ahead of T5"
    exit 0
  fi
  (( attempt < 30 )) || fail "candidate early T3 semantic judge exhausted"
  sleep 120
done
