#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
formal_dir="$pipeline/task_retention_promoted_final_fit/local_round1000_formal_screen"
state_dir="$pipeline/task_retention_promoted_final_fit/local_round1000_t3_semantic"
prediction="$formal_dir/t1_t6_t3/predictions.jsonl"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state_dir/smoke16" "$state_dir/full75"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while [[ ! -s "$prediction" ]]; do
  heartbeat
  [[ ! -f "$formal_dir/needs_attention" ]] || fail "promoted round-1000 formal screen needs attention"
  sleep 30
done
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "promoted round-1000 T3 eval panel drift"
python_bin=/home/v-yuejiali/fkft/.venv/bin/python
extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
[[ -x "$python_bin" ]] || fail "TRAPI judge environment unavailable"
cache="$state_dir/cache.jsonl"

attempt=0
while [[ ! -f "$state_dir/smoke_complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
      --output-dir "$state_dir/smoke16" --judge-backend trapi \
      --judge-workers 4 --judge-max-retries 2 --judge-timeout-seconds 60 \
      --judge-cache "$cache" --max-records 16 --judge-audit-samples 16 --seed 42 \
      >> "$state_dir/smoke.log" 2>&1 \
    && "$python_bin" - "$state_dir/smoke16/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['generation_rows']==x['judged_rows']==16 and x['judge_errors']==0,x
assert x['judge_backend']=='trapi' and len(x['judge_rubric_sha256'])==64,x
PY
  then touch "$state_dir/smoke_complete"; break; fi
  ((attempt < 10)) || fail "promoted round-1000 T3 judge smoke exhausted"
  sleep 60
done

attempt=0
while [[ ! -f "$state_dir/complete" ]]; do
  heartbeat; attempt=$((attempt+1))
  if PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
      --output-dir "$state_dir/full75" --judge-backend trapi \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$cache" --judge-audit-samples 75 --seed 42 \
      >> "$state_dir/full.log" 2>&1
  then
    if "$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
        --metrics "$state_dir/full75/metrics.json" \
        --output "$state_dir/t3_semantic_gate.json" --checkpoint-round 1000 \
        >> "$state_dir/controller.log" 2>&1
    then touch "$state_dir/scientific_gate_pass"
    else
      rc=$?; ((rc == 2)) || fail "promoted round-1000 T3 semantic validator failed rc=$rc"
      touch "$state_dir/scientific_gate_failed"
    fi
    touch "$state_dir/complete"; heartbeat
    log "promoted round-1000 T3 semantic diagnostic complete"
    break
  fi
  ((attempt < 10)) || fail "promoted round-1000 T3 semantic judge exhausted"
  sleep 60
done
