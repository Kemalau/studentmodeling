#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=promoted-round1250-same-rule-repair-ab-recovery7-20260814
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1250_same_rule_repair_ab/recovery7"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project_dir/scripts:/home/v-yuejiali/fkft/src"
jobs=(promoted-round1250-repair-base-seed42 promoted-round1250-repair-same-rule-seed42)

export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2
mkdir -p "$state_dir/logs" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

rm -f "$state_dir/complete" "$state_dir/needs_attention" "$state_dir/replication_recommended"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

while true; do
  heartbeat
  partial="$state_dir/status.json.partial"
  if ! timeout 90 amlt --json-tables status "$experiment" \
      --no-calculate-result-size --hide-urls --force-status-update > "$partial" 2>> "$state_dir/controller.log"; then
    rm -f "$partial"
    sleep 60
    continue
  fi
  mv "$partial" "$state_dir/status.json"
  state="$($python_bin - "$state_dir/status.json" <<'PY'
import collections,json,sys
rows=json.load(open(sys.argv[1],encoding='utf-8'))
if len(rows)!=2:
    print('invalid')
else:
    counts=collections.Counter(str(row.get('status','')).lower() for row in rows)
    if counts=={'pass':2}: print('pass')
    elif any(key in counts for key in ('failed','killed','cancelled','canceled')): print('failed')
    else: print('active')
PY
)"
  case "$state" in
    active)
      sleep 60
      ;;
    failed)
      for job in "${jobs[@]}"; do
        timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
          > "$state_dir/logs/$job.log" 2>&1 || true
      done
      fail "one or more recovery7 jobs terminated without passing"
      ;;
    pass)
      log "both recovery7 jobs passed operationally"
      break
      ;;
    *)
      fail "AMLT status cardinality or schema drift"
      ;;
  esac
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
heartbeat
amlt results download "$experiment" --status pass \
  --include '*.json' --include 'predictions.jsonl' \
  --include 'predictions_redacted.jsonl' --output "$state_dir/download" \
  --n-workers 8 >> "$state_dir/controller.log" 2>&1 \
  || fail "result download failed"

for job in "${jobs[@]}"; do
  variant=base
  [[ "$job" == promoted-round1250-repair-same-rule-seed42 ]] && variant=same_rule
  mapfile -t predictions < <(find "$state_dir/download" -type f \
    -path "*${job}*/eval/formal_screen/predictions.jsonl" | sort)
  [[ ${#predictions[@]} -eq 1 ]] \
    || fail "expected one T3 prediction file arm=$job found=${#predictions[@]}"
  output="$state_dir/t3_semantic/$variant"
  mkdir -p "$output/judged"
  attempt=0
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$output/judged" --judge-backend trapi --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
      --judge-audit-samples 75 --seed 42 >> "$output/live.log" 2>&1; do
    attempt=$((attempt + 1))
    (( attempt < 30 )) || fail "T3 semantic judge exhausted arm=$job"
    heartbeat
    sleep 120
  done
  "$python_bin" scripts/validate_checkpoint_t3_semantic_result.py \
    --metrics "$output/judged/metrics.json" --output "$output/gate.json" \
    --checkpoint-round 1500 >> "$output/live.log" 2>&1 || rc=$?
  rc=${rc:-0}
  (( rc == 0 || rc == 2 )) \
    || fail "T3 semantic validation error arm=$job rc=$rc"
  unset rc
done

"$python_bin" scripts/summarize_promoted_round1250_same_rule_repair_ab.py \
  --root "$state_dir/download" --t3-root "$state_dir/t3_semantic" \
  --output "$state_dir/summary.json" >> "$state_dir/controller.log" 2>&1 \
  || fail "A/B summary failed"

if "$python_bin" - "$state_dir/summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['replication_recommended'] else 1)
PY
then
  touch "$state_dir/replication_recommended"
fi
touch "$state_dir/complete"
heartbeat
log "recovery7 A/B complete; recipe remains unpromoted pending replication"
