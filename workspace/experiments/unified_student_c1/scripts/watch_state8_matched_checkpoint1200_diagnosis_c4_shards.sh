#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="${STATE8_CP1200_C4_STATE:-$project/results/state8_matched_checkpoint1200_screen_20260810/c4_shards}"
config="$project/amlt/feeds_state8_matched_checkpoint1200_diagnosis_c4_shards_20260810.yaml"
base="${STATE8_CP1200_C4_BASE:-state8-matched-checkpoint1200-diagnosis-c4-shards-20260810}"
recovery_prefix="${STATE8_CP1200_C4_RECOVERY_PREFIX:-state8-matched-checkpoint1200-diagnosis-c4-shards-recovery}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
max_waves=12
jobs=(
  state8-matched-cp1200-diagnosis-c4-shard00
  state8-matched-cp1200-diagnosis-c4-shard10
  state8-matched-cp1200-diagnosis-c4-shard20
  state8-matched-cp1200-diagnosis-c4-shard30
  state8-matched-cp1200-diagnosis-c4-shard40
)
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
echo $$ > "$state/watcher.active.pid"
cleanup(){ rm -f "$state/watcher.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/watcher.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }
rm -f "$state/needs_attention"

experiment="$(<"$state/active_experiment.txt")"
wave=0
[[ "$experiment" =~ recovery([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"

submit_failed(){
  local next="$1"; shift; local failed=("$@") selectors=() output rc
  for job in "${failed[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_matched_checkpoint1200_diagnosis_c4_shards_amlt.py \
    --config "$config" >> "$state/watcher.log" 2>&1 || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$next.preflight.yaml" 2> "$state/$next.preflight.stderr" \
    || { flock -u 8; return 1; }
  validation=("$python_bin" scripts/validate_state8_matched_checkpoint1200_diagnosis_c4_shards_amlt.py \
    --config "$state/$next.preflight.yaml")
  for job in "${failed[@]}"; do validation+=(--job "$job"); done
  "${validation[@]}" >> "$state/watcher.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
    --description "Automatic failed-shard recovery for hash-bound checkpoint-1200 diagnosis C4" 2>&1)"; rc=$?
  printf '%s\n' "$output" > "$state/$next.submission.txt"
  flock -u 8
  return "$rc"
}

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  mapfile -t passed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import passed_job_names
sys.stdout.write(''.join(value+'\n' for value in passed_job_names(sys.argv[1])))
PY
)
  for job in "${passed[@]}"; do
    for expected in "${jobs[@]}"; do [[ "$job" == "$expected" ]] && touch "$state/passed.$job"; done
  done
  all_passed=1
  for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || all_passed=0; done
  (( all_passed == 0 )) || { touch "$state/remote_passed"; break; }
  mapfile -t paused < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import paused_job_names
sys.stdout.write(''.join(value+'\n' for value in paused_job_names(sys.argv[1])))
PY
)
  if (( ${#paused[@]} > 0 )); then
    selectors=(); for job in "${paused[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/watcher.log" 2>&1 || true
    sleep_unlocked 60; continue
  fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep_unlocked 60; continue; fi
  mapfile -t failed < <("$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
sys.stdout.write(''.join(value+'\n' for value in terminal_nonpass_job_names(sys.argv[1])))
PY
)
  (( ${#failed[@]} > 0 )) || { sleep_unlocked 60; continue; }
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_waves )) || fail "checkpoint-1200 C4 shards exhausted recovery waves"
  wave=$((wave+1)); next="${recovery_prefix}${wave}-20260810"
  until submit_failed "$next" "${failed[@]}"; do heartbeat; sleep_unlocked 120; done
  experiment="$next"; printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted failed C4 shards wave=$wave jobs=${failed[*]}"
done

rm -rf "$state/download" "$state/merged"; mkdir -p "$state/download" "$state/merged"
for source in "$base" $(seq 1 "$wave" | sed "s#^#${recovery_prefix}#;s#\$#-20260810#"); do
  amlt results download "$source" --status pass --include '*.json' --include '*.jsonl' \
    --output "$state/download" >> "$state/watcher.log" 2>&1 || true
done
mapfile -t metrics < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_shard_*/metrics.json' | sort)
[[ ${#metrics[@]} -ge 5 ]] || fail "C4 shard artifacts incomplete"
declare -A shard_by_start=()
for metric in "${metrics[@]}"; do
  start="$("$python_bin" - "$metric" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['pair_start']))
PY
)" || fail "could not read C4 shard range"
  shard_by_start[$start]="$(dirname "$metric")"
done
[[ ${#shard_by_start[@]} -eq 5 ]] || fail "C4 shard range coverage drift"
merge_args=(); for start in 0 10 20 30 40; do
  [[ -n "${shard_by_start[$start]:-}" ]] || fail "missing C4 shard start=$start"
  merge_args+=(--shard "${shard_by_start[$start]}")
done
PYTHONPATH="$project/scripts" "$python_bin" scripts/merge_state8_diagnosis_c4_shards.py \
  "${merge_args[@]}" --expected-pairs 50 --output-dir "$state/merged" \
  >> "$state/watcher.log" 2>&1 || fail "C4 shard merge failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$state/merged/judge_predictions.jsonl" \
  --eval-jsonl "$state/merged/judge_eval.jsonl" --output-dir "$state/c4_semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 1200 \
  >> "$state/c4_semantic/judge.log" 2>&1 || fail "C4 semantic judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "$state/merged/metrics.json" \
  --output "$state/c4_semantic/summary.json" >> "$state/watcher.log" 2>&1 \
  || fail "C4 semantic summary failed"
touch "$state/complete"; heartbeat
log "checkpoint-1200 sharded diagnosis C4 merged and semantically judged"
