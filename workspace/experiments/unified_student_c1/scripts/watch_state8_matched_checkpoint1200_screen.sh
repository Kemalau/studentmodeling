#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_matched_checkpoint1200_screen_20260810"
config="$project/amlt/feeds_state8_matched_checkpoint1200_screen_20260810.yaml"
base=state8-matched-checkpoint1200-screen-20260810
recovery_prefix=state8-matched-checkpoint1200-screen-recovery
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
max_waves=12
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
  local next="$1"; shift; local jobs=("$@") selectors=() output rc
  for job in "${jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_matched_checkpoint1200_screen_amlt.py \
    --config "$config" >> "$state/watcher.log" 2>&1 || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$next.preflight.yaml" 2> "$state/$next.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_matched_checkpoint1200_screen_amlt.py \
    --config "$state/$next.preflight.yaml" >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
    --description "Automatic failed-shard recovery for hash-bound matched checkpoint-1200 screen" 2>&1)"; rc=$?
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
    case "$job" in
      state8-matched-checkpoint1200-t3|state8-matched-checkpoint1200-c4)
        touch "$state/passed.$job" ;;
    esac
  done
  if [[ -f "$state/passed.state8-matched-checkpoint1200-t3" && \
        -f "$state/passed.state8-matched-checkpoint1200-c4" ]]; then
    touch "$state/remote_passed"; break
  fi
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
  (( wave < max_waves )) || fail "checkpoint-1200 screen exhausted recovery waves"
  wave=$((wave+1)); next="${recovery_prefix}${wave}-20260810"
  until submit_failed "$next" "${failed[@]}"; do heartbeat; sleep_unlocked 120; done
  experiment="$next"; printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  log "submitted failed screen shards wave=$wave experiment=$experiment jobs=${failed[*]}"
done

rm -rf "$state/download"; mkdir -p "$state/download"
for source in "$base" $(seq 1 "$wave" | sed "s#^#${recovery_prefix}#;s#\$#-20260810#"); do
  amlt results download "$source" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/download" >> "$state/watcher.log" 2>&1 || true
done
mapfile -t formal < <(find "$state/download" -type f -path '*/state8-matched-checkpoint1200-t3/eval/formal_t3_75/predictions.jsonl')
mapfile -t diag_pred < <(find "$state/download" -type f -path '*/state8-matched-checkpoint1200-c4/eval/diagnosis_c4_50/judge_predictions.jsonl')
mapfile -t diag_eval < <(find "$state/download" -type f -path '*/state8-matched-checkpoint1200-c4/eval/diagnosis_c4_50/judge_eval.jsonl')
mapfile -t diag_metrics < <(find "$state/download" -type f -path '*/state8-matched-checkpoint1200-c4/eval/diagnosis_c4_50/metrics.json')
mapfile -t answer < <(find "$state/download" -type f -path '*/state8-matched-checkpoint1200-c4/eval/answer_c4_50/recurrent_causal_gate.json')
[[ ${#formal[@]} -ge 1 && ${#diag_pred[@]} -ge 1 && ${#diag_eval[@]} -ge 1 && ${#diag_metrics[@]} -ge 1 && ${#answer[@]} -ge 1 ]] \
  || fail "screen artifact cardinality incomplete"
formal="${formal[-1]}"; diag_pred="${diag_pred[-1]}"; diag_eval="${diag_eval[-1]}"; diag_metrics="${diag_metrics[-1]}"; answer="${answer[-1]}"
test "$(sha256sum "$eval_jsonl" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 \
  || fail "formal evaluation reference drift"

mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$formal" --eval-jsonl "$eval_jsonl" --output-dir "$state/t3_semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$state/t3_semantic/cache.jsonl" \
  --judge-audit-samples 75 --seed 1200 >> "$state/t3_semantic/judge.log" 2>&1 \
  || fail "T3 semantic judge failed"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$diag_pred" --eval-jsonl "$diag_eval" --output-dir "$state/c4_semantic" \
  --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$state/c4_semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 1200 >> "$state/c4_semantic/judge.log" 2>&1 \
  || fail "C4 semantic judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "$diag_metrics" --output "$state/c4_semantic/summary.json" \
  >> "$state/c4_semantic/summarize.log" 2>&1 || fail "C4 semantic summary failed"

"$python_bin" - "$state/t3_semantic/metrics.json" "$state/c4_semantic/summary.json" "$answer" "$state/diagnostic_summary.json" <<'PY'
import json,pathlib,sys
t3=json.load(open(sys.argv[1])); c4=json.load(open(sys.argv[2])); answer=json.load(open(sys.argv[3]))
assert t3['judge_backend']=='trapi' and t3['judge_errors']==0,t3
groups=t3['metrics_by_group']; support=sum(int(v['llm_judge_n']) for v in groups.values())
accuracy=sum(float(v['llm_judge_accuracy'])*int(v['llm_judge_n']) for v in groups.values())/support
out={'stage':'state8_matched_checkpoint1200_diagnostic_v1','status':'complete','diagnostic_only':True,
 'checkpoint_step':1200,'t3_semantic_accuracy':accuracy,'t3_support':support,
 'diagnosis_c4':c4,'answer_c4':answer,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'final_inference_route':'strict_state_only'}
pathlib.Path(sys.argv[4]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "checkpoint-1200 diagnostic semantic screen complete"
