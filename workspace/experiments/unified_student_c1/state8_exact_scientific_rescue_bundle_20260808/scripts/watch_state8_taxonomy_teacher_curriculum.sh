#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment="${STATE8_CURRICULUM_EXPERIMENT:-state8-taxonomy-teacher-curriculum-20260807}"
job="${STATE8_CURRICULUM_JOB:-state8-taxonomy-teacher-curriculum-seed42}"
slug="${STATE8_CURRICULUM_SLUG:-taxonomy_t3heavy}"
case "$slug" in taxonomy_t3heavy|taxonomy_balanced) ;; *) exit 2 ;; esac
state="$project/results/state8_taxonomy_curriculum_candidates_20260807/$slug"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
stall_seconds="${STATE8_CURRICULUM_STALL_SECONDS:-2700}"
startup_stall_seconds="${STATE8_CURRICULUM_STARTUP_STALL_SECONDS:-7200}"
posttrain_stall_seconds="${STATE8_CURRICULUM_POSTTRAIN_STALL_SECONDS:-21600}"
poll_seconds="${STATE8_CURRICULUM_POLL_SECONDS:-120}"
progress="$state/progress"

mkdir -p "$state" "$progress"
cd "$project" || exit 1
# Prefer the default profile while it has a live token, then fall back to the
# dedicated Feeds profile.  AMLT jobs are workspace-scoped rather than tied to
# the local cache that submitted them; pinning an expired cache made healthy
# remote jobs invisible to the recovery controller.
if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

observe_training(){
  local phase="$1" log_file="" listing=""
  if [[ "$phase" != running ]]; then
    rm -f "$progress/stalled.json" "$progress/health_alert.json"
    return 0
  fi
  listing="$(timeout 90 amlt logs list "$experiment" ":$job" 2>/dev/null)" || return 0
  log_file="$(LISTING="$listing" "$python_bin" - <<'PY'
import os,re
paths=[]
for line in os.environ['LISTING'].splitlines():
 value=line.strip()
 if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
  retry=re.search(r'retry_(\d+)',value); paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)" || return 0
  [[ -n "$log_file" ]] || return 0
  timeout 90 amlt logs view -F "$log_file" -n 600 "$experiment" ":$job" \
    > "$progress/latest_rank0.log" 2> "$progress/latest_rank0.stderr" || return 0
  "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
    --log "$progress/latest_rank0.log" --experiment "$experiment" \
    --log-file "$log_file" --status running --output "$progress/progress.json" \
    --total-rounds 1250 --max-run-seconds 172800 >/dev/null 2>&1 || return 0
  if "$python_bin" - "$progress/progress.json" "$progress/latest_rank0.log" "$stall_seconds" "$startup_stall_seconds" "$posttrain_stall_seconds" "$job" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); unhealthy=(x.get('health') or {}).get('healthy') is False
log=open(sys.argv[2],encoding='utf-8',errors='replace').read()
training_complete=(float(x.get('progress_fraction') or 0)>=1.0
                   or int(x.get('optimizer_step') or -1)>=int(x.get('total_rounds') or 1250)
                   or (sys.argv[6]=='state8-taxonomy-t3heavy-finalizer-seed42'
                       and '"round": 1250' in log
                       and '"status": "complete"' in log
                       and 'STATE8_CURRICULUM_STAGE=training' in log))
awaiting_first=x.get('phase')=='awaiting_first_training_record'
threshold=float(sys.argv[5] if training_complete else sys.argv[4] if awaiting_first else sys.argv[3])
stalled=float(x.get('seconds_since_progress') or 0)>=threshold
raise SystemExit(0 if stalled or unhealthy else 1)
PY
  then
    cp "$progress/progress.json" "$progress/stalled.json"
    timeout 90 amlt show "$experiment" > "$state/stalled.show.txt" 2>&1 || true
    timeout 180 amlt logs view -n 1200 "$experiment" ":$job" \
      > "$state/stalled.log" 2>&1 || true
    if timeout 90 amlt cancel "$experiment" ":$job" -y \
      > "$state/stalled.cancel.txt" 2>&1; then
      log "cancelled stalled/unhealthy curriculum for exact recovery experiment=$experiment"
    else
      fail "stalled curriculum cancellation failed experiment=$experiment"
    fi
  else
    rm -f "$progress/stalled.json" "$progress/health_alert.json"
  fi
}

while true; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
print(rows[0]['status'] if len(rows)==1 else 'invalid')
PY
)" || { sleep 120; continue; }
  observe_training "$phase"
  case "$phase" in
    preparing|queued|running|paused|resuming|pre-queued) sleep "$poll_seconds"; continue ;;
    pass) touch "$state/remote_passed"; break ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 1000 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
      fail "remote curriculum ended in state=$phase"
      ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"
readarray -t predictions < <(find "$state/download" -type f -path "*/$job/eval/formal_screen/predictions.jsonl")
readarray -t metrics < <(find "$state/download" -type f -path "*/$job/eval/formal_screen/metrics.json")
readarray -t receipts < <(find "$state/download" -type f -path "*/$job/metrics/taxonomy_teacher_curriculum.json")
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 && ${#receipts[@]} -eq 1 ]] \
  || fail "artifact cardinality drift"

"$python_bin" - "${metrics[0]}" "${receipts[0]}" <<'PY' || exit 1
import json,sys
m=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2]))
assert m['raw_history_visible_to_decoder'] is False,m
assert m['persistent_history_kv_cache'] is False,m
assert m['state_shape']==[64,8,5120],m
assert r['semantic_judge_authoritative'] is True,r
assert r['exact_label_metric_authoritative'] is False,r
assert r['taxonomy_candidates_visible_to_decoder'] is False,r
assert float(r['label_auxiliary_weight'])==4.0,r
PY

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/t3_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/t3_semantic" --judge-backend trapi \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/t3_semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 42 >> "$state/t3_semantic/judge.log" 2>&1 \
  || fail "semantic judge failed"

"$python_bin" - "$state/t3_semantic/metrics.json" "$state/semantic_gate.json" <<'PY' || exit 1
import json,pathlib,sys
x=json.load(open(sys.argv[1]))
assert x['generation_rows']==x['judged_rows'] and x['judge_errors']==0,x
assert x['judge_backend']=='trapi',x
groups=x['metrics_by_group']
assert groups and all(int(g.get('llm_judge_exact_shortcuts',-1))==0 for g in groups.values()),groups
support=sum(int(g['llm_judge_n']) for g in groups.values())
accuracy=sum(float(g['llm_judge_accuracy'])*int(g['llm_judge_n']) for g in groups.values())/support
gate={'stage':'state8_taxonomy_curriculum_semantic_measurement_v1','measurement_complete':True,
      'semantic_judge_authoritative':True,'exact_match_authoritative':False,
      'judge_backend':'trapi','support':support,'semantic_accuracy':accuracy,
      'metrics':sys.argv[1]}
pathlib.Path(sys.argv[2]).write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n')
PY

touch "$state/complete"; heartbeat
log "remote curriculum and authoritative semantic judgment completed"
