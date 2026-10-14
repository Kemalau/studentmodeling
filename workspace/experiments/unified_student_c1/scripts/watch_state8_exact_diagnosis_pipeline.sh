#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_exact_diagnosis_pipeline_20260808"
config="$project/amlt/feeds_state8_exact_diagnosis_pipeline_20260808.yaml"
experiment=state8-exact-diagnosis-first-curriculum-c4-20260808
recovery_prefix=state8-exact-diagnosis-first-curriculum-c4-recovery
recovery_suffix=-20260808
if [[ -s "$state/active_experiment.txt" ]]; then experiment="$(<"$state/active_experiment.txt")"; fi
job=state8-exact-diagnosis-first-curriculum-c4-seed64
max_recovery_waves=24
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds
  fi
fi
mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"; flock -n 9 || exit 0
# Keep the persistent watcher restartable after an interrupt: a child sleep
# must never inherit and outlive the process that owns the singleton lock.
sleep_unlocked(){ sleep "$1" 9>&-; }
echo $$ > "$state/watcher.active.pid"
cleanup(){ rm -f "$state/watcher.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/watcher.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

phase_of(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS'])
print(str(rows[0].get('status') or '').casefold() if len(rows)==1 else 'invalid')
PY
}

submit_recovery(){
  local recovery_experiment="$1" rc status observed
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_exact_diagnosis_pipeline_amlt.py \
    --config "$config" >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt run "$config" --dump -t Feeds \
    > "$state/$recovery_experiment.preflight.yaml" \
    2> "$state/$recovery_experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_exact_diagnosis_pipeline_amlt.py \
    --config "$state/$recovery_experiment.preflight.yaml" \
    >> "$state/watcher.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds \
    > "$state/$recovery_experiment.quota.json" \
    2>> "$state/watcher.log" || { flock -u 8; return 1; }
  amlt run "$config" "$recovery_experiment" -t Feeds -y \
    --description "Automatic resumable recovery of exact 8-state diagnosis-first curriculum, strict state-only answer/diagnosis C4" \
    > "$state/$recovery_experiment.submission.txt" 2>&1
  rc=$?
  if (( rc != 0 )); then
    # A submission command can lose its response after the service has already
    # accepted the experiment.  Never issue an ambiguous duplicate: first
    # prove whether this exact recovery name now exists.
    status="$(timeout 90 amlt --json-tables status "$recovery_experiment" \
      --hide-urls 2>/dev/null)" || { flock -u 8; return "$rc"; }
    observed="$(phase_of "$status")" || { flock -u 8; return "$rc"; }
    case "$observed" in
      preparing|pre-queued|queued|running|paused|resuming|pass)
        printf '\nsubmission_rc=%s but exact experiment exists with status=%s\n' \
          "$rc" "$observed" >> "$state/$recovery_experiment.submission.txt"
        ;;
      *) flock -u 8; return "$rc" ;;
    esac
  fi
  flock -u 8
  return 0
}

wave=0
if [[ "$experiment" =~ recovery([0-9]+)-20260808 ]]; then
  wave="${BASH_REMATCH[1]}"
fi

submitted_preflight(){
  local name="$1" current="$state/$1.preflight.yaml" legacy
  if [[ -s "$current" ]]; then
    printf '%s\n' "$current"
    return 0
  fi
  if [[ "$name" == state8-exact-diagnosis-first-curriculum-c4-20260808 ]]; then
    printf '%s\n' "$project/results/state8_exact_diagnosis_pipeline_20260808.preflight.yaml"
  elif [[ "$name" =~ recovery([0-9]+)-20260808 ]]; then
    legacy="$state/recovery${BASH_REMATCH[1]}.preflight.yaml"
    [[ -s "$legacy" ]] && printf '%s\n' "$legacy"
  fi
}

submitted_c4_sha(){
  local preflight="$1"
  [[ -s "$preflight" ]] || return 1
  "$python_bin" - "$preflight" <<'PY'
import sys,yaml
x=yaml.safe_load(open(sys.argv[1]))
jobs=x.get('jobs') or []
assert len(jobs)==1,jobs
env=(jobs[0].get('submit_args') or {}).get('env') or {}
value=str(env.get('STATE8_DIAGNOSIS_FIRST_C4_SHA256') or '')
assert len(value)==64,value
print(value)
PY
}

replace_stale_provenance_attempt(){
  local preflight observed expected status phase stale_experiment replacement receipt
  preflight="$(submitted_preflight "$experiment")"
  [[ -n "$preflight" && -s "$preflight" ]] || return 0
  observed="$(submitted_c4_sha "$preflight")" || return 1
  expected="$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)"
  [[ "$observed" == "$expected" ]] && return 0

  stale_experiment="$experiment"
  status="$(timeout 90 amlt --json-tables status "$stale_experiment" --hide-urls 2>/dev/null)" \
    || return 1
  phase="$(phase_of "$status")" || return 1
  case "$phase" in
    preparing|pre-queued|queued|running|paused|resuming)
      timeout 90 amlt cancel "$stale_experiment" ":$job" -y \
        >> "$state/watcher.log" 2>&1 || return 1
      log "cancel requested for stale diagnosis-C4 provenance experiment=$stale_experiment observed=$observed expected=$expected"
      return 1
      ;;
    pass|killed|failed|cancelled|canceled) ;;
    *) return 1 ;;
  esac

  wave=$((wave+1))
  replacement="${recovery_prefix}${wave}${recovery_suffix}"
  until submit_recovery "$replacement"; do
    heartbeat
    log "transient hash-bound replacement submission failure wave=$wave; retrying"
    sleep_unlocked 120
  done
  experiment="$replacement"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  receipt="$state/stale_provenance_replacement_wave${wave}.json"
  STATUS="$status" STALE_EXPERIMENT="$stale_experiment" REPLACEMENT="$replacement" \
    OBSERVED="$observed" EXPECTED="$expected" PREFLIGHT="$preflight" \
    "$python_bin" - "$receipt" <<'PY'
import json,os,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
 'stage':'state8_exact_stale_provenance_replacement_v1',
 'status':'replacement_submitted',
 'stale_experiment':os.environ['STALE_EXPERIMENT'],
 'replacement_experiment':os.environ['REPLACEMENT'],
 'stale_preflight':os.environ['PREFLIGHT'],
 'observed_diagnosis_c4_sha256':os.environ['OBSERVED'],
 'expected_diagnosis_c4_sha256':os.environ['EXPECTED'],
 'stale_status':json.loads(os.environ['STATUS']),
 'stale_results_eligible_for_promotion':False,
},indent=2,sort_keys=True)+'\n')
PY
  log "submitted hash-bound replacement experiment=$experiment for stale=$stale_experiment"
  return 0
}

while ! replace_stale_provenance_attempt; do
  heartbeat
  sleep_unlocked 120
done

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep_unlocked 120; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(phase_of "$status")" || { sleep_unlocked 120; continue; }
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" \
        >> "$state/watcher.log" 2>&1 || true
      sleep_unlocked 120
      ;;
    preparing|pre-queued|queued|running|resuming) sleep_unlocked 120 ;;
    killed)
      if [[ -f "$state/superseded_before_run" ]]; then exit 0; fi
      timeout 90 amlt show "$experiment" \
        > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) \
        || fail "exact diagnosis pipeline exhausted recovery waves after killed job"
      wave=$((wave+1)); experiment="${recovery_prefix}${wave}${recovery_suffix}"
      printf '%s\n' "$experiment" > "$state/active_experiment.txt"
      until submit_recovery "$experiment"; do
        heartbeat; log "transient recovery submission failure wave=$wave; retrying"; sleep_unlocked 120
      done
      log "submitted automatic recovery wave=$wave experiment=$experiment after killed job"
      ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_recovery_waves )) \
        || fail "exact diagnosis pipeline exhausted recovery waves terminal_status=$phase"
      wave=$((wave+1)); experiment="${recovery_prefix}${wave}${recovery_suffix}"
      printf '%s\n' "$experiment" > "$state/active_experiment.txt"
      until submit_recovery "$experiment"; do
        heartbeat; log "transient recovery submission failure wave=$wave; retrying"; sleep_unlocked 120
      done
      log "submitted automatic recovery wave=$wave experiment=$experiment terminal_status=$phase"
      ;;
  esac
done

if [[ ! -f "$state/results_downloaded" ]]; then
  rm -rf "$state/download"; mkdir -p "$state/download"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/download" >> "$state/watcher.log" 2>&1 \
    || fail "result download failed"
  touch "$state/results_downloaded"
fi

mapfile -t formal < <(find "$state/download" -type f -path "*/$job/eval/formal_t3_300/predictions.jsonl")
mapfile -t diag_pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")
mapfile -t diag_eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")
mapfile -t diag_metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_50/metrics.json")
[[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "evaluation artifact cardinality drift"

if [[ ! -f "$state/t3_semantic_complete" ]]; then
  mkdir -p "$state/t3_semantic"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$state/t3_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$state/t3_semantic/judge.log" 2>&1 \
    || fail "T3 semantic judge failed"
  "$python_bin" - "$state/t3_semantic/metrics.json" "$state/t3_semantic/authority.json" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); groups=x['metrics_by_group']
support=sum(int(row['llm_judge_n']) for row in groups.values())
api=sum(int(row['llm_judge_api_or_cache']) for row in groups.values())
shortcuts=sum(int(row['llm_judge_exact_shortcuts']) for row in groups.values())
assert x['judge_backend']=='trapi' and x['judge_errors']==0,x
assert support==x['generation_rows']==x['judged_rows'],x
assert api==support and shortcuts==0,groups
result={'stage':'state8_exact_t3_semantic_authority_v1','pass':True,
 'support':support,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'canonical_label_role':'auxiliary_only'}
pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY
  touch "$state/t3_semantic_complete"
fi

if [[ ! -f "$state/c4_semantic_complete" ]]; then
  mkdir -p "$state/c4_semantic"
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$state/c4_semantic" --judge-backend trapi \
    --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$state/c4_semantic/judge.log" 2>&1 \
    || fail "diagnosis-C4 semantic judge failed"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" \
    --output "$state/c4_semantic/summary.json" \
    >> "$state/c4_semantic/summarize.log" 2>&1 \
    || fail "diagnosis-C4 semantic summary failed"
  touch "$state/c4_semantic_complete"
fi

"$python_bin" - "$state/t3_semantic/metrics.json" "$state/c4_semantic/summary.json" \
  "$state/final_semantic_summary.json" <<'PY'
import json,pathlib,sys
t3,c4=map(lambda path:json.load(open(path)),sys.argv[1:3])
groups=t3['metrics_by_group']
support=sum(int(row['llm_judge_n']) for row in groups.values())
accuracy=sum(float(row['llm_judge_accuracy'])*int(row['llm_judge_n']) for row in groups.values())/support
out={'stage':'state8_exact_diagnosis_semantic_summary_v1','status':'complete',
 't3_semantic_accuracy':accuracy,'t3_support':support,
 'diagnosis_c4':c4,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'canonical_label_role':'auxiliary_only'}
pathlib.Path(sys.argv[3]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
set +e
"$python_bin" scripts/select_state8_exact_diagnosis_curriculum.py \
  --root "$state" --output "$state/selection.json" \
  > "$state/selection.log" 2>&1
selection_rc=$?
set -e
if (( selection_rc == 0 )); then
  touch "$state/scientific_gate_pass"
elif (( selection_rc == 2 )); then
  touch "$state/scientific_gate_failed"
else
  fail "exact diagnosis curriculum selector crashed rc=$selection_rc"
fi
touch "$state/complete"
heartbeat
log "exact diagnosis-first semantic evaluation and joint selection complete"
