#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_taskanchor_milestone_screen_20260811"
config="$project/amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml"
base=state8-taskanchor-milestone-screen-20260811
recovery_prefix=state8-taskanchor-milestone-screen-recovery
experiment="$base"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
steps=(200 400 600 800 1000)
max_waves=12
wave=0
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state/preserved_results"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
rm -f "$state/needs_attention"

if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260811$ ]] && wave="${BASH_REMATCH[1]}"
else
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
fi

status_phase(){
  STATUS="$1" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); states={str(x.get('status') or '').casefold() for x in rows}
active={'preparing','pre-queued','queued','running','resuming'}
if states and states <= {'pass'}: print('pass')
elif states & active: print('active')
elif 'paused' in states: print('paused')
elif states: print('terminal')
else: print('invalid')
PY
}

preserve_passed(){
  local status="$1" job
  mapfile -t observed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${observed[@]}"; do
    [[ -f "$state/passed.$job" ]] && continue
    amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 || return 1
    touch "$state/passed.$job"; log "preserved job=$job"
  done
}

submit_missing(){
  local next="$1"; shift
  local jobs=("$@") selectors=() output rc
  for job in "${jobs[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  "$python_bin" scripts/validate_state8_taskanchor_milestone_screen_amlt.py \
    --config "$config" >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state/$next.preflight.yaml" \
    2> "$state/$next.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_taskanchor_milestone_screen_amlt.py \
    --config "$state/$next.preflight.yaml" --allow-job-subset \
    >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
    --description "Automatic hash-bound recovery for read-only task-anchor milestone screens" 2>&1)"; rc=$?
  printf '%s\n' "$output" > "$state/$next.submission.txt"
  flock -u 8
  (( rc == 0 ))
}

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$status" || fail "result preservation failed"
  missing=()
  for step in "${steps[@]}"; do
    job="state8-taskanchor-milestone-$step"
    [[ -f "$state/passed.$job" ]] || missing+=("$job")
  done
  (( ${#missing[@]} == 0 )) && { touch "$state/remote_passed"; break; }
  phase="$(status_phase "$status")"
  case "$phase" in
    active) sleep_unlocked 60 ;;
    paused)
      selectors=(); for job in "${missing[@]}"; do selectors+=(":$job"); done
      timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 2000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
      (( wave < max_waves )) || fail "recovery waves exhausted: ${missing[*]}"
      wave=$((wave+1)); next="${recovery_prefix}${wave}-20260811"
      until submit_missing "$next" "${missing[@]}"; do heartbeat; sleep_unlocked 120; done
      experiment="$next"; printf '%s\n' "$experiment" > "$state/active_experiment.txt"
      log "submitted recovery wave=$wave jobs=${missing[*]}" ;;
  esac
done

for step in "${steps[@]}"; do
  job="state8-taskanchor-milestone-$step"
  mapfile -t roots < <(find "$state/preserved_results" -type d -name "$job")
  [[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality step=$step"
  root="${roots[0]}"; out="$state/step$step"
  mkdir -p "$out/t3_semantic" "$out/c4_semantic"
  mapfile -t formal_pred < <(find "$root" -type f -path '*/eval/formal_t3_75/predictions.jsonl')
  mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_t3_75/metrics.json')
  mapfile -t answer < <(find "$root" -type f -path '*/eval/answer_c4_50/recurrent_causal_gate.json')
  mapfile -t diag_pred < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/judge_predictions.jsonl')
  mapfile -t diag_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/judge_eval.jsonl')
  mapfile -t diag_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/metrics.json')
  [[ ${#formal_pred[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 && ${#answer[@]} -eq 1 \
     && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "evaluation artifact cardinality step=$step"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${formal_pred[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$out/t3_semantic" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$out/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
      >> "$out/t3_semantic/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
      --output-dir "$out/c4_semantic" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$out/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
      >> "$out/c4_semantic/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$out/c4_semantic/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" --output "$out/c4_semantic/summary.json" \
    >> "$out/c4_semantic/summarize.log" 2>&1 || fail "C4 summary failed step=$step"
  "$python_bin" - "$step" "$out/t3_semantic/metrics.json" \
    "$out/c4_semantic/summary.json" "${answer[0]}" "${formal_metrics[0]}" \
    "$out/summary.json" <<'PY'
import json,pathlib,sys
step=int(sys.argv[1]); t3,diagnosis,answer,formal=map(lambda p:json.load(open(p)),sys.argv[2:6])
groups=t3['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
s=answer['summary']; clean=s['clean']
out={'stage':'state8_taskanchor_milestone_semantic_summary_v1','status':'complete',
 'checkpoint_step':step,'diagnostic_only':True,'state_token_count':8,
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'t3_semantic_accuracy':accuracy,'t3_support':support,
 't3_metrics_by_group':groups,'formal_metrics':formal['metrics_by_group'],
 'answer_c4':{'clean_n':s['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
  'random':clean['random']['flip_to_A_rate'],'same_rule':clean['samemal']['flip_to_A_rate'],
  'noop':clean['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
  'scientific_gate_pass':s['scientific_gate_pass']},'diagnosis_c4':diagnosis,
 'artifact_sha256':answer['artifact_sha256']}
path=pathlib.Path(sys.argv[6]); tmp=path.with_suffix('.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
done

"$python_bin" - "$state" "${steps[@]}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); steps=list(map(int,sys.argv[2:]))
rows=[json.load(open(root/f'step{s}/summary.json')) for s in steps]
for row in rows:
 a=row['answer_c4']; d=row['diagnosis_c4']
 row['answer_diagnostic_gate']=(a['D_specific']>=.15 and a['random']<=.25 and a['same_rule']<=.10 and a['noop']<=.05)
 row['diagnosis_diagnostic_gate']=(d['semantic_D_specific']>=.15 and d['random_matches_either_stable_target_rate']<=.25 and d['noop_recipient_retention']>=.75 and d['same_rule_recipient_retention']>=.65 and d['cells']['swap_to_donor']['semantic_equivalence_rate']>=.50)
eligible=[r for r in rows if r['answer_diagnostic_gate'] and r['diagnosis_diagnostic_gate']]
selected=max(eligible,key=lambda r:(r['t3_semantic_accuracy'],r['diagnosis_c4']['semantic_D_specific']))['checkpoint_step'] if eligible else None
out={'stage':'state8_taskanchor_milestone_trajectory_v1','status':'complete',
 'diagnostic_only':True,'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'selection_rule':'joint 50-pair answer and 20-pair diagnosis causal screen, then T3 semantic accuracy; requires full confirmation',
 'selected_for_full_confirmation':selected,'milestones':rows}
path=root/'comparison.json'; tmp=path.with_suffix('.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY

touch "$state/complete"; heartbeat
log "task-anchor milestone trajectory and semantic judgment complete"
