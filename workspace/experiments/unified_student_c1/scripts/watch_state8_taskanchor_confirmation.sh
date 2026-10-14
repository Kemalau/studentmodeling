#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
screen="$project/results/state8_taskanchor_milestone_screen_20260811"
state="$project/results/state8_taskanchor_confirmation_20260811"
config="$project/amlt/feeds_state8_taskanchor_confirmation_generated_20260811.yaml"
template="$project/amlt/feeds_state8_taskanchor_milestone_screen_20260811.yaml"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
base=state8-taskanchor-full-confirmation-20260811
recovery_prefix=state8-taskanchor-full-confirmation-recovery
max_waves=32
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state/preserved_results" "$state/t3_semantic/full1494" \
  "$state/diagnosis_c4_semantic"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  [[ -z "$heartbeat_pid" ]] || { kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; }
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
sleep_unlocked(){ sleep "$1" 9>&-; }
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
rm -f "$state/needs_attention"

while [[ ! -f "$screen/complete" ]]; do
  heartbeat
  [[ ! -f "$screen/needs_attention" ]] || fail "milestone screen needs attention"
  sleep_unlocked 60
done
[[ -s "$screen/comparison.json" ]] || fail "milestone comparison missing"

readarray -t selected < <("$python_bin" - "$screen/comparison.json" \
    "$template" <<'PY'
import json,sys,yaml
comparison=json.load(open(sys.argv[1])); step=comparison.get('selected_for_full_confirmation')
if step is None: raise SystemExit(2)
rows={int(row['checkpoint_step']):row for row in comparison['milestones']}
row=rows[int(step)]; artifacts=row['artifact_sha256']
config=yaml.safe_load(open(sys.argv[2]))
jobs=config['jobs']; match=[job for job in jobs if str(job['submit_args']['env']['MILESTONE_STEP'])==str(step)]
assert len(match)==1,match
env=match[0]['submit_args']['env']
assert env['MILESTONE_ADAPTER_SHA256']==artifacts['adapter_model'],(env,artifacts)
assert env['MILESTONE_STATE_SHA256']==artifacts['recurrent_state'],(env,artifacts)
print(step); print(artifacts['adapter_model']); print(artifacts['recurrent_state']); print(env['MILESTONE_RECEIPT_SHA256'])
PY
) || {
  touch "$state/scientific_gate_failed" "$state/complete"
  log "no milestone passed the bounded joint gate; confirmation not submitted"
  exit 2
}
[[ ${#selected[@]} -eq 4 ]] || fail "invalid milestone selection"
step="${selected[0]}"; adapter_sha="${selected[1]}"; state_sha="${selected[2]}"; receipt_sha="${selected[3]}"

"$python_bin" scripts/materialize_state8_taskanchor_confirmation_amlt.py \
  --template "$template" --output "$config" --step "$step" \
  --adapter-sha256 "$adapter_sha" --state-sha256 "$state_sha" \
  --receipt-sha256 "$receipt_sha" >> "$state/controller.log" 2>&1 \
  || fail "could not materialize full confirmation config"
"$python_bin" scripts/validate_state8_taskanchor_confirmation_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 \
  || fail "full confirmation config validation failed"

all_jobs=()
for index in $(seq -w 0 15); do all_jobs+=("state8-taskanchor-confirm-step${step}-formal-shard$index"); done
for seed in 13 42 97; do all_jobs+=("state8-taskanchor-confirm-step${step}-c4-seed$seed"); done

preserve_passed(){
  local experiment="$1" status job
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || return 1
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${passed[@]}"; do
    [[ -f "$state/passed.$job" ]] && continue
    amlt results download "$experiment" ":$job" \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 || return 1
    touch "$state/passed.$job"; log "preserved job=$job"
  done
}

submit_jobs(){
  local experiment="$1"; shift
  local requested=("$@") selected_jobs=() selectors=() audit available auditable output rc
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  audit="$state/$experiment.active_a100.json"
  "$python_bin" scripts/count_active_a100_requests.py --user-limit 32 --recent 800 \
    --output "$audit" >/dev/null 2>> "$state/controller.log" || { flock -u 8; return 1; }
  read -r available auditable < <("$python_bin" - "$audit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(int(x['available_user_slots']),int(bool(x['auditable'])))
PY
  )
  [[ "$auditable" == 1 ]] || { flock -u 8; return 1; }
  (( available > 0 )) || { flock -u 8; return 2; }
  selected_jobs=("${requested[@]:0:available}")
  for job in "${selected_jobs[@]}"; do selectors+=(":$job"); done
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_taskanchor_confirmation_amlt.py \
    --config "$state/$experiment.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" \
    2>> "$state/controller.log" || { flock -u 8; return 1; }
  output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Hash-bound step-$step State8 full confirmation: complete 21676-row formal panel and three 150-pair answer/diagnosis C4 seeds" 2>&1)"; rc=$?
  printf '%s\n' "$output" > "$state/$experiment.submission.txt"
  flock -u 8
  (( rc == 0 )) || return 1
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${selected_jobs[@]}" > "$state/active_jobs.txt"
  log "submitted experiment=$experiment jobs=${#selected_jobs[@]} step=$step"
}

experiment="$base"; wave=0
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260811$ ]] && wave="${BASH_REMATCH[1]}"
else
  until submit_jobs "$experiment" "${all_jobs[@]}"; do heartbeat; sleep_unlocked 120; done
fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" || fail "could not preserve passed confirmation sibling"
  missing=(); for job in "${all_jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]} == 0 )) && { touch "$state/remote_passed"; break; }
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
states={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
active={'preparing','pre-queued','queued','running','resuming'}
print('active' if states & active else 'paused' if 'paused' in states else 'terminal')
PY
)"
  if [[ "$phase" == active ]]; then sleep_unlocked 60; continue; fi
  if [[ "$phase" == paused ]]; then
    selectors=(); for job in "${missing[@]}"; do selectors+=(":$job"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" >> "$state/controller.log" 2>&1 || true
    sleep_unlocked 60; continue
  fi
  (( wave < max_waves )) || fail "confirmation exhausted recovery waves"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260811"
  until submit_jobs "$experiment" "${missing[@]}"; do heartbeat; sleep_unlocked 120; done
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/"
mapfile -t predictions < <(find "$state/download" -type f -path '*/eval/formal_shard*/predictions.jsonl' | sort)
mapfile -t metrics < <(find "$state/download" -type f -path '*/eval/formal_shard*/metrics.json' | sort)
mapfile -t c4_results < <(find "$state/download" -type f -path '*/eval/c4_seed*/recurrent_causal_gate.json' | sort)
mapfile -t diagnosis_predictions < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_seed*/judge_predictions.jsonl' | sort)
mapfile -t diagnosis_evals < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_seed*/judge_eval.jsonl' | sort)
mapfile -t diagnosis_metrics < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_seed*/metrics.json' | sort)
[[ ${#predictions[@]} -eq 16 && ${#metrics[@]} -eq 16 && ${#c4_results[@]} -eq 3 \
  && ${#diagnosis_predictions[@]} -eq 3 && ${#diagnosis_evals[@]} -eq 3 \
  && ${#diagnosis_metrics[@]} -eq 3 ]] || fail "confirmation artifact cardinality drift"
"$python_bin" scripts/merge_state8_final_eval.py \
  --predictions "${predictions[@]}" --metrics "${metrics[@]}" \
  --eval-jsonl "$eval_jsonl" --c4 "${c4_results[@]}" --output-dir "$state/merged" \
  >> "$state/controller.log" 2>&1 || fail "confirmation merge failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
(exec 9>&-; while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$state/merged/t3_predictions.jsonl" --eval-jsonl "$eval_jsonl" \
    --output-dir "$state/t3_semantic/full1494" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 \
    --judge-timeout-seconds 60 --judge-cache "$state/t3_semantic/cache.jsonl" \
    --judge-audit-samples 100 --seed 42 >> "$state/t3_semantic/judge.log" 2>&1; do
  heartbeat; sleep_unlocked 120
done
set +e
"$python_bin" scripts/validate_t3_semantic_result.py \
  --metrics "$state/t3_semantic/full1494/metrics.json" \
  --output "$state/t3_semantic/gate.json" >> "$state/t3_semantic/validate.log" 2>&1
t3_rc=$?
set -e
(( t3_rc == 0 || t3_rc == 2 )) || fail "T3 semantic validator crashed"

diagnosis_summaries=()
for index in 0 1 2; do
  seed="$("$python_bin" - "${diagnosis_metrics[$index]}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['pair_seed'])
PY
)"
  out="$state/diagnosis_c4_semantic/seed$seed"; mkdir -p "$out"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${diagnosis_predictions[$index]}" --eval-jsonl "${diagnosis_evals[$index]}" \
      --output-dir "$out" --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
      --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$out/cache.jsonl" --judge-audit-samples 100 --seed "$seed" \
      >> "$out/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$out/predictions_redacted.jsonl" \
    --generation-metrics "${diagnosis_metrics[$index]}" --output "$out/summary.json" \
    >> "$out/summarize.log" 2>&1 || fail "diagnosis C4 summary failed seed=$seed"
  diagnosis_summaries+=("$out/summary.json")
done
"$python_bin" - "${diagnosis_summaries[@]}" "$state/diagnosis_c4_semantic/multiseed.json" <<'PY'
import json,pathlib,statistics,sys
rows=[json.load(open(path)) for path in sys.argv[1:-1]]
assert {int(row['generation']['pair_seed']) for row in rows}=={13,42,97},rows
result={'stage':'state8_closed_diagnosis_c4_semantic_multiseed_v1',
 'semantic_judge_authoritative':True,'exact_label_role':'auxiliary_only_not_used_for_gate',
 'rows':rows,'seed_passes':sum(bool(row['pass']) for row in rows),
 'mean_semantic_D_specific':statistics.mean(row['semantic_D_specific'] for row in rows),
 'mean_swap_to_donor':statistics.mean(row['cells']['swap_to_donor']['semantic_equivalence_rate'] for row in rows),
 'mean_random_matches_either':statistics.mean(row['random_matches_either_stable_target_rate'] for row in rows),
 'pass':sum(bool(row['pass']) for row in rows)>=2}
pathlib.Path(sys.argv[-1]).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY
kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; heartbeat_pid=""

set +e
"$python_bin" scripts/validate_state8_taskanchor_confirmation.py \
  --formal "$state/merged/metrics.json" --c4 "$state/merged/c4_multiseed.json" \
  --semantic-gate "$state/t3_semantic/gate.json" \
  --diagnosis-c4 "$state/diagnosis_c4_semantic/multiseed.json" \
  --step "$step" --adapter-sha256 "$adapter_sha" --state-sha256 "$state_sha" \
  --output "$state/confirmation.json" >> "$state/controller.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  az storage blob upload --auth-mode login --account-name csnewsandfeeds4150361735 \
    --container-name amulet --name studentmodeling/gates/state8_taskanchor_confirmation_v1/selection.json \
    --file "$state/confirmation.json" --overwrite >> "$state/controller.log" 2>&1 \
    || fail "confirmation gate upload failed"
  touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then
  touch "$state/scientific_gate_failed"
else fail "confirmation validator crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "task-anchor full confirmation reached terminal state rc=$rc step=$step"

