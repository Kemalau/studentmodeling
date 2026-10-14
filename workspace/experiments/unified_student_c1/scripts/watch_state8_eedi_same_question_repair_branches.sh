#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
selection="${STATE8_REPAIR_SELECTION_DIR:-$project/results/state8_eedi_same_question_early_selection_20260811}"
state="${STATE8_REPAIR_STATE_DIR:-$project/results/state8_eedi_same_question_repair_branches_20260811}"
template="$project/amlt/feeds_state8_eedi_same_question_causal_repair_20260811.yaml"
config="$state/bound.yaml"
experiment="${STATE8_REPAIR_BASE_EXPERIMENT:-state8-eedi-same-question-repair-branches-20260811}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment="$experiment"
max_waves=12

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$selection/complete" ]]; do heartbeat; sleep_unlocked 60; done
if [[ "${STATE8_CURRICULUM_REPAIR:-0}" == 1 ]]; then
  if [[ -f "$selection/scientific_gate_pass" ]]; then
    touch "$state/not_required" "$state/complete"; heartbeat
    log "curriculum selection passed; no repair branches submitted"
    exit 0
  fi
  [[ -f "$selection/scientific_gate_failed" ]] \
    || fail "curriculum selection is terminal without a gate result"
  plan="$state/repair_plan.json"
  inputs=()
  for arm in generic current_query; do
    for step in 2000 3000 4000 5000; do
      inputs+=(--candidate "$project/results/state8_curriculum_screens_20260812/$arm/checkpoint-$step/summary.json")
    done
  done
  "$python_bin" scripts/plan_state8_curriculum_repair.py "${inputs[@]}" \
    --output "$plan" >> "$state/controller.log" 2>&1 \
    || fail "curriculum repair planning failed"
elif [[ ! -f "$selection/repair_required" && ! -f "$selection/extension_required" ]]; then
  touch "$state/not_required" "$state/complete"; heartbeat
  log "viable early checkpoint exists; no repair branches submitted"
  exit 0
elif [[ -f "$selection/extension_required" ]]; then
  plan="$selection/extension_plan.json"
else
  plan="$selection/repair_plan.json"
fi
[[ -f "$plan" ]] || fail "repair-required marker has no repair plan"

"$python_bin" scripts/materialize_state8_eedi_same_question_repair_branches.py \
  --template "$template" --plan "$plan" --results-root "$project/results" \
  --output "$config" >> "$state/controller.log" 2>&1 \
  || fail "repair config materialization failed"
"$python_bin" scripts/validate_state8_eedi_same_question_repair_branches_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 \
  || fail "repair config validation failed"

mapfile -t all_jobs < <("$python_bin" - "$config" <<'PY'
import sys,yaml
x=yaml.safe_load(open(sys.argv[1])); print(*(j['name'] for j in x['jobs']),sep='\n')
PY
)
[[ ${#all_jobs[@]} -ge 1 && ${#all_jobs[@]} -le 3 ]] \
  || fail "repair job matrix cardinality drift"

submit_jobs(){
  local name="$1"; shift
  local names=("$@") selectors=() current class output rc
  for job_name in "${names[@]}"; do selectors+=(":$job_name"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds \
    > "$state/$name.preflight.yaml" 2> "$state/$name.preflight.stderr" \
    || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_eedi_same_question_repair_branches_amlt.py \
    --config "$state/$name.preflight.yaml" >> "$state/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$name.quota.json" \
      2>> "$state/controller.log" || { flock -u 8; return 1; }
    "$python_bin" - "$state/$name.quota.json" "${#names[@]}" <<'PY' \
      || { flock -u 8; return 1; }
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
row=select_a100_80gb(parse_json_table(open(sys.argv[1]).read()))
requested=int(sys.argv[2])
assert row['series']=='NDAMv4',row
assert row['accelerator']['type']=='A100' and row['accelerator']['memory_gb']==80,row
quota=row['quotas_(used/limit)']['standard']
assert quota['limit']-quota['used']>=requested,(quota,requested)
assert 1<=requested<=3,requested
PY
    output="$(amlt run "$config" "${selectors[@]}" "$name" -t Feeds -y \
      --description "Orthogonal State8 repair branches selected from failed 20/100 semantic and dual-C4 screens" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8
  printf '%s\n' "$name" > "$state/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state/active_jobs.txt"
}

if [[ -s "$state/active_experiment.txt" && -s "$state/active_jobs.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
else
  until submit_jobs "$experiment" "${all_jobs[@]}"; do
    heartbeat; sleep_unlocked 120
  done
fi
touch "$state/submitted"; heartbeat
log "repair branches accepted for Feeds submission"

wave=0
[[ "$experiment" =~ recovery([0-9]+)$ ]] && wave="${BASH_REMATCH[1]}"
mkdir -p "$state/preserved_results"
while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls \
    --force-status-update 2>/dev/null)" || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row.get('job_name') or '').lstrip(':'))
PY
  )
  if (( ${#passed[@]} > 0 )); then
    amlt results download "$experiment" --status pass \
      --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 \
      || { sleep_unlocked 60; continue; }
    for job_name in "${passed[@]}"; do
      for expected in "${all_jobs[@]}"; do
        [[ "$job_name" == "$expected" ]] && touch "$state/passed.$expected"
      done
    done
  fi
  all_passed=1
  for job_name in "${all_jobs[@]}"; do [[ -f "$state/passed.$job_name" ]] || all_passed=0; done
  (( all_passed == 0 )) || { touch "$state/remote_passed"; break; }
  mapfile -t paused < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='paused': print(str(row.get('job_name') or '').lstrip(':'))
PY
  )
  if (( ${#paused[@]} > 0 )); then
    selectors=(); for job_name in "${paused[@]}"; do selectors+=(":$job_name"); done
    timeout 90 amlt resume "$experiment" "${selectors[@]}" \
      >> "$state/controller.log" 2>&1 || true
    sleep_unlocked 60; continue
  fi
  active="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
print(int(bool(s&{'preparing','pre-queued','queued','running','resuming'})))
PY
)"
  [[ "$active" == 0 ]] || { sleep_unlocked 60; continue; }
  mapfile -t failed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
terminal={'failed','killed','cancelled','canceled','stopped'}
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold() in terminal: print(str(row.get('job_name') or '').lstrip(':'))
PY
  )
  (( ${#failed[@]} > 0 )) || { sleep_unlocked 60; continue; }
  for job_name in "${failed[@]}"; do
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job_name" \
      > "$state/$experiment.$job_name.failure.log" 2>&1 || true
  done
  (( wave < max_waves )) || fail "repair branches exhausted recovery waves"
  wave=$((wave+1)); experiment="${base_experiment}-recovery${wave}"
  until submit_jobs "$experiment" "${failed[@]}"; do heartbeat; sleep_unlocked 120; done
  log "submitted failed repair siblings wave=$wave jobs=${failed[*]}"
done

rm -rf "$state/download"; mkdir -p "$state/download"
cp -a "$state/preserved_results/." "$state/download/"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"

one_artifact(){
  local pattern="$1" values=() first digest value
  mapfile -t values < <(find "$state/download" -type f -path "$pattern" | sort)
  (( ${#values[@]} >= 1 )) || return 1
  first="${values[0]}"; digest="$(sha256sum "$first" | cut -d' ' -f1)"
  for value in "${values[@]}"; do
    [[ "$(sha256sum "$value" | cut -d' ' -f1)" == "$digest" ]] || return 1
  done
  printf '%s\n' "$first"
}

judge(){
  local predictions="$1" evaluation="$2" output="$3" audit="$4" attempt=0
  bash scripts/wait_for_trapi_auth.sh
  while [[ ! -f "$output/metrics.json" ]]; do
    attempt=$((attempt+1)); heartbeat
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
      --output-dir "$output" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$output/cache.jsonl" --judge-audit-samples "$audit" \
      --seed 42 >> "$output/judge.log" 2>&1 && break
    (( attempt < 30 )) || fail "semantic judge exhausted output=$output"
    sleep_unlocked 120
  done
}

candidate_args=()
for job_name in "${all_jobs[@]}"; do
  slug="$("$python_bin" - "$config" "$job_name" <<'PY'
import sys,yaml
x=yaml.safe_load(open(sys.argv[1])); j=next(v for v in x['jobs'] if v['name']==sys.argv[2])
e=j['submit_args']['env']; print(f"repair_{e['REPAIR_BRANCH']}_seed{e['REPAIR_SEED']}")
PY
)" || fail "repair slug extraction failed job=$job_name"
  arm="$state/$slug"; mkdir -p "$arm/t3_semantic" "$arm/malrule_c4" "$arm/eedi_c4"
  formal_pred="$(one_artifact "*/$job_name/eval/formal_t3_300/predictions.jsonl")" \
    || fail "formal predictions missing job=$job_name"
  formal_metrics="$(one_artifact "*/$job_name/eval/formal_t3_300/metrics.json")" \
    || fail "formal metrics missing job=$job_name"
  malrule_pred="$(one_artifact "*/$job_name/eval/diagnosis_c4_50/judge_predictions.jsonl")" \
    || fail "MalRule predictions missing job=$job_name"
  malrule_eval="$(one_artifact "*/$job_name/eval/diagnosis_c4_50/judge_eval.jsonl")" \
    || fail "MalRule eval missing job=$job_name"
  malrule_metrics="$(one_artifact "*/$job_name/eval/diagnosis_c4_50/metrics.json")" \
    || fail "MalRule metrics missing job=$job_name"
  eedi_pred="$(one_artifact "*/$job_name/eval/eedi_same_question_c4_25/judge_predictions.jsonl")" \
    || fail "Eedi predictions missing job=$job_name"
  eedi_eval="$(one_artifact "*/$job_name/eval/eedi_same_question_c4_25/judge_eval.jsonl")" \
    || fail "Eedi eval missing job=$job_name"
  eedi_metrics="$(one_artifact "*/$job_name/eval/eedi_same_question_c4_25/metrics.json")" \
    || fail "Eedi metrics missing job=$job_name"

  judge "$formal_pred" "$eval_jsonl" "$arm/t3_semantic" 75
  judge "$malrule_pred" "$malrule_eval" "$arm/malrule_c4" 100
  judge "$eedi_pred" "$eedi_eval" "$arm/eedi_c4" 100
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$arm/malrule_c4/predictions_redacted.jsonl" \
    --generation-metrics "$malrule_metrics" --output "$arm/malrule_c4/summary.json" \
    || fail "MalRule repair summary failed job=$job_name"
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
    --judge-redacted "$arm/eedi_c4/predictions_redacted.jsonl" \
    --generation-metrics "$eedi_metrics" --output "$arm/eedi_c4/summary.json" \
    || fail "Eedi repair summary failed job=$job_name"
  "$python_bin" - "$config" "$job_name" "$formal_metrics" \
    "$arm/t3_semantic/metrics.json" "$arm/malrule_c4/summary.json" \
    "$arm/eedi_c4/summary.json" "$arm/summary.json" <<'PY' \
    || fail "repair candidate summary failed job=$job_name"
import json,pathlib,sys,yaml
config=yaml.safe_load(open(sys.argv[1])); job=next(x for x in config['jobs'] if x['name']==sys.argv[2])
env=job['submit_args']['env']; formal,t3,malrule,eedi=(json.load(open(p)) for p in sys.argv[3:7])
metadata={'job':job['name'],'seed':int(env['REPAIR_SEED']),
 'remote_root':env['REPAIR_REMOTE_ROOT'],'finalfit_variant':'same_question_repair_promoted',
 'repair_branch':env['REPAIR_BRANCH'],'source_anchor_mode':env['SOURCE_ANCHOR_MODE']}
out={'stage':'state8_eedi_same_question_repair_semantic_summary_v1','status':'complete',
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'formal_metrics':formal,'t3_semantic':t3,'malrule_diagnosis_c4':malrule,
 'eedi_same_question_diagnosis_c4':eedi,'candidate_metadata':metadata}
pathlib.Path(sys.argv[7]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
  candidate_args+=(--candidate "$slug" "$arm/summary.json")
done

set +e
"$python_bin" scripts/select_state8_eedi_same_question_candidate.py \
  "${candidate_args[@]}" --output "$state/selection.json" \
  > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "repair candidate selector crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "repair branches semantically evaluated and selected rc=$rc"
