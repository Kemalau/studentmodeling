#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source "$project/scripts/use_studentmodeling_feeds_auth.sh"
state="$project/results/state8_curriculum_cp5000_semantic_causal_repair_20260812"
config="$state/bound.yaml"
experiment=state8-cp5000-semantic-causal-repair-20260812
job=state8-eedi-repair-semantic-binding-seed7201
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

"$python_bin" scripts/validate_state8_eedi_same_question_repair_branches_amlt.py \
  --config "$config" >> "$state/controller.log" 2>&1 \
  || fail "bound config validation failed"

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls \
    --force-status-update 2>/dev/null)" || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
if s=={'pass'}: print('pass')
elif s&{'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in s: print('paused')
elif s&{'failed','killed','cancelled','canceled','stopped'}: print('failed')
else: print('unknown')
PY
  )"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active|unknown) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    failed)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
        > "$state/failure.log" 2>&1 || true
      fail "remote repair job failed" ;;
  esac
done

rm -rf "$state/download"
mkdir -p "$state/download"
amlt results download "$experiment" ":$job" --include '*.json' --include '*.jsonl' \
  --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"
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

formal_pred="$(one_artifact "*/$job/eval/formal_t3_300/predictions.jsonl")" \
  || fail "formal predictions missing"
formal_metrics="$(one_artifact "*/$job/eval/formal_t3_300/metrics.json")" \
  || fail "formal metrics missing"
malrule_pred="$(one_artifact "*/$job/eval/diagnosis_c4_50/judge_predictions.jsonl")" \
  || fail "MalRule predictions missing"
malrule_eval="$(one_artifact "*/$job/eval/diagnosis_c4_50/judge_eval.jsonl")" \
  || fail "MalRule judge references missing"
malrule_metrics="$(one_artifact "*/$job/eval/diagnosis_c4_50/metrics.json")" \
  || fail "MalRule metrics missing"
eedi_pred="$(one_artifact "*/$job/eval/eedi_same_question_c4_25/judge_predictions.jsonl")" \
  || fail "Eedi predictions missing"
eedi_eval="$(one_artifact "*/$job/eval/eedi_same_question_c4_25/judge_eval.jsonl")" \
  || fail "Eedi judge references missing"
eedi_metrics="$(one_artifact "*/$job/eval/eedi_same_question_c4_25/metrics.json")" \
  || fail "Eedi metrics missing"

judge(){
  local predictions="$1" evaluation="$2" output="$3" audit="$4" attempt=0
  mkdir -p "$output"
  while [[ ! -f "$output/metrics.json" ]]; do
    attempt=$((attempt+1)); heartbeat
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
      --output-dir "$output" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$output/cache.jsonl" --judge-audit-samples "$audit" \
      --seed 7201 >> "$output/judge.log" 2>&1 && break
    (( attempt < 30 )) || fail "semantic judge exhausted output=$output"
    sleep_unlocked 120
  done
}

semantic="$state/semantic"
judge "$formal_pred" "$eval_jsonl" "$semantic/t3" 75
judge "$malrule_pred" "$malrule_eval" "$semantic/malrule_c4" 100
judge "$eedi_pred" "$eedi_eval" "$semantic/eedi_c4" 100
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$semantic/malrule_c4/predictions_redacted.jsonl" \
  --generation-metrics "$malrule_metrics" \
  --output "$semantic/malrule_c4/summary.json" \
  || fail "MalRule semantic C4 summary failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$semantic/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "$eedi_metrics" \
  --output "$semantic/eedi_c4/summary.json" \
  || fail "Eedi semantic C4 summary failed"

"$python_bin" - "$config" "$formal_metrics" "$semantic/t3/metrics.json" \
  "$semantic/malrule_c4/summary.json" "$semantic/eedi_c4/summary.json" \
  "$semantic/summary.json" <<'PY' || fail "semantic candidate summary failed"
import json,pathlib,sys,yaml
config=yaml.safe_load(open(sys.argv[1])); job=config['jobs'][0]; env=job['submit_args']['env']
formal,t3,malrule,eedi=(json.load(open(path)) for path in sys.argv[2:6])
metadata={'job':job['name'],'seed':int(env['REPAIR_SEED']),
 'remote_root':env['REPAIR_REMOTE_ROOT'],'finalfit_variant':'same_question_repair_promoted',
 'repair_branch':env['REPAIR_BRANCH'],'source_anchor_mode':env['SOURCE_ANCHOR_MODE']}
out={'stage':'state8_eedi_same_question_repair_semantic_summary_v1','status':'complete',
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'formal_metrics':formal,'t3_semantic':t3,'malrule_diagnosis_c4':malrule,
 'eedi_same_question_diagnosis_c4':eedi,'candidate_metadata':metadata}
pathlib.Path(sys.argv[6]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

set +e
"$python_bin" scripts/select_state8_eedi_same_question_candidate.py \
  --candidate repair_semantic_binding_seed7201 "$semantic/summary.json" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "semantic selector crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "checkpoint-5000 semantic causal repair judged; selector_rc=$rc"
