#!/usr/bin/env bash
set -uo pipefail
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_source_anchor_checkpoint20_screen_20260811"
config="$project/amlt/feeds_state8_source_anchor_checkpoint20_screen_20260811.yaml"
base=state8-source-anchor-cp20-screen-20260811
recovery_prefix=state8-source-anchor-cp20-screen-recovery
experiment="$base"; wave=0; max_waves=12
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
jobs=(state8-source-anchor-cp20-full-mse1 state8-source-anchor-cp20-updater-mse1 state8-source-anchor-cp20-full-mse2)
arms=(full_mse1 updater_mse1 full_mse2)
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"
mkdir -p "$state/preserved_results"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
rm -f "$state/needs_attention"
if [[ -s "$state/active_experiment.txt" ]]; then
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260811$ ]] && wave="${BASH_REMATCH[1]}"
else printf '%s\n' "$experiment" > "$state/active_experiment.txt"; fi

preserve_passed(){
  local current="$1" status="$2" job
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  for job in "${passed[@]}"; do
    [[ -f "$state/passed.$job" ]] && continue
    amlt results download "$current" ":$job" --include '*.json' --include '*.jsonl' --include '*.txt' \
      --output "$state/preserved_results" >> "$state/controller.log" 2>&1 || return 1
    touch "$state/passed.$job"; log "preserved job=$job"
  done
}

submit_missing(){
  local next="$1"; shift; local missing=("$@") selectors=() output rc current class
  for job in "${missing[@]}"; do selectors+=(":$job"); done
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state/$next.preflight.yaml" \
    2> "$state/$next.preflight.stderr" || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$next" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$next.quota.json" || { flock -u 8; return 1; }
    output="$(amlt run "$config" "${selectors[@]}" "$next" -t Feeds -y \
      --description "Automatic recovery of hash-bound State8 source-anchor checkpoint-20 screen" 2>&1)"; rc=$?
    printf '%s\n' "$output" > "$state/$next.submission.txt"; (( rc==0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8; printf '%s\n' "$next" > "$state/active_experiment.txt"
}

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  preserve_passed "$experiment" "$status" || fail "result preservation failed"
  missing=(); for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]}==0 )) && { touch "$state/remote_passed"; break; }
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
print('active' if s&{'preparing','pre-queued','queued','running','resuming'} else ('paused' if 'paused' in s else 'terminal'))
PY
)"
  case "$phase" in
    active) sleep_unlocked 60 ;;
    paused) timeout 90 amlt resume "$experiment" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 60 ;;
    *)
      timeout 180 amlt logs view -n 3000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
      (( wave<max_waves )) || fail "checkpoint-20 screen recovery exhausted"
      wave=$((wave+1)); next="${recovery_prefix}${wave}-20260811"
      until submit_missing "$next" "${missing[@]}"; do heartbeat; sleep_unlocked 120; done
      experiment="$next"; log "submitted recovery wave=$wave jobs=${missing[*]}" ;;
  esac
done

for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"; arm="${arms[$index]}"; out="$state/semantic/$arm"
  mapfile -t roots < <(find "$state/preserved_results" -type d -name "$job")
  [[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality arm=$arm"
  root="${roots[0]}"; mkdir -p "$out/t3" "$out/c4"
  mapfile -t formal < <(find "$root" -type f -path '*/eval/formal_t3_75/predictions.jsonl')
  mapfile -t diag_pred < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/judge_predictions.jsonl')
  mapfile -t diag_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/judge_eval.jsonl')
  mapfile -t diag_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/metrics.json')
  [[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
    || fail "semantic artifact cardinality arm=$arm"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal[0]}" --eval-jsonl "$eval_jsonl" --output-dir "$out/t3" \
    --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$out/t3/cache.jsonl" \
    --judge-audit-samples 75 --seed 42 >> "$out/t3/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" --output-dir "$out/c4" \
    --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$out/c4/cache.jsonl" \
    --judge-audit-samples 100 --seed 42 >> "$out/c4/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  "$python_bin" scripts/summarize_state8_diagnosis_c4.py --judge-redacted "$out/c4/predictions_redacted.jsonl" \
    --generation-metrics "${diag_metrics[0]}" --output "$out/c4/summary.json" \
    >> "$out/c4/summarize.log" 2>&1 || fail "C4 summary failed arm=$arm"
done
"$python_bin" - "$state" "${arms[@]}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=[]
for arm in sys.argv[2:]:
 t3=json.load(open(root/'semantic'/arm/'t3/metrics.json')); c4=json.load(open(root/'semantic'/arm/'c4/summary.json'))
 groups=t3['metrics_by_group']; e=[v for k,v in groups.items() if '/eedi_mining/' in k][0]
 mal=[v for k,v in groups.items() if '/malrule/' in k]
 rows.append({'arm':arm,'T3_eedi_semantic':e['llm_judge_accuracy'],
  'T3_synthetic_semantic':sum(v['llm_judge_accuracy'] for v in mal)/len(mal),
  'T3_schema':sum(v['natural_schema_valid_rate'] for v in groups.values())/len(groups),
  'diagnosis_C4_D_specific':c4['semantic_D_specific'],
  'diagnosis_C4_swap_to_donor':c4['cells']['swap_to_donor']['semantic_equivalence_rate'],
  'diagnosis_C4_random':c4['random_matches_either_stable_target_rate'],
  'diagnostic_only':True})
out={'stage':'state8_source_anchor_checkpoint20_semantic_comparison_v1','status':'complete',
 'diagnostic_only':True,'not_a_promotion_gate':True,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'arms':rows}
(root/'comparison.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat; log "checkpoint-20 semantic comparison complete"
