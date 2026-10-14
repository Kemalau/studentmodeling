#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
: "${CHECKPOINT_STEP:?CHECKPOINT_STEP must be 20 or 100}"
[[ "$CHECKPOINT_STEP" == 20 || "$CHECKPOINT_STEP" == 100 ]] || exit 2
step="$CHECKPOINT_STEP"
state="$project/results/state8_current_query_checkpoint${step}_screen_20260811"
template="$project/amlt/feeds_state8_current_query_checkpoint_screen_20260811.yaml"
base="state8-current-query-cp${step}-screen-20260811"
experiment="$base"; recovery=0; max_recovery=3
job="state8-current-query-cp${step}-screen"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
blob="studentmodeling/checkpoints/state8_stateonly_repair_seed3072_taskanchor_source_anchor_updater_mse1_current_query_v1/checkpoint-${step}/checkpoint_completion.json"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

receipt="$state/checkpoint_completion.json"
while [[ ! -f "$state/checkpoint_ready" ]]; do
  heartbeat
  exists="$(az storage blob exists --account-name csnewsandfeeds4150361735 \
    --container-name amulet --name "$blob" --auth-mode login --query exists -o tsv 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ "$exists" == true ]] || { sleep_unlocked 60; continue; }
  az storage blob download --account-name csnewsandfeeds4150361735 --container-name amulet \
    --name "$blob" --file "$receipt" --auth-mode login --no-progress -o none \
    || { rm -f "$receipt"; sleep_unlocked 60; continue; }
  "$python_bin" - "$receipt" "$step" <<'PY' || { rm -f "$receipt"; sleep_unlocked 30; continue; }
import json,sys
x=json.load(open(sys.argv[1])); step=int(sys.argv[2])
assert x.get('status')=='complete' and int(x.get('step',-1))==step,x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY
  touch "$state/checkpoint_ready"; log "checkpoint-$step receipt is complete"
done

config="$state/bound.yaml"
"$python_bin" scripts/materialize_state8_current_query_checkpoint_screen.py \
  --template "$template" --receipt "$receipt" --step "$step" --output "$config" \
  >> "$state/controller.log" 2>&1 || fail "config materialization failed"
"$python_bin" scripts/validate_state8_current_query_checkpoint_screen_amlt.py \
  --config "$config" --expected-step "$step" >> "$state/controller.log" 2>&1 \
  || fail "bound config validation failed"

submit_experiment(){
  local name="$1" output rc current class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$config" --dump -t Feeds > "$state/$name.preflight.yaml" \
    2> "$state/$name.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_current_query_checkpoint_screen_amlt.py \
    --config "$state/$name.preflight.yaml" --expected-step "$step" \
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  current="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
  if [[ "$class" == missing ]]; then
    amlt --json-tables ti sing -t Feeds > "$state/$name.quota.json" 2>> "$state/controller.log" \
      || { flock -u 8; return 1; }
    output="$(amlt run "$config" "$name" -t Feeds -y \
      --description "Frozen query-aligned State8 checkpoint-$step semantic and diagnosis-C4 screen" 2>&1)"; rc=$?
    printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
      > "$state/$name.submission.txt"
    (( rc == 0 )) || { flock -u 8; return 1; }
  elif [[ "$class" != exists ]]; then flock -u 8; return 1; fi
  flock -u 8; printf '%s\n' "$name" > "$state/active_experiment.txt"
}

if [[ -s "$state/active_experiment.txt" ]]; then experiment="$(<"$state/active_experiment.txt")"
else until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 120; done; fi

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
s={str(x.get('status') or '').casefold() for x in json.loads(os.environ['STATUS'])}
print('pass' if s=={'pass'} else ('active' if s&{'preparing','pre-queued','queued','running','resuming'} else ('paused' if 'paused' in s else 'terminal')))
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused) timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 60 ;;
    *)
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/$experiment.failure.log" 2>&1 || true
      (( recovery < max_recovery )) || fail "checkpoint-$step screen recovery exhausted"
      recovery=$((recovery+1)); experiment="${base}-recovery${recovery}"
      until submit_experiment "$experiment"; do heartbeat; sleep_unlocked 120; done
      log "submitted recovery=$recovery" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" ":$job" --include '*.json' --include '*.jsonl' \
  --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"
mapfile -t roots < <(find "$state/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality drift"
root="${roots[0]}"; out="$state/semantic"; mkdir -p "$out/t3" "$out/c4"
mapfile -t formal < <(find "$root" -type f -path '*/eval/formal_t3_75/predictions.jsonl')
mapfile -t diag_pred < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/judge_predictions.jsonl')
mapfile -t diag_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/judge_eval.jsonl')
mapfile -t diag_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_20/metrics.json')
[[ ${#formal[@]} -eq 1 && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "semantic artifact cardinality drift"
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
  >> "$out/c4/summarize.log" 2>&1 || fail "C4 summary failed"
baseline="$project/results/state8_source_anchor_checkpoint${step}_screen_20260811/comparison.json"
"$python_bin" - "$out/t3/metrics.json" "$out/c4/summary.json" "$baseline" "$state/comparison.json" "$step" <<'PY' \
  || fail "comparison failed"
import json,pathlib,sys
t3,c4,baseline=map(lambda p:json.load(open(p)),sys.argv[1:4]); step=int(sys.argv[5])
groups=t3['metrics_by_group']; legacy=[x for x in baseline['arms'] if x['arm']=='updater_mse1'][0]
def acc(fragment):
 rows=[v for k,v in groups.items() if fragment in k]; assert len(rows)==1,(fragment,groups.keys()); return rows[0]['llm_judge_accuracy']
current={'T3_eedi_semantic':acc('eedi_mining/open_taxonomy_generation'),
 'T3_synthetic_cross_semantic':acc('malrule/synthetic_cross_template'),
 'T3_synthetic_same_semantic':acc('malrule/synthetic_same_template'),
 'diagnosis_C4_D_specific':c4['semantic_D_specific'],
 'diagnosis_C4_swap_to_donor':c4['cells']['swap_to_donor']['semantic_equivalence_rate'],
 'diagnosis_C4_random':c4['random_matches_either_stable_target_rate']}
legacy_values={'T3_eedi_semantic':legacy['T3_eedi_semantic'],
 'T3_synthetic_mean_semantic':legacy['T3_synthetic_semantic'],
 'diagnosis_C4_D_specific':legacy['diagnosis_C4_D_specific'],
 'diagnosis_C4_swap_to_donor':legacy['diagnosis_C4_swap_to_donor'],
 'diagnosis_C4_random':legacy['diagnosis_C4_random']}
out={'stage':'state8_current_query_checkpoint_semantic_comparison_v1','status':'complete',
 'checkpoint_step':step,'diagnostic_only':True,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,'current_query':current,'legacy_generic_query':legacy_values}
pathlib.Path(sys.argv[4]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat; log "checkpoint-$step query-aligned semantic screen complete"
