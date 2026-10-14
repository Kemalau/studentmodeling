#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_current_query_causal_repair_20260811"
experiment=state8-current-query-causal-repair-20260811
job=state8-current-query-updater-mse1-seed3072
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); states={str(x.get('status') or '').casefold() for x in rows}
if len(rows)==1 and states=={'pass'}: print('pass')
elif states & {'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in states: print('paused')
elif not states or states <= {'', 'unknown'}: print('unknown')
else: print('terminal')
PY
)" || fail "status classification failed"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    unknown)
      log "AMLT returned unknown status; preserving remote job and retrying"
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
      fail "query-aligned repair reached terminal non-pass state" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" ":$job" \
  --include '*.json' --include '*.jsonl' --include '*.txt' --output "$state/download" \
  >> "$state/controller.log" 2>&1 || fail "result download failed"
mapfile -t roots < <(find "$state/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality drift"
root="${roots[0]}"; out="$state/semantic"; mkdir -p "$out/t3" "$out/c4"
mapfile -t formal_pred < <(find "$root" -type f -path '*/eval/formal_t3_300/predictions.jsonl')
mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_t3_300/metrics.json')
mapfile -t diag_pred < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
mapfile -t diag_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
mapfile -t diag_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
mapfile -t source_receipt < <(find "$root" -type f -path '*/metrics/diagnosis_first_source_anchored_repair.json')
mapfile -t completion < <(find "$root" -type f -path '*/metrics/completion.json')
[[ ${#formal_pred[@]} -eq 1 && ${#formal_metrics[@]} -eq 1 \
   && ${#diag_pred[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 \
   && ${#source_receipt[@]} -eq 1 && ${#completion[@]} -eq 1 ]] \
  || fail "evaluation artifact cardinality drift"

until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${formal_pred[0]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$out/t3" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$out/t3/cache.jsonl" --judge-audit-samples 75 --seed 42 \
    >> "$out/t3/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${diag_pred[0]}" --eval-jsonl "${diag_eval[0]}" \
    --output-dir "$out/c4" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$out/c4/cache.jsonl" --judge-audit-samples 100 --seed 42 \
    >> "$out/c4/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$out/c4/predictions_redacted.jsonl" \
  --generation-metrics "${diag_metrics[0]}" --output "$out/c4/summary.json" \
  >> "$out/c4/summarize.log" 2>&1 || fail "diagnosis-C4 summary failed"
"$python_bin" scripts/summarize_state8_source_anchor_candidate.py \
  --arm current_query_updater_mse1 --job "$job" \
  --formal-metrics "${formal_metrics[0]}" --t3-semantic "$out/t3/metrics.json" \
  --diagnosis-c4 "$out/c4/summary.json" --source-receipt "${source_receipt[0]}" \
  --completion "${completion[0]}" --output "$state/summary.json" \
  || fail "candidate summary failed"
"$python_bin" - "$state/summary.json" "$state/comparison.json" <<'PY' || fail "comparison failed"
import json,pathlib,sys
current=json.load(open(sys.argv[1])); groups=current['t3_metrics_by_group']; c4=current['diagnosis_c4']
def acc(fragment):
 values=[v for k,v in groups.items() if fragment in k]; assert len(values)==1,(fragment,groups.keys())
 return float(values[0]['llm_judge_accuracy'])
current_metrics={
 'T3_eedi_semantic':acc('eedi_mining/open_taxonomy_generation'),
 'T3_synthetic_cross_semantic':acc('malrule/synthetic_cross_template'),
 'T3_synthetic_same_semantic':acc('malrule/synthetic_same_template'),
 'diagnosis_C4_D_specific':float(c4['semantic_D_specific']),
 'diagnosis_C4_swap_to_donor':float(c4['cells']['swap_to_donor']['semantic_equivalence_rate']),
 'diagnosis_C4_random_stable':float(c4['random_matches_either_stable_target_rate']),
 'diagnosis_C4_same_rule_retention':float(c4['same_rule_recipient_retention']),
 'diagnosis_C4_noop_retention':float(c4['noop_recipient_retention']),
}
baseline={'T3_eedi_semantic':0.0,'T3_synthetic_cross_semantic':0.60,
          'T3_synthetic_same_semantic':0.60,'diagnosis_C4_D_specific':0.20,
          'diagnosis_C4_swap_to_donor':0.30,'diagnosis_C4_random_stable':0.20}
out={'stage':'state8_current_query_causal_repair_comparison_v1','status':'complete',
     'semantic_judge_authoritative':True,'exact_match_authoritative':False,
     'current_query_updater_mse1':current_metrics,'legacy_checkpoint100_updater_mse1':baseline,
     'delta_vs_legacy':{key:current_metrics[key]-value for key,value in baseline.items()}}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat
log "query-aligned repair semantic evaluation complete"
