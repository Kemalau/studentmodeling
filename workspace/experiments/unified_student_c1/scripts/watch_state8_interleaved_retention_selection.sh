#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
remote="$project/results/state8_interleaved_retention_recovery3_20260813"
state="$project/results/state8_interleaved_retention_selection_20260813"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
job=state8-interleaved-retention-early-label-high-seed8401
remote_root=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401

mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"; trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$remote/remote_passed" ]]; do
  heartbeat
  [[ ! -f "$remote/remote_failed" ]] || fail "interleaved remote job failed"
  sleep 30
done

mapfile -t roots < <(find "$remote/download" -type d -name "$job")
[[ ${#roots[@]} -eq 1 ]] || fail "interleaved artifact root cardinality drift=${#roots[@]}"
root="${roots[0]}"
mapfile -t formal_metrics < <(find "$root" -type f -path '*/eval/formal_t3_300/metrics.json')
mapfile -t formal_predictions < <(find "$root" -type f -path '*/eval/formal_t3_300/predictions.jsonl')
mapfile -t malrule_predictions < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_predictions.jsonl')
mapfile -t malrule_eval < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/judge_eval.jsonl')
mapfile -t malrule_metrics < <(find "$root" -type f -path '*/eval/diagnosis_c4_50/metrics.json')
mapfile -t eedi_predictions < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_predictions.jsonl')
mapfile -t eedi_eval < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/judge_eval.jsonl')
mapfile -t eedi_metrics < <(find "$root" -type f -path '*/eval/eedi_same_question_c4_25/metrics.json')
[[ ${#formal_metrics[@]} -eq 1 && ${#formal_predictions[@]} -eq 1 \
   && ${#malrule_predictions[@]} -eq 1 && ${#malrule_eval[@]} -eq 1 \
   && ${#malrule_metrics[@]} -eq 1 && ${#eedi_predictions[@]} -eq 1 \
   && ${#eedi_eval[@]} -eq 1 && ${#eedi_metrics[@]} -eq 1 ]] \
  || fail "interleaved semantic artifact cardinality drift"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
judge(){
  local predictions="$1" evaluation="$2" destination="$3" audit="$4" attempt=0
  mkdir -p "$destination"
  bash scripts/wait_for_trapi_auth.sh
  while [[ ! -s "$destination/metrics.json" ]]; do
    heartbeat; attempt=$((attempt+1))
    PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$predictions" --eval-jsonl "$evaluation" \
      --output-dir "$destination" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$destination/cache.jsonl" --judge-audit-samples "$audit" \
      --seed 8401 >> "$destination/judge.log" 2>&1 && break
    (( attempt < 30 )) || fail "semantic judge exhausted destination=$destination"
    sleep 120
  done
}

judge "${formal_predictions[0]}" "$eval_jsonl" "$state/t3_semantic" 75
judge "${malrule_predictions[0]}" "${malrule_eval[0]}" "$state/malrule_c4" 100
judge "${eedi_predictions[0]}" "${eedi_eval[0]}" "$state/eedi_c4" 50
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/malrule_c4/predictions_redacted.jsonl" \
  --generation-metrics "${malrule_metrics[0]}" \
  --output "$state/malrule_c4/summary.json" \
  >> "$state/controller.log" 2>&1 || fail "MalRule C4 summary failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/eedi_c4/predictions_redacted.jsonl" \
  --generation-metrics "${eedi_metrics[0]}" \
  --output "$state/eedi_c4/summary.json" \
  >> "$state/controller.log" 2>&1 || fail "Eedi C4 summary failed"

"$python_bin" - "${formal_metrics[0]}" "$state/t3_semantic/metrics.json" \
  "$state/malrule_c4/summary.json" "$state/eedi_c4/summary.json" \
  "$state/candidate_summary.json" "$job" "$remote_root" <<'PY' \
  || fail "interleaved candidate summary failed"
import json,pathlib,sys
formal,t3,malrule,eedi=(json.load(open(path)) for path in sys.argv[1:5])
metadata={
  'job':sys.argv[6], 'seed':8401, 'remote_root':sys.argv[7],
  'finalfit_variant':'same_question_repair_promoted',
  'repair_branch':'early_label_high', 'source_anchor_mode':'full_mse1',
}
out={
  'stage':'state8_eedi_same_question_repair_semantic_summary_v1',
  'status':'complete','semantic_judge_authoritative':True,
  'exact_match_authoritative':False,'formal_metrics':formal,
  't3_semantic':t3,'malrule_diagnosis_c4':malrule,
  'eedi_same_question_diagnosis_c4':eedi,'candidate_metadata':metadata,
  'interleaved_retention':True,
}
pathlib.Path(sys.argv[5]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

set +e
"$python_bin" scripts/select_state8_eedi_same_question_candidate.py \
  --candidate repair_interleaved_retention_seed8401 "$state/candidate_summary.json" \
  --output "$state/selection.json" > "$state/selection.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$state/scientific_gate_pass"
elif (( rc == 2 )); then touch "$state/scientific_gate_failed"
else fail "interleaved candidate selector crashed rc=$rc"; fi
touch "$state/complete"; heartbeat
log "interleaved candidate semantically evaluated rc=$rc"
