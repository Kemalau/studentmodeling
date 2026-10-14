#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_interleaved_full_eval_20260814"
config="$project/amlt/feeds_state8_interleaved_full_formal_eval_16way_20260814.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment=state8-interleaved-full-eval-seed8401-20260814
recovery_prefix=state8-interleaved-full-eval-recovery
recovery_suffix=-seed8401-20260814
max_recovery_waves=32

formal_jobs=(); for index in $(seq -w 0 15); do formal_jobs+=("state8-interleaved-full-formal-shard$index"); done
c4_jobs=(state8-interleaved-full-malrule-c4-seed13 state8-interleaved-full-malrule-c4-seed42 state8-interleaved-full-malrule-c4-seed97 state8-interleaved-full-eedi-c4-final150)
all_jobs=("${formal_jobs[@]}" "${c4_jobs[@]}")

mkdir -p "$state/preserved_results" "$state/t3_semantic/full1494"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

preserve_passed(){
  local experiment="$1" status
  source scripts/use_studentmodeling_feeds_auth.sh
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || return 1
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
 if str(row.get('status') or '').casefold()=='pass': print(str(row['job_name']).lstrip(':'))
PY
  )
  (( ${#passed[@]} > 0 )) || return 0
  timeout 1800 amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/preserved_results" </dev/null \
    >> "$state/controller.log" 2>&1 || return 1
  for job in "${passed[@]}"; do touch "$state/passed.$job"; done
}

submit_missing(){
  local experiment="$1"; shift
  local names=("$@") selectors=() dump="$state/${experiment}.preflight.dump.yaml"
  source scripts/use_studentmodeling_feeds_auth.sh
  for name in "${names[@]}"; do selectors+=(":$name"); done
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$dump" \
    2> "$state/${experiment}.preflight.stderr" || return 1
  "$python_bin" scripts/validate_state8_interleaved_full_eval_amlt.py \
    --config "$dump" --jobs "${names[@]}" >> "$state/controller.log" 2>&1 || return 1
  output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Recovery-only missing shards for frozen state8 seed8401 full formal/C4 evaluation; protocol and hashes unchanged" 2>&1)" || {
      printf '%s\n' "$output" > "$state/${experiment}.submission.txt"; return 1; }
  printf '%s\n' "$output" > "$state/${experiment}.submission.txt"
  printf '%s\n' "$experiment" > "$state/active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state/active_jobs.txt"
  log "submitted recovery experiment=$experiment jobs=${#names[@]}"
}

experiment="$base_experiment"; wave=0
printf '%s\n' "$experiment" > "$state/active_experiment.txt"
printf '%s\n' "${all_jobs[@]}" > "$state/active_jobs.txt"
while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json.partial"; mv "$state/latest_status.json.partial" "$state/latest_status.json"
  "$python_bin" - "$status" "$state/live_status.json" <<'PY'
import collections,json,pathlib,sys
rows=json.loads(sys.argv[1]); counts=collections.Counter(str(r.get('status') or '').casefold() for r in rows)
pathlib.Path(sys.argv[2]).write_text(json.dumps({'jobs':len(rows),'status_counts':dict(counts),'rows':rows},indent=2,sort_keys=True)+'\n')
PY
  if "$python_bin" - "$status" <<'PY'
import json,sys
active={'queued','preparing','running','pausing','resuming','paused'}
raise SystemExit(0 if any(str(r.get('status') or '').casefold() in active for r in json.loads(sys.argv[1])) else 1)
PY
  then sleep 60; continue; fi
  preserve_passed "$experiment" || { sleep 60; continue; }
  missing=(); for job in "${all_jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  if (( ${#missing[@]} == 0 )); then break; fi
  (( wave < max_recovery_waves )) || fail "exhausted recovery waves missing=${missing[*]}"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}${recovery_suffix}"
  until submit_missing "$experiment" "${missing[@]}"; do heartbeat; sleep 120; done
done

rm -rf "$state/download"; mkdir -p "$state/download"; cp -a "$state/preserved_results/." "$state/download/"
readarray -t predictions < <(find "$state/download" -type f -path '*/eval/formal_shard*/predictions.jsonl' | sort)
readarray -t metrics < <(find "$state/download" -type f -path '*/eval/formal_shard*/metrics.json' | sort)
readarray -t answer_c4 < <(find "$state/download" -type f -path '*/eval/c4_seed*/recurrent_causal_gate.json' | sort)
readarray -t diagnosis_predictions < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_seed*/judge_predictions.jsonl' | sort)
readarray -t diagnosis_evals < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_seed*/judge_eval.jsonl' | sort)
readarray -t diagnosis_metrics < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_seed*/metrics.json' | sort)
readarray -t eedi_predictions < <(find "$state/download" -type f -path '*/eval/eedi_same_question_c4_final150/judge_predictions.jsonl' | sort)
readarray -t eedi_evals < <(find "$state/download" -type f -path '*/eval/eedi_same_question_c4_final150/judge_eval.jsonl' | sort)
readarray -t eedi_metrics < <(find "$state/download" -type f -path '*/eval/eedi_same_question_c4_final150/metrics.json' | sort)
[[ ${#predictions[@]} -eq 16 && ${#metrics[@]} -eq 16 && ${#answer_c4[@]} -eq 3 \
   && ${#diagnosis_predictions[@]} -eq 3 && ${#diagnosis_evals[@]} -eq 3 \
   && ${#diagnosis_metrics[@]} -eq 3 && ${#eedi_predictions[@]} -eq 1 \
   && ${#eedi_evals[@]} -eq 1 && ${#eedi_metrics[@]} -eq 1 ]] \
  || fail "artifact cardinality drift formal=${#predictions[@]}/${#metrics[@]} answer_c4=${#answer_c4[@]} diagnosis=${#diagnosis_predictions[@]}/${#diagnosis_evals[@]}/${#diagnosis_metrics[@]} eedi=${#eedi_predictions[@]}/${#eedi_evals[@]}/${#eedi_metrics[@]}"

"$python_bin" scripts/merge_state8_final_eval.py --predictions "${predictions[@]}" \
  --metrics "${metrics[@]}" --eval-jsonl "$eval_jsonl" --c4 "${answer_c4[@]}" \
  --output-dir "$state/merged" >> "$state/controller.log" 2>&1 || fail "formal/C4 merge audit failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
exec 11>"$state/t3_semantic/judge.lock"; flock -x 11
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
bash scripts/wait_for_trapi_auth.sh
attempt=0
while [[ ! -s "$state/t3_semantic/full1494/metrics.json" ]]; do
  attempt=$((attempt+1))
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$state/merged/t3_predictions.jsonl" --eval-jsonl "$eval_jsonl" \
    --output-dir "$state/t3_semantic/full1494" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 \
    --judge-timeout-seconds 60 --judge-cache "$state/t3_semantic/cache.jsonl" \
    --judge-audit-samples 100 --seed 8401 >> "$state/t3_semantic/judge.log" 2>&1 && break
  (( attempt < 30 )) || fail "T3 semantic judge exhausted"
  sleep 120
done
kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; heartbeat_pid=""; flock -u 11

mkdir -p "$state/diagnosis_c4_semantic"
diagnosis_summaries=()
for index in 0 1 2; do
  seed="$("$python_bin" - "${diagnosis_metrics[$index]}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['pair_seed'])
PY
)"
  semantic_dir="$state/diagnosis_c4_semantic/seed$seed"
  mkdir -p "$semantic_dir"
  bash scripts/wait_for_trapi_auth.sh
  attempt=0
  while [[ ! -s "$semantic_dir/summary.json" ]]; do
    attempt=$((attempt+1)); heartbeat
    if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
        --predictions-jsonl "${diagnosis_predictions[$index]}" \
        --eval-jsonl "${diagnosis_evals[$index]}" --output-dir "$semantic_dir" \
        --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
        --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
        --judge-cache "$semantic_dir/cache.jsonl" --judge-audit-samples 100 \
        --seed "$seed" >> "$semantic_dir/judge.log" 2>&1; then
      "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
        --judge-redacted "$semantic_dir/predictions_redacted.jsonl" \
        --generation-metrics "${diagnosis_metrics[$index]}" \
        --output "$semantic_dir/summary.json" >> "$semantic_dir/controller.log" 2>&1 \
        || fail "MalRule diagnosis-C4 summary failed seed=$seed"
      break
    fi
    (( attempt < 30 )) || fail "MalRule diagnosis-C4 judge exhausted seed=$seed"
    sleep 120
  done
  diagnosis_summaries+=("$semantic_dir/summary.json")
done
"$python_bin" - "${diagnosis_summaries[@]}" "$state/diagnosis_c4_semantic/multiseed.json" <<'PY' \
  || fail "MalRule diagnosis-C4 multiseed summary failed"
import json,pathlib,statistics,sys
rows=[json.load(open(p)) for p in sys.argv[1:-1]]
seeds=[int(r['generation']['pair_seed']) for r in rows]
assert set(seeds)=={13,42,97},seeds
out={'stage':'state8_closed_diagnosis_c4_semantic_multiseed_v1',
 'semantic_judge_authoritative':True,'exact_label_role':'auxiliary_only_not_used_for_gate',
 'rows':rows,'seed_passes':sum(bool(r['pass']) for r in rows),
 'mean_semantic_D_specific':statistics.mean(r['semantic_D_specific'] for r in rows),
 'mean_swap_to_donor':statistics.mean(r['cells']['swap_to_donor']['semantic_equivalence_rate'] for r in rows),
 'mean_random_matches_either':statistics.mean(r['random_matches_either_stable_target_rate'] for r in rows),
 'pass':sum(bool(r['pass']) for r in rows)>=2}
pathlib.Path(sys.argv[-1]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

eedi_semantic="$state/eedi_same_question_c4_semantic"
mkdir -p "$eedi_semantic"
bash scripts/wait_for_trapi_auth.sh
attempt=0
while [[ ! -s "$eedi_semantic/summary.json" ]]; do
  attempt=$((attempt+1)); heartbeat
  if PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${eedi_predictions[0]}" --eval-jsonl "${eedi_evals[0]}" \
      --output-dir "$eedi_semantic" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 \
      --judge-timeout-seconds 60 --judge-cache "$eedi_semantic/cache.jsonl" \
      --judge-audit-samples 150 --seed 20260811 >> "$eedi_semantic/judge.log" 2>&1; then
    "$python_bin" scripts/summarize_state8_diagnosis_c4.py \
      --judge-redacted "$eedi_semantic/predictions_redacted.jsonl" \
      --generation-metrics "${eedi_metrics[0]}" --output "$eedi_semantic/summary.json" \
      >> "$eedi_semantic/controller.log" 2>&1 \
      || fail "Eedi same-question C4 summary failed"
    "$python_bin" - "$eedi_semantic/summary.json" <<'PY' \
      || fail "Eedi same-question C4 finalization failed"
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); x=json.load(open(p)); g=x.get('generation') or {}
assert g.get('stage')=='state8_eedi_same_question_diagnosis_c4_v1' and g.get('num_pairs')==150,x
assert g.get('same_question_query_identical') is True and g.get('student_response_visible_to_decoder') is False,x
assert g.get('raw_history_visible_to_decoder') is False and g.get('persistent_history_kv_cache') is False,x
x.update(stage='state8_eedi_same_question_diagnosis_c4_semantic_final_v1',
         exploratory_only=False,confirmatory=True,frozen_panel=True)
p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
    break
  fi
  (( attempt < 30 )) || fail "Eedi same-question C4 judge exhausted"
  sleep 120
done

"$python_bin" - "$state/merged/metrics.json" "$state/merged/c4_multiseed.json" \
  "$state/t3_semantic/full1494/metrics.json" \
  "$state/diagnosis_c4_semantic/multiseed.json" \
  "$state/eedi_same_question_c4_semantic/summary.json" \
  "$state/final_summary.json" <<'PY' || fail "final summary validation failed"
import json,pathlib,sys
formal,c4,t3,diagnosis_c4,eedi_c4=(json.load(open(p)) for p in sys.argv[1:6])
assert formal['sampled_rows']==21676 and formal['state_shape']==[64,8,5120],formal
assert formal['raw_history_visible_to_decoder'] is False and formal['persistent_history_kv_cache'] is False,formal
assert t3['generation_rows']==1494 and t3['judged_rows']==1494 and t3['judge_errors']==0,t3
assert diagnosis_c4['stage']=='state8_closed_diagnosis_c4_semantic_multiseed_v1' and len(diagnosis_c4['rows'])==3,diagnosis_c4
assert eedi_c4['stage']=='state8_eedi_same_question_diagnosis_c4_semantic_final_v1' and eedi_c4['confirmatory'] is True,eedi_c4
tasks={k.split('/')[0] for k in formal['metrics_by_group']}
assert tasks=={'predict_correctness','predict_response','diagnose_misconception','predict_prospective_error','predict_error_persistence'},tasks
out={'stage':'state8_interleaved_full_evaluation_summary_v1','status':'complete','evaluation_only':True,
 'promotion_claim':False,'formal':formal,'T3_semantic':t3,'answer_C4':c4,
 'diagnosis_C4':diagnosis_c4,'eedi_same_question_C4':eedi_c4,
 'artifact_sha256':formal['artifact_sha256']}
pathlib.Path(sys.argv[6]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state/complete"; heartbeat; log "full evaluation and semantic T3 complete"
