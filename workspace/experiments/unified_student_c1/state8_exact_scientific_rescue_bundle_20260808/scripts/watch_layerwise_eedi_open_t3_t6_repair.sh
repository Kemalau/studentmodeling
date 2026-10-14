#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/eedi_open_t3_t6_repair"
config="$project_dir/amlt/feeds_layerwise_eedi_open_t3_t6_repair_20260803.yaml"
validator=scripts/validate_layerwise_eedi_open_t3_t6_repair_amlt.py
base_experiment=layerwise-eedi-open-t3-t6-repair-seed42-20260803
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
train_jsonl="$project_dir/data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl"
jobs=(
  layerwise-eedi-open-t3-t6-repair-balanced-lr5e7-seed42
  layerwise-eedi-open-t3-t6-repair-balanced-lr1e6-seed42
  layerwise-eedi-open-t3-t6-repair-strong-lr5e7-seed42
)
mkdir -p "$state_dir/preserved" "$state_dir/passed_receipts" \
  "$state_dir/retired_arms" "$state_dir/t3_semantic"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/complete" "$state_dir/scientific_gate_pass" "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

wait_quota(){
  local quota
  while true; do
    heartbeat
    quota="$(timeout 90 amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table,select_a100_80gb
q=select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
assert int(q['standard']['limit'])-int(q['standard']['used'])>=4
assert int(q['user_max']['limit'])-int(q['user_max'].get('used') or 0)>=4
PY
    then return 0; fi
    sleep 120
  done
}

preserve_passed(){
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include '*.json' --include 'predictions.jsonl' \
    --include 'predictions_redacted.jsonl' --output "$state_dir/preserved" \
    >> "$state_dir/controller.log" 2>&1
}

declare -A arm_experiment waves
arm_map="$state_dir/arm_experiments.tsv"
if [[ -s "$arm_map" ]]; then
  while IFS=$'\t' read -r job experiment wave; do
    [[ -n "$job" && -n "$experiment" ]] || continue
    arm_experiment["$job"]="$experiment"; waves["$job"]="${wave:-0}"
  done < "$arm_map"
else
  for job in "${jobs[@]}"; do arm_experiment["$job"]="$base_experiment"; waves["$job"]=0; done
fi

save_arm_map(){
  local tmp="$arm_map.tmp.$$" job
  : > "$tmp"
  for job in "${jobs[@]}"; do
    printf '%s\t%s\t%s\n' "$job" "${arm_experiment[$job]}" "${waves[$job]}" >> "$tmp"
  done
  mv "$tmp" "$arm_map"
}
save_arm_map

submit_recovery(){
  local job="$1" wave="$2" experiment output rc
  experiment="layerwise-eedi-open-t3-t6-repair-${job#layerwise-eedi-open-t3-t6-repair-}-recovery${wave}-20260803"
  wait_quota
  exec 8>"$project_dir/results/feeds_submission_global.lock"; flock 8
  amlt run "$config" ":$job" --dump -t Feeds \
    > "$state_dir/${experiment}.preflight.dump.yaml" \
    2> "$state_dir/${experiment}.preflight.stderr" \
    || { flock -u 8; return 1; }
  python "$validator" --config "$state_dir/${experiment}.preflight.dump.yaml" \
    --jobs "$job" >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; return 1; }
  output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
    --description "T6-repair recovery=$wave arm=$job; scientific recipe unchanged" 2>&1)"; rc=$?
  printf '[%s] rc=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$output" \
    > "$state_dir/${experiment}.submission.txt"
  flock -u 8
  (( rc == 0 )) || return 1
  arm_experiment["$job"]="$experiment"; save_arm_map
}

while true; do
  heartbeat
  transient=false
  for job in "${jobs[@]}"; do
    receipt="$state_dir/passed_receipts/$job.complete"
    [[ -f "$receipt" ]] && continue
    [[ -f "$state_dir/retired_arms/$job.retired" ]] && continue
    experiment="${arm_experiment[$job]}"
    status="$(timeout 90 amlt status "$experiment" ":$job" 2>&1)" || { transient=true; break; }
    printf '[%s] experiment=%s job=%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$experiment" "$job" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission|TooManyRequests|QuotaExceeded|InternalServerError|ServiceUnavailable' <<< "$status"; then
      transient=true; break
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
    then
      preserve_passed "$experiment" || { transient=true; break; }
      printf 'experiment=%s\njob=%s\npassed_at=%s\n' "$experiment" "$job" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$receipt"
      log "recorded Passed arm=$job experiment=$experiment"
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_active_job
raise SystemExit(0 if has_active_job(sys.argv[1]) else 1)
PY
    then continue; fi
    mapfile -t failed < <(python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import terminal_nonpass_job_names
print(*terminal_nonpass_job_names(sys.argv[1]),sep='\n')
PY
    )
    [[ ${#failed[@]} -eq 1 && "${failed[0]}" == "$job" ]] \
      || fail "unrecognized terminal state experiment=$experiment arm=$job"
    timeout 180 amlt logs view -n 3000 "$experiment" ":$job" \
      > "$state_dir/${experiment}.${job}.failure.log" 2>&1 || true
    waves["$job"]=$((waves["$job"]+1))
    (( waves["$job"] <= 6 )) || fail "arm exhausted six recovery waves: $job"
    preserve_passed "$experiment" || true
    submit_recovery "$job" "${waves[$job]}" \
      || fail "recovery submission failed arm=$job wave=${waves[$job]}"
    log "recovered arm=$job wave=${waves[$job]} experiment=${arm_experiment[$job]}"
  done
  [[ "$transient" == true ]] && { sleep 120; continue; }
  all_terminal=true
  for job in "${jobs[@]}"; do
    [[ -f "$state_dir/passed_receipts/$job.complete" \
        || -f "$state_dir/retired_arms/$job.retired" ]] || all_terminal=false
  done
  [[ "$all_terminal" == true ]] && break
  sleep 120
done

passed_count=0
for job in "${jobs[@]}"; do
  [[ -f "$state_dir/passed_receipts/$job.complete" ]] && passed_count=$((passed_count+1))
done
if (( passed_count == 0 )); then
  python - "$state_dir" "$state_dir/selection.json" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
retired=[]
for path in sorted((root/'retired_arms').glob('*.retired')):
    fields={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if '=' in line:
            key,value=line.split('=',1); fields[key]=value
    retired.append({'job':path.name.removesuffix('.retired'),**fields})
payload={
 'stage':'layerwise_eedi_open_t3_t6_repair_selection_v1',
 'diagnostic_only':True,
 'selection_complete':False,
 'any_all_gates_pass':False,
 'selected_candidate':None,
 'passed_arm_count':0,
 'retired_arms':retired,
 'reason':'all_remaining_diagnostic_arms_retired_to_prioritize_promoted_exact_epoch_fit',
}
path=pathlib.Path(sys.argv[2]); temporary=path.with_suffix(path.suffix+'.partial')
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
temporary.replace(path)
PY
  touch "$state_dir/complete"
  rm -f "$state_dir/scientific_gate_pass" "$state_dir/needs_attention"
  heartbeat
  log "T6-repair retired without a selectable scientific result"
  exit 0
fi

rm -rf "$state_dir/download"; mkdir -p "$state_dir/download"
cp -a "$state_dir/preserved/." "$state_dir/download/"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
backend=responses; extra="$project_dir/scripts"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  python_bin=/home/v-yuejiali/fkft/.venv/bin/python
  backend=trapi; extra="$project_dir/scripts:/home/v-yuejiali/fkft/src"
fi
summaries=()
for job in "${jobs[@]}"; do
  if [[ -f "$state_dir/retired_arms/$job.retired" ]]; then
    log "excluding explicitly retired diagnostic arm=$job from incomplete comparison"
    continue
  fi
  mapfile -t predictions < <(find "$state_dir/download" -type f \
    -path "*${job}*/eval/formal_screen/predictions.jsonl" | sort)
  [[ ${#predictions[@]} -eq 1 ]] || fail "expected one formal prediction file arm=$job found=${#predictions[@]}"
  prediction="${predictions[0]}"
  job_root="$(dirname "$(dirname "$(dirname "$prediction")")")"
  if [[ "$job" == *balanced-lr1e6* ]]; then lr=1e-6; slug=balanced_lr1e6
  elif [[ "$job" == *balanced-lr5e7* ]]; then lr=5e-7; slug=balanced_lr5e7
  else lr=5e-7; slug=strong_lr5e7; fi
  output="$state_dir/t3_semantic/$slug"; mkdir -p "$output"; heartbeat
  PYTHONPATH="$extra" "$python_bin" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$prediction" --eval-jsonl "$eval_jsonl" \
    --output-dir "$output" --judge-backend "$backend" --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$state_dir/t3_semantic/cache.jsonl" \
    --judge-audit-samples 75 --seed 42 >> "$output/live.log" 2>&1 \
    || fail "T6-repair T3 judge failed arm=$job"
  python scripts/summarize_t3_seen_unseen.py \
    --train-jsonl "$train_jsonl" --eval-jsonl "$eval_jsonl" \
    --predictions-jsonl "$prediction" \
    --judgments-jsonl "$output/predictions_redacted.jsonl" \
    --output "$output/seen_unseen.json" >> "$output/live.log" 2>&1 \
    || fail "T6-repair seen/unseen failed arm=$job"
  summary="$state_dir/$slug.summary.json"
  python scripts/summarize_layerwise_single_task_retention.py \
    --root "$job_root" --t3-metrics "$output/metrics.json" \
    --branch eedi_full_open_t3 --loss-profile state_path \
    --learning-rate "$lr" --checkpoint-round 4500 --output "$summary" \
    >> "$state_dir/controller.log" 2>&1 \
    || fail "T6-repair candidate summary failed arm=$job"
  summaries+=("$summary")
done
python scripts/summarize_layerwise_eedi_open_t3_t6_repair.py \
  --summaries "${summaries[@]}" --output "$state_dir/selection.json" \
  >> "$state_dir/controller.log" 2>&1 || fail "T6-repair selection failed"
touch "$state_dir/complete"
python - "$state_dir/selection.json" <<'PY' && touch "$state_dir/scientific_gate_pass"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if x.get('selection_complete') is True and x.get('any_all_gates_pass') is True else 1)
PY
heartbeat
log "T6-repair evaluation complete selection_complete=$(python - "$state_dir/selection.json" <<'PY'
import json,sys
print(str(json.load(open(sys.argv[1],encoding='utf-8'))['selection_complete']).lower())
PY
)"
