#!/usr/bin/env bash
set -uo pipefail
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
parent="$project/results/state8_diagnosis_first_matched_curriculum_20260810"
state="$project/results/state8_diagnosis_first_matched_final_diagnosis_c4_corrected_20260810"
template="$project/amlt/feeds_state8_matched_final_diagnosis_c4_corrected_20260810.yaml"
rendered="$project/amlt/feeds_state8_matched_final_diagnosis_c4_corrected_generated_20260810.yaml"
launcher="$project/scripts/run_state8_matched_final_diagnosis_c4_corrected.sh"
account=csnewsandfeeds4150361735; container=amulet
root=studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4
receipt_blob="$root/metrics/immutable_receipt.json"
base=state8-matched-final-diagnosis-c4-corrected-20260810
recovery_prefix=state8-matched-final-diagnosis-c4-corrected-recovery
job=state8-matched-final-diagnosis-c4-corrected
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
max_waves=16
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }; trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
sleep_unlocked(){ sleep "$1" 9>&-; }
rm -f "$state/needs_attention"

while [[ "$(az storage blob exists --auth-mode login --account-name "$account" --container-name "$container" --name "$receipt_blob" --query exists -o tsv 2>/dev/null)" != true ]]; do
  heartbeat; [[ ! -f "$parent/needs_attention" ]] || fail 'parent matched curriculum needs attention'; sleep_unlocked 60
done
az storage blob download --auth-mode login --account-name "$account" --container-name "$container" --name "$receipt_blob" --file "$state/immutable_receipt.json" --overwrite -o none || fail 'immutable receipt download failed'
"$python_bin" scripts/materialize_state8_matched_final_diagnosis_c4_corrected.py --template "$template" --receipt "$state/immutable_receipt.json" --output "$rendered" --launcher "$launcher" > "$state/materialize.json" || fail 'corrected C4 materialization failed'
"$python_bin" scripts/validate_state8_matched_final_diagnosis_c4_corrected_amlt.py --config "$rendered" >> "$state/controller.log" 2>&1 || fail 'corrected C4 config validation failed'

submit(){
  local experiment="$1" output rc current class
  exec 8>"$project/results/feeds_submission_global.lock"; flock -x 8
  amlt run "$rendered" --dump -t Feeds > "$state/$experiment.preflight.yaml" 2> "$state/$experiment.preflight.stderr" || { flock -u 8; return 1; }
  "$python_bin" scripts/validate_state8_matched_final_diagnosis_c4_corrected_amlt.py --config "$state/$experiment.preflight.yaml" >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }
  amlt --json-tables ti sing -t Feeds > "$state/$experiment.quota.json" || { flock -u 8; return 1; }
  output="$(amlt run "$rendered" "$experiment" -t Feeds -y --description 'Corrected matched-final diagnosis C4; explicit Malrule taxonomy; strict state-only; hash-bound immutable checkpoint' 2>&1)"; rc=$?
  printf '%s\n' "$output" > "$state/$experiment.submission.txt"
  if (( rc != 0 )); then
    current="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
    class="$("$python_bin" - "$rc" "$current" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || { flock -u 8; return 1; }
    [[ "$class" == exists ]] || { flock -u 8; return 1; }
  fi
  flock -u 8; printf '%s\n' "$experiment" > "$state/active_experiment.txt"
}

experiment="$base"; wave=0
if [[ -s "$state/active_experiment.txt" ]]; then experiment="$(<"$state/active_experiment.txt")"; [[ "$experiment" =~ recovery([0-9]+)-20260810$ ]] && wave="${BASH_REMATCH[1]}"; fi
while true; do
  if [[ ! -s "$state/active_experiment.txt" ]]; then until submit "$experiment"; do heartbeat; sleep_unlocked 120; done; log "submitted corrected final diagnosis C4 experiment=$experiment"; fi
  heartbeat; status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep_unlocked 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],1) else 1)
PY
  then break; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep_unlocked 60; continue; fi
  if grep -Eqi '\bpaused\b' <<< "$status"; then timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true; sleep_unlocked 60; continue; fi
  timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
  timeout 180 amlt logs view -n 3000 "$experiment" > "$state/$experiment.failure.log" 2>&1 || true
  (( wave < max_waves )) || fail 'corrected final diagnosis C4 exhausted recovery waves'
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}-20260810"; rm -f "$state/active_experiment.txt"; log "recovering corrected final diagnosis C4 wave=$wave"
done
rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --output "$state/download" >> "$state/controller.log" 2>&1 || fail 'corrected final C4 result download failed'
mapfile -t pred < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_20_corrected/judge_predictions.jsonl')
mapfile -t evals < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_20_corrected/judge_eval.jsonl')
mapfile -t metrics < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_20_corrected/metrics.json')
[[ ${#pred[@]} -eq 1 && ${#evals[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] || fail 'corrected final C4 artifact cardinality drift'
mkdir -p "$state/c4_semantic"; [[ -x "$judge_python" ]] || fail 'TRAPI judge unavailable'
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py --predictions-jsonl "${pred[0]}" --eval-jsonl "${evals[0]}" --output-dir "$state/c4_semantic" --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 67 >> "$state/c4_semantic/judge.log" 2>&1 || fail 'corrected final C4 semantic judge failed'
"$python_bin" scripts/summarize_state8_diagnosis_c4.py --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" --generation-metrics "${metrics[0]}" --output "$state/c4_semantic/summary.json" >> "$state/controller.log" 2>&1 || fail 'corrected final C4 summary failed'
touch "$state/complete"; heartbeat; log 'corrected matched-final diagnosis C4 complete'
