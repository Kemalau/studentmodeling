#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=promoted-round1000-c4-n50-seed42-20260804
job=promoted-round1000-c4-n50-seed42
remote_state="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1000_c4_feeds"
target_state="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/local_round1000_c4"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$remote_state" "$target_state"
cd "$project_dir"
exec 9>"$remote_state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$remote_state/controller.active.pid"
trap 'rm -f "$remote_state/controller.active.pid"' EXIT

heartbeat(){
  date -u +%Y-%m-%dT%H:%M:%SZ > "$remote_state/controller.heartbeat"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$target_state/controller.heartbeat"
}
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$remote_state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$remote_state/needs_attention" "$target_state/needs_attention"; exit 1; }

rm -f "$remote_state/complete" "$remote_state/needs_attention"
while true; do
  heartbeat
  status="$(timeout 45 amlt status "$experiment" 2>&1)" || { sleep 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$remote_state/status.log"
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
  then
    break
  fi
  amlt show "$experiment" > "$remote_state/show.txt" 2>&1 || true
  amlt logs view -n 500 "$experiment" ":$job" > "$remote_state/failure.log" 2>&1 || true
  fail "round-1000 C4 reached terminal non-pass state"
done

rm -rf "$remote_state/download"
mkdir -p "$remote_state/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.txt' --output "$remote_state/download" \
  >> "$remote_state/controller.log" 2>&1 \
  || fail "could not download round-1000 C4 results"

readarray -t summaries < <(find "$remote_state/download" -type f -path '*/eval/c4/summary.json')
readarray -t gates < <(find "$remote_state/download" -type f -path '*/eval/c4/recurrent_causal_gate.json')
readarray -t exact_audits < <(find "$remote_state/download" -type f -path '*/eval/c4/exact_resume_audit.json')
readarray -t recipe_audits < <(find "$remote_state/download" -type f -path '*/eval/c4/promoted_recipe_audit.json')
[[ ${#summaries[@]} -eq 1 && ${#gates[@]} -eq 1 && ${#exact_audits[@]} -eq 1 && ${#recipe_audits[@]} -eq 1 ]] \
  || fail "downloaded round-1000 result cardinality drift"

"$python_bin" - "${summaries[0]}" "${gates[0]}" "${exact_audits[0]}" "${recipe_audits[0]}" <<'PY' \
  || fail "downloaded round-1000 C4 audit failed"
import json,sys
summary=json.load(open(sys.argv[1])); gate=json.load(open(sys.argv[2]))['summary']
exact=json.load(open(sys.argv[3])); recipe=json.load(open(sys.argv[4]))
assert summary['stage']=='layerwise_checkpoint_c4' and summary['checkpoint_round']==1000,summary
assert summary['diagnostic_only'] is True and summary['not_a_formal_gate'] is True,summary
assert gate['all_pairs_n']==50 and gate['clean_n']==summary['clean_n'],(gate,summary)
assert exact['status']=='passed' and exact['exact_resume_ready'] is True,exact
assert exact['checkpoint_round']==1000 and exact['continuation_source_round']==750,exact
assert exact['planned_optimizer_steps']==15491,exact
assert recipe['status']=='passed' and recipe['checkpoint_round']==1000,recipe
assert recipe['candidate_id']=='eedi_full_open_t3__state_path__lr2e-06',recipe
PY

cp "${summaries[0]}" "$target_state/summary.json"
cp "${gates[0]}" "$target_state/c4/recurrent_causal_gate.json"
cp "${exact_audits[0]}" "$target_state/exact_resume_audit.json"
cp "${recipe_audits[0]}" "$target_state/promoted_recipe_audit.json"
rm -f "$target_state/needs_attention"
touch "$target_state/complete" "$remote_state/complete"
heartbeat
log "promoted round-1000 C4 passed remotely and was audited into the canonical state directory"
