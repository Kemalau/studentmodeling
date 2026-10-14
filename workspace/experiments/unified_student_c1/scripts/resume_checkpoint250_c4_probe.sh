#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/checkpoint250_c4_probe"
config="amlt/feeds_m4_checkpoint_c4_probe_20260731.yaml"
experiment="recurrent-m4-checkpoint250-c4-seed42-20260731"
prefix="studentmodeling/checkpoints/world_model_v1_m4_generation_kd_trainable_seed42_fixedu4_v2_20260730/checkpoint-round-250"
account="csnewsandfeeds4150361735"
container="amulet"
mkdir -p "$state_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
rm -f "$state_dir/needs_attention"

blob_exists() {
  az storage blob exists --account-name "$account" --container-name "$container" \
    --name "$1" --auth-mode login --query exists -o tsv 2>/dev/null
}

deadline=$(( $(date +%s) + 2592000 ))
while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  ready=true
  for file in adapter/adapter_model.safetensors recurrent_state.pt checkpoint_completion.json; do
    [[ "$(blob_exists "$prefix/$file")" == true ]] || ready=false
  done
  [[ "$ready" == true ]] && break
  [[ ! -f results/recurrent_m4_pipeline_20260730/needs_attention ]] \
    || fail "M4 pipeline exhausted recovery before checkpoint round 250"
  (( $(date +%s) < deadline )) || fail "checkpoint round 250 wait exceeded 30 days"
  sleep 60
done
log "atomic checkpoint round 250 is complete"

while true; do
  quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
  if python - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1]))
quota = row['quotas_(used/limit)']
available = int(quota['standard']['limit']) - int(quota['standard']['used'])
user_available = int(quota['user_max']['limit']) - int(quota['user_max'].get('used') or 0)
assert available >= 1 and user_available >= 1
PY
  then
    printf '%s\n' "$quota" > "$state_dir/quota_before_submission.json"
    break
  fi
  sleep 120
done

amlt run "$config" --dump -t Feeds > "$state_dir/preflight_dump.yaml" 2>&1 \
  || fail "C4 probe AMLT dump failed"
python scripts/validate_checkpoint_c4_probe_amlt_config.py \
  --config "$state_dir/preflight_dump.yaml" >> "$state_dir/controller.log" 2>&1 \
  || fail "C4 probe preflight failed"

base_experiment="$experiment"
probe_passed=false
for recovery_wave in 0 1 2 3; do
  if (( recovery_wave > 0 )); then
    experiment="${base_experiment}-recovery${recovery_wave}"
  fi
  set +e
  status="$(amlt status --no-update-db "$experiment" 2>&1)"
  rc=$?
  set -e
  class="$(python - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
  )" || fail "C4 probe status is ambiguous"
  if [[ "$class" == missing ]]; then
    exec 8>"$project_dir/results/feeds_submission_global.lock"
    flock 8
    amlt run "$config" "$experiment" -t Feeds -y \
      --description "Diagnostic-only frozen C4 trend at atomic M4 round-250 checkpoint; recovery_wave=$recovery_wave; final 150-pair C4 remains mandatory" \
      > "$state_dir/${experiment}.submission.txt" 2>&1 \
      || { flock -u 8; fail "C4 probe submission failed"; }
    flock -u 8
  elif [[ "$class" != exists ]]; then
    fail "C4 probe pre-existing state is ambiguous"
  fi

  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
    status="$(amlt status "$experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" \
      >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then
      probe_passed=true
      break
    fi
    printf '%s\n' "$status" > "$state_dir/${experiment}.terminal_status.txt"
    amlt show "$experiment" > "$state_dir/${experiment}.show.txt" 2>&1 || true
    amlt logs view -n 1200 "$experiment" > "$state_dir/${experiment}.failure.log" 2>&1 || true
    log "checkpoint-250 C4 probe attempt failed: $experiment"
    break
  done
  [[ "$probe_passed" == true ]] && break
done
[[ "$probe_passed" == true ]] || fail "checkpoint-250 C4 probe exhausted three recovery waves"

download="$state_dir/download"
mkdir -p "$download"
amlt results download "$experiment" --status pass --include 'recurrent_causal_gate.json' \
  --include 'probe_scope.txt' --output "$download" >> "$state_dir/controller.log" 2>&1 \
  || fail "C4 probe result download failed"
mapfile -t gates < <(find "$download" -type f -name recurrent_causal_gate.json)
[[ ${#gates[@]} -eq 1 ]] || fail "Expected one checkpoint C4 result"
python - "${gates[0]}" "$state_dir/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8'))
assert value.get('raw_history_visible_to_decoder') is False, value
assert value.get('persistent_history_kv_cache') is False, value
assert value.get('intervention_target') == 'final_continuous_recurrent_state', value
summary = value['summary']
payload = {
    'stage': 'diagnostic_checkpoint_round_250',
    'diagnostic_only': True,
    'formal_final_c4_required': True,
    'D_specific': summary['D_specific'],
    'D_damage': summary['D_damage'],
    'clean_n': summary['clean_n'],
    'scientific_gate_pass_at_checkpoint': summary['scientific_gate_pass'],
}
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
touch "$state_dir/complete"
log "checkpoint-250 C4 diagnostic complete"
