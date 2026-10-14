#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
variant="${BOOTSTRAP_VARIANT:-donor_only}"
if [[ "$variant" == donor_only ]]; then
  state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_pair8"
  config="$project_dir/amlt/feeds_m4_causal_bootstrap_pair8_20260731.yaml"
  base_experiment="recurrent-m4-pair8-overfit-sum-bootstrap-seed42-20260731"
  job="m4-pair8-overfit-sum-bootstrap-seed42"
  expected_branch="pair8_sum_bootstrap"
elif [[ "$variant" == native25 ]]; then
  state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_pair8_native25"
  config="$project_dir/amlt/feeds_m4_causal_bootstrap_pair8_native25_20260731.yaml"
  base_experiment="recurrent-m4-pair8-overfit-sum-bootstrap-native25-seed42-20260731"
  job="m4-pair8-overfit-sum-bootstrap-native25-seed42"
  expected_branch="pair8_sum_bootstrap_native25"
elif [[ "$variant" == balanced ]]; then
  state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_pair8_balanced"
  config="$project_dir/amlt/feeds_m4_causal_bootstrap_pair8_balanced_20260731.yaml"
  base_experiment="recurrent-m4-pair8-overfit-sum-bootstrap-balanced-seed42-20260731"
  job="m4-pair8-overfit-sum-bootstrap-balanced-seed42"
  expected_branch="pair8_sum_bootstrap_balanced"
elif [[ "$variant" == native25_balanced ]]; then
  state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/causal_bootstrap_pair8_native25_balanced"
  config="$project_dir/amlt/feeds_m4_causal_bootstrap_pair8_native25_balanced_20260731.yaml"
  base_experiment="recurrent-m4-pair8-overfit-sum-bootstrap-native25-balanced-seed42-20260731"
  job="m4-pair8-overfit-sum-bootstrap-native25-balanced-seed42"
  expected_branch="pair8_sum_bootstrap_native25_balanced"
else
  echo "unknown BOOTSTRAP_VARIANT=$variant" >&2
  exit 2
fi
expected_tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
expected_subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
account="csnewsandfeeds4150361735"
container="amulet"
remote_root="studentmodeling/diagnostics/world_model_v1_m4_${expected_branch}_seed42_20260731"
mkdir -p "$state_dir/download"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
if [[ -f "$state_dir/complete" && -f "$state_dir/gate.json" ]]; then
  python - "$state_dir/gate.json" <<'PY' || exit 1
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
assert x.get('pass') is True and all((x.get('checks') or {}).values()), x
PY
  exit 0
fi
rm -f "$state_dir/complete" "$state_dir/needs_attention"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

account_matches() {
  python -c '
import json, sys
value = json.loads(sys.argv[1])
assert value["tenant"] == sys.argv[2] and value["subscription"] == sys.argv[3], value
assert str(value["state"]).lower() == "enabled" and value["type"] == "user", value
' "$1" "$expected_tenant" "$expected_subscription" >/dev/null 2>&1
}

wait_auth() {
  local account
  while true; do
    heartbeat
    account="$(az account show --query '{tenant:tenantId,subscription:id,state:state,type:user.type}' -o json 2>/dev/null || true)"
    if ! account_matches "$account"; then
      az account set --subscription "$expected_subscription" >/dev/null 2>&1 \
        || { sleep 60; continue; }
      account="$(az account show --query '{tenant:tenantId,subscription:id,state:state,type:user.type}' -o json 2>/dev/null || true)"
    fi
    account_matches "$account" || { sleep 60; continue; }
    project_output="$(amlt project 2>&1)" || { sleep 60; continue; }
    grep -Eq '(^|[[:space:]])(project|PROJECT_NAME)[[:space:]]+v-yuejiali([[:space:]]|$)' \
      <<< "$project_output" || { sleep 60; continue; }
    log "fixed Feeds user authentication recovered"
    return 0
  done
}

wait_quota() {
  while true; do
    heartbeat
    local quota
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" "$state_dir/quota_before_submission.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
row = select_a100_80gb(parse_json_table(sys.argv[1])); q = row['quotas_(used/limit)']
available = int(q['standard']['limit']) - int(q['standard']['used'])
user_available = int(q['user_max']['limit']) - int(q['user_max'].get('used') or 0)
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps({
    'selected_row': row, 'required_units': 1,
    'standard_available': available, 'user_available': user_available,
}, indent=2, sort_keys=True) + '\n')
assert available >= 1 and user_available >= 1
PY
    then return 0; fi
    sleep 120
  done
}

submit_if_missing() {
  local experiment="$1" status rc class output
  exec 8>"$project_dir/results/feeds_submission_global.lock"
  flock 8
  amlt run "$config" ":$job" --dump -t Feeds > "$state_dir/${experiment}.dump.yaml" 2>&1 \
    || { flock -u 8; exec 8>&-; fail "bootstrap AMLT dump failed"; }
  python scripts/validate_m4_causal_bootstrap_pair8_amlt_config.py \
    --config "$state_dir/${experiment}.dump.yaml" --variant "$variant" \
    >> "$state_dir/controller.log" 2>&1 \
    || { flock -u 8; exec 8>&-; fail "bootstrap fail-closed preflight failed"; }
  status="$(amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  class="$(python - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)" || { flock -u 8; exec 8>&-; fail "could not classify bootstrap experiment"; }
  if [[ "$class" == missing ]]; then
    output="$(amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Pair-8 $variant bootstrap to symmetric causal curriculum diagnostic" 2>&1)"
    rc=$?
    printf '%s\n' "$output" > "$state_dir/${experiment}.submission.txt"
    (( rc == 0 )) || { flock -u 8; exec 8>&-; fail "bootstrap submission failed"; }
  elif [[ "$class" != exists ]]; then
    flock -u 8
    exec 8>&-
    fail "ambiguous bootstrap experiment state"
  fi
  flock -u 8
  exec 8>&-
}

download_passed() {
  local experiment="$1"
  amlt results download "$experiment" --status pass \
    --include 'pair_overfit_summary.json' \
    --include 'causal_validation_metrics.json' \
    --include 'training_metrics.json' \
    --include 'completion.json' \
    --include 'run_config.json' \
    --output "$state_dir/download" >> "$state_dir/controller.log" 2>&1
}

recover_authoritative_completion() {
  local experiment="$1" direct="$state_dir/direct_authoritative_download"
  local required=(pair_overfit_summary.json causal_validation_metrics.json training_metrics.json completion.json run_config.json)
  local name exists gate_rc
  for name in "${required[@]}" \
    final_adapter/adapter_model.safetensors recurrent_state.pt; do
    exists="$(az storage blob exists --account-name "$account" \
      --container-name "$container" --auth-mode login \
      --name "$remote_root/$name" --query exists -o tsv 2>/dev/null || true)"
    [[ "$exists" == true ]] || return 1
  done
  rm -rf "$direct"
  mkdir -p "$direct"
  for name in "${required[@]}"; do
    az storage blob download --account-name "$account" \
      --container-name "$container" --auth-mode login \
      --name "$remote_root/$name" --file "$direct/$name" --overwrite \
      >> "$state_dir/controller.log" 2>&1 || return 1
  done
  python scripts/summarize_m4_causal_bootstrap_pair8.py \
    --root "$direct" --output "$state_dir/gate.json" \
    --expected-branch "$expected_branch" \
    >> "$state_dir/controller.log" 2>&1
  gate_rc=$?
  python - "$state_dir" "$remote_root" "$experiment" <<'PY' || return 1
import hashlib, json, pathlib, sys
root, remote, experiment = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
direct = root / 'direct_authoritative_download'
completion = json.load(open(direct / 'completion.json', encoding='utf-8'))
gate = json.load(open(root / 'gate.json', encoding='utf-8'))
assert completion.get('status') == 'diagnostic_complete', completion
assert completion.get('micro_steps') == 200, completion
assert completion.get('causal_pairs_seen') == 200, completion
assert completion.get('unique_causal_pairs_seen') == 8, completion
assert gate.get('stage') == 'M4_causal_pair8_bootstrap', gate
gate_pass = gate.get('pass') is True and all((gate.get('checks') or {}).values())
receipt = {
    'status': 'recovered_from_authoritative_remote_checkpoint',
    'experiment': experiment,
    'remote_root': remote,
    'gate_pass': gate_pass,
    'metrics_sha256': {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(direct.glob('*.json'))
    },
}
(root / 'direct_authoritative_recovery.json').write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
PY
  if (( gate_rc == 0 )); then
    touch "$state_dir/complete"
    log "authoritative remote pair-8 gate passed; releasing post-completion copy job"
  else
    touch "$state_dir/scientific_gate_failed" "$state_dir/needs_attention"
    log "authoritative remote pair-8 run completed but missed the scientific gate; releasing GPU"
  fi
  amlt cancel "$experiment" ":$job" -y \
    > "$state_dir/post_completion_copy_cancel.txt" 2>&1 || true
  return 0
}

wait_auth
wait_quota
active_experiment="$base_experiment"
for recovery_wave in 0 1 2 3; do
  submit_if_missing "$active_experiment"
  while true; do
    heartbeat
    if recover_authoritative_completion "$active_experiment"; then
      exit 0
    fi
    status="$(amlt status "$active_experiment" 2>&1)"
    printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$active_experiment" "$status" >> "$state_dir/status.log"
    if grep -Eqi 'No authentication method succeeded|refreshing your credentials|management\.azure\.com|connection|timed out|permission' <<< "$status"; then
      wait_auth
      continue
    fi
    if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
      sleep 120
      continue
    fi
    if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
    then
      printf '%s\n' "$status" > "$state_dir/terminal_status.txt"
      download_passed "$active_experiment" || fail "bootstrap result download failed"
      python scripts/summarize_m4_causal_bootstrap_pair8.py \
        --root "$state_dir/download" --output "$state_dir/gate.json" \
        --expected-branch "$expected_branch" \
        >> "$state_dir/controller.log" 2>&1 \
        || fail "bootstrap causal-path gate failed"
      touch "$state_dir/complete"
      log "pair-8 causal bootstrap passed; full-data promotion is authorized"
      exit 0
    fi
    break
  done
  printf '%s\n' "$status" > "$state_dir/${active_experiment}.terminal_status.txt"
  amlt show "$active_experiment" > "$state_dir/${active_experiment}.show.txt" 2>&1 || true
  timeout 180s amlt logs view -n 1200 "$active_experiment" \
    > "$state_dir/${active_experiment}.failure.log" 2>&1 || true
  (( recovery_wave < 3 )) || fail "bootstrap exhausted recovery waves"
  wait_auth
  wait_quota
  active_experiment="recurrent-m4-pair8-overfit-sum-bootstrap-${variant}-recovery$((recovery_wave + 1))-seed42-20260731"
done
