#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/checkpoint_c4_curve_feeds"
superseded_marker="$project_dir/results/recurrent_m4_pipeline_20260730/original_pilot_superseded_by_causal_rescue"
base_config="amlt/feeds_m4_checkpoint_c4_probe_20260731.yaml"
remote_root="studentmodeling/checkpoints/world_model_v1_m4_generation_kd_trainable_seed42_fixedu4_v2_20260730"
account="csnewsandfeeds4150361735"
container="amulet"
# Round 1750 is the latest durable checkpoint after repeated external SIGTERM.
# Probe it before spending the remaining rounds so the behavioral-only route can
# be deprioritized early if C4 remains flat while the causal curriculum runs.
rounds=(1000 1750 2000 3000)
mkdir -p "$state_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
heartbeat() { date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail() { log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
rm -f "$state_dir/needs_attention"

experiment_class() {
  local name="$1" output rc
  output="$(amlt status --no-update-db "$name" 2>&1)"; rc=$?
  python - "$rc" "$output" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
}

blob_ready() {
  az storage blob exists --account-name "$account" --container-name "$container" \
    --auth-mode login --name "$remote_root/checkpoint-round-$1/checkpoint_completion.json" \
    --query exists -o tsv 2>/dev/null
}

wait_quota() {
  while true; do
    heartbeat
    quota="$(amlt --json-tables ti sing -t Feeds 2>&1)" || { sleep 120; continue; }
    if python - "$quota" <<'PY'
import sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
q = select_a100_80gb(parse_json_table(sys.argv[1]))['quotas_(used/limit)']
assert q['standard']['limit'] - q['standard']['used'] >= 1
assert q['user_max']['limit'] - (q['user_max'].get('used') or 0) >= 1
PY
    then return 0; fi
    sleep 120
  done
}

refresh_curve() {
  python - "$state_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.load(open(p, encoding='utf-8')) for p in sorted(
    root.glob('round-*/summary.json'), key=lambda p: int(p.parent.name.split('-')[-1]))]
tmp = root / 'curve_summary.json.partial'
tmp.write_text(json.dumps({'checkpoints': rows}, indent=2, sort_keys=True) + '\n')
tmp.replace(root / 'curve_summary.json')
PY
}

deadline=$(( $(date +%s) + 2592000 ))
for round in "${rounds[@]}"; do
  if (( round > 1750 )) && [[ -f "$superseded_marker" ]]; then
    refresh_curve
    touch "$state_dir/superseded_after_round1750" "$state_dir/complete"
    log "original M4 objective was superseded after round=1750; checkpoint curve closed without nonexistent later rounds"
    exit 0
  fi
  round_dir="$state_dir/round-$round"
  mkdir -p "$round_dir"
  [[ ! -f "$round_dir/summary.json" ]] || { log "round=$round already complete"; continue; }
  while [[ "$(blob_ready "$round")" != true ]]; do
    heartbeat
    (( $(date +%s) < deadline )) || fail "round=$round checkpoint wait exceeded 30 days"
    sleep 60
  done
  log "round=$round atomic checkpoint detected"
  config="amlt/feeds_m4_checkpoint_c4_round${round}_generated.yaml"
  python scripts/render_checkpoint_c4_probe.py --base "$base_config" --round "$round" --output "$config" \
    >> "$state_dir/controller.log" 2>&1 || fail "round=$round config render failed"
  python scripts/validate_checkpoint_c4_probe_amlt_config.py --config "$config" --round "$round" \
    >> "$state_dir/controller.log" 2>&1 || fail "round=$round config validation failed"
  wait_quota
  base_experiment="recurrent-m4-checkpoint${round}-c4-seed42-20260731"
  passed=false
  for recovery in 0 1 2 3; do
    experiment="$base_experiment"
    (( recovery == 0 )) || experiment="$base_experiment-recovery$recovery"
    class="$(experiment_class "$experiment")" || fail "round=$round ambiguous experiment state"
    if [[ "$class" == missing ]]; then
      exec 8>"$project_dir/results/feeds_submission_global.lock"
      flock 8
      amlt run "$config" --dump -t Feeds > "$round_dir/${experiment}.dump.yaml" 2>&1 \
        || { flock -u 8; fail "round=$round AMLT dump failed"; }
      python scripts/validate_checkpoint_c4_probe_amlt_config.py \
        --config "$round_dir/${experiment}.dump.yaml" --round "$round" \
        >> "$state_dir/controller.log" 2>&1 \
        || { flock -u 8; fail "round=$round dump validation failed"; }
      amlt run "$config" "$experiment" -t Feeds -y \
        --description "Frozen 150-pair recurrent C4 at atomic M4 round-$round; recovery=$recovery" \
        > "$round_dir/${experiment}.submission.txt" 2>&1 \
        || { flock -u 8; fail "round=$round submission failed"; }
      flock -u 8
    elif [[ "$class" != exists ]]; then
      fail "round=$round ambiguous pre-existing experiment"
    fi
    while true; do
      heartbeat
      status="$(amlt status "$experiment" 2>&1)"
      printf '[%s] %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$status" >> "$round_dir/status.log"
      if grep -Eqi 'No authentication method succeeded|refreshing your credentials|connection|timed out|permission' <<< "$status"; then sleep 120; continue; fi
      # Singularity can acknowledge experiment creation before its job row is
      # visible.  Treat that bounded control-plane window as non-terminal so a
      # valid upload is not duplicated into an unnecessary recovery experiment.
      if grep -Fqi 'No jobs found in the current experiment' <<< "$status"; then
        sleep 30
        continue
      fi
      if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then sleep 120; continue; fi
      if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 1) else 1)
PY
      then passed=true; break; fi
      amlt show "$experiment" > "$round_dir/${experiment}.show.txt" 2>&1 || true
      timeout 180s amlt logs view -n 1000 "$experiment" > "$round_dir/${experiment}.failure.log" 2>&1 || true
      break
    done
    [[ "$passed" == true ]] && break
  done
  [[ "$passed" == true ]] || fail "round=$round probe exhausted recovery waves"
  download="$round_dir/download"
  mkdir -p "$download"
  amlt results download "$experiment" --status pass --include recurrent_causal_gate.json \
    --include probe_scope.txt --output "$download" >> "$state_dir/controller.log" 2>&1 \
    || fail "round=$round result download failed"
  mapfile -t gates < <(find "$download" -type f -name recurrent_causal_gate.json)
  [[ ${#gates[@]} -eq 1 ]] || fail "round=$round expected one C4 result"
  python - "${gates[0]}" "$round_dir/summary.json" "$round" <<'PY' || fail "round summary failed"
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8')); s = x['summary']
assert x['raw_history_visible_to_decoder'] is False and x['persistent_history_kv_cache'] is False
row = {'round': int(sys.argv[3]), 'D_specific': s['D_specific'], 'D_damage': s['D_damage'],
       'clean_n': s['clean_n'], 'swap_flip_to_A': s['clean']['swap']['flip_to_A_rate'],
       'random_flip_to_A': s['clean']['random']['flip_to_A_rate'],
       'noop_flip_to_A': s['clean']['noop']['flip_to_A_rate'],
       'samemal_flip_to_A': s['clean']['samemal']['flip_to_A_rate'],
       'scientific_gate_pass': s['scientific_gate_pass']}
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps(row, indent=2, sort_keys=True) + '\n')
print(json.dumps(row, indent=2, sort_keys=True))
PY
  refresh_curve
  log "round=$round remote C4 complete"
done
touch "$state_dir/complete"
log "M4 remote checkpoint C4 curve complete"
