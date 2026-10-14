#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/checkpoint_c4_curve_local"
current_probe_dir="$project_dir/results/recurrent_m4_pipeline_20260730/checkpoint250_c4_probe/local_probe"
remote_root="studentmodeling/checkpoints/world_model_v1_m4_generation_kd_trainable_seed42_fixedu4_v2_20260730"
account="csnewsandfeeds4150361735"
container="amulet"
rounds=(500 1000 2000 3000)
mkdir -p "$state_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
cd "$project_dir"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "$state_dir/controller.log"
}

fail() {
  log "ERROR: $*"
  touch "$state_dir/needs_attention"
  exit 1
}

heartbeat() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
}

blob_exists() {
  az storage blob exists --account-name "$account" --container-name "$container" \
    --name "$1" --auth-mode login --query exists -o tsv 2>/dev/null
}

download_blob() {
  local name="$1" destination="$2"
  local attempt
  for attempt in 1 2 3 4 5; do
    if az storage blob download --account-name "$account" --container-name "$container" \
      --name "$name" --auth-mode login --file "$destination" --overwrite \
      >> "$state_dir/download.log" 2>&1; then
      return 0
    fi
    log "blob download retry=$attempt name=$name"
    sleep 60
  done
  return 1
}

wait_for_gpu() {
  while true; do
    heartbeat
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true)"
    [[ -z "$active" ]] && return 0
    sleep 60
  done
}

refresh_curve() {
  python - "$state_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
points = []
for path in sorted(root.glob('round-*/summary.json'), key=lambda p: int(p.parent.name.split('-')[-1])):
    points.append(json.load(open(path, encoding='utf-8')))
temporary = root / 'curve_summary.json.partial'
temporary.write_text(json.dumps({'checkpoints': points}, indent=2, sort_keys=True) + '\n')
temporary.replace(root / 'curve_summary.json')
PY
}

rm -f "$state_dir/needs_attention"
deadline=$(( $(date +%s) + 2592000 ))
for round in "${rounds[@]}"; do
  round_dir="$state_dir/round-$round"
  checkpoint="$round_dir/checkpoint"
  prefix="$remote_root/checkpoint-round-$round"
  mkdir -p "$checkpoint/adapter"
  if [[ -f "$round_dir/summary.json" ]]; then
    log "round=$round already complete"
    continue
  fi

  while true; do
    heartbeat
    ready=true
    for file in adapter/adapter_model.safetensors adapter/adapter_config.json \
      recurrent_state.pt checkpoint_completion.json; do
      [[ "$(blob_exists "$prefix/$file")" == true ]] || ready=false
    done
    [[ "$ready" == true ]] && break
    (( $(date +%s) < deadline )) || fail "round=$round checkpoint wait exceeded 30 days"
    sleep 60
  done
  log "round=$round atomic checkpoint detected"

  download_blob "$prefix/adapter/adapter_model.safetensors" \
    "$checkpoint/adapter/adapter_model.safetensors" || fail "round=$round adapter download failed"
  download_blob "$prefix/adapter/adapter_config.json" \
    "$checkpoint/adapter/adapter_config.json" || fail "round=$round adapter config download failed"
  download_blob "$prefix/recurrent_state.pt" "$checkpoint/recurrent_state.pt" \
    || fail "round=$round recurrent state download failed"
  download_blob "$prefix/checkpoint_completion.json" "$checkpoint/checkpoint_completion.json" \
    || fail "round=$round completion receipt download failed"

  python - "$checkpoint" "$round" <<'PY' \
    > "$round_dir/integrity.json" || fail "round=$round checkpoint integrity failed"
import hashlib, json, sys
from pathlib import Path
root, expected_round = Path(sys.argv[1]), int(sys.argv[2])
receipt = json.load(open(root / 'checkpoint_completion.json', encoding='utf-8'))
assert receipt.get('status') == 'complete' and receipt.get('round') == expected_round, receipt
assert receipt.get('world_size') == 4, receipt
files = {
    'adapter_model': root / 'adapter/adapter_model.safetensors',
    'recurrent_state': root / 'recurrent_state.pt',
}
actual = {}
for key, path in files.items():
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    actual[key] = digest.hexdigest()
    assert actual[key] == receipt['sha256'][key], (key, actual[key], receipt['sha256'][key])
print(json.dumps({'round': expected_round, 'status': 'verified', 'sha256': actual}, indent=2, sort_keys=True))
PY

  attempt=0
  while [[ ! -f "$round_dir/recurrent_causal_gate.json" ]]; do
    attempt=$((attempt + 1))
    (( attempt <= 3 )) || fail "round=$round local C4 exhausted three attempts"
    wait_for_gpu
    log "round=$round local C4 attempt=$attempt"
    if CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$project_dir/scripts" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      /home/v-yuejiali/.venvs/studentmodeling/bin/python -u \
      scripts/evaluate_recurrent_causal_gate.py \
      --model /home/v-yuejiali/studentmodeling/models/Qwen3-32B \
      --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --jsonl data/processed/malrule_c1_clean/test_same_template.jsonl \
      --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
      --output-json "$round_dir/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' \
      --max-length 1536 \
      --num-pairs 150 \
      --seed 42 > "$round_dir/attempt-$attempt.log" 2>&1; then
      break
    fi
    sleep 60
  done

  python - "$round_dir/recurrent_causal_gate.json" "$round_dir/summary.json" "$round" <<'PY' \
    || fail "round=$round C4 summary validation failed"
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8'))
assert value.get('raw_history_visible_to_decoder') is False, value
assert value.get('persistent_history_kv_cache') is False, value
assert value.get('intervention_target') == 'final_continuous_recurrent_state', value
summary = value['summary']
payload = {
    'round': int(sys.argv[3]),
    'D_specific': summary['D_specific'],
    'D_damage': summary['D_damage'],
    'clean_n': summary['clean_n'],
    'swap_flip_to_A': summary['clean']['swap']['flip_to_A_rate'],
    'random_flip_to_A': summary['clean']['random']['flip_to_A_rate'],
    'noop_flip_to_A': summary['clean']['noop']['flip_to_A_rate'],
    'samemal_flip_to_A': summary['clean']['samemal']['flip_to_A_rate'],
    'scientific_gate_pass': summary['scientific_gate_pass'],
}
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  refresh_curve
  log "round=$round local C4 complete"
done

touch "$state_dir/complete"
log "M4 local checkpoint C4 curve complete"
