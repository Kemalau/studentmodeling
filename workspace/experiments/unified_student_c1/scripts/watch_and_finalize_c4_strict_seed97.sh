#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
cd "$ROOT"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2

experiment=c4-cf1-visionpad-strict3seed-v2-20260824
job_name=c4-cf1-visionpad-strict-seed97
job_ref=:$job_name
job_id=cyan_stem_ld52bh4tn9
resource_group=rg-cs-newsandfeeds-singularity
workspace=CS-NewsAndFeeds-Singularity
download_root=results/c4_strict3seed_v2_seed97_download_20260824
artifact_dir="$download_root/$experiment/$job_name"
monitor_dir=results/c4_strict3seed_v2_seed97_monitor_20260824
audit_json="$download_root/seed97_audit.json"
mkdir -p "$monitor_dir"

last=""
while true; do
  now=$(date -u +%FT%TZ)
  if raw=$(az ml job show --name "$job_id" -g "$resource_group" -w "$workspace" \
      --query '{status:status,error:error}' -o json 2>&1); then
    status=$(python -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' <<<"$raw")
  else
    status=query_error
  fi
  if [[ "$status" != "$last" ]]; then
    printf '%s %s\n%s\n' "$now" "$status" "$raw" | tee -a "$monitor_dir/status.log"
    last=$status
  fi
  case "$status" in
    Completed|Failed|Canceled|NotResponding)
      break
      ;;
  esac
  sleep 120
done

amlt status "$experiment" "$job_ref" >"$monitor_dir/final_status.txt" 2>&1 || true
amlt logs view -n 500 "$experiment" "$job_ref" >"$monitor_dir/final_log.txt" 2>&1 || true

if [[ "$status" != Completed ]]; then
  printf '%s\n' "$status" >"$monitor_dir/TERMINAL_FAILURE"
  exit 2
fi

patterns=(
  gate_summary.json state_swap_masked.json state_probe_masked.json
  response_same_masked.json response_cross_masked.json input_manifest.txt
  output_manifest.txt train.log 'trainer_state_step*.json'
)
download_args=()
for pattern in "${patterns[@]}"; do
  download_args+=(-I "$pattern")
done

for attempt in 1 2 3 4 5; do
  rm -rf "$download_root"
  if amlt results download "$experiment" "$job_ref" "${download_args[@]}" \
      -o "$download_root" >"$monitor_dir/download_attempt${attempt}.log" 2>&1; then
    break
  fi
  if [[ "$attempt" == 5 ]]; then
    printf '%s\n' download_failed >"$monitor_dir/TERMINAL_FAILURE"
    exit 3
  fi
  sleep 120
done

python scripts/audit_c4_strict_seed.py \
  --artifact-dir "$artifact_dir" \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --seed 97 \
  --output "$audit_json" \
  >"$download_root/seed97_audit.log"

python - "$audit_json" "$download_root/seed97_audit.md" <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
x = json.loads(source.read_text())
m = x["metrics"]
c = x["checks"]
text = f"""# Strict C4 CF1 + vision-pad seed 97 audit

- Audit passed: `{x['passed']}`
- Training complete: `{c['training_complete']}` (`1500/1500` steps)
- Final validation loss: `{c['final_eval_loss']:.6f}`
- Cross-template response accuracy: `{m['response_cross_accuracy']:.6f}`
- Best cross-template state probe: `{m['state_probe_best_cross_accuracy']:.6f}`
- Frozen C4 pairs: `{c['swap_shape']['num_pairs']}` pairs, `{c['swap_shape']['record_count']}` records, all five conditions complete
- Clean subset: `n={m['clean_n']}`
- Directed / random / same-rule / no-op flip rates: `{m['directed_flip_rate']:.6f}` / `{m['random_flip_rate']:.6f}` / `{m['same_rule_flip_rate']:.6f}` / `{m['noop_flip_rate']:.6f}`
- D_specific: `{m['D_specific']:.6f}`
- Exact frozen pair manifest: `{c['frozen_pair_manifest_exact']}`
- Full 64-layer trajectory: `{c['all_64_layers']}`
- Gate summary independently reproduced: `{c['gate_summary_reproduced']}`
- Input and downloaded-output hashes verified: `{all(c['input_hashes'].values()) and all(c['downloaded_output_hashes'].values())}`
"""
target.write_text(text, encoding="utf-8")
PY

printf '%s\n' complete >"$monitor_dir/COMPLETE"
