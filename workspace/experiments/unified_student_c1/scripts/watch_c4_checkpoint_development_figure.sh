#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
monitor_dir="$project_dir/results/c4_strict3seed_and_curves_monitor_20260824"
input_dir="$monitor_dir/download"
figure_dir="/home/v-yuejiali/studentmodeling/paper_iclr/figures"
log="$monitor_dir/development_figure.log"

cd "$project_dir"
while [[ ! -f "$monitor_dir/complete" ]]; do
  if [[ -f "$monitor_dir/needs_attention" ]]; then
    printf '[%s] upstream monitor needs attention; no figure rendered\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$log"
    exit 1
  fi
  printf '[%s] waiting for all strict checkpoint curves\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$log"
  sleep 120
done

python scripts/plot_c4_checkpoint_development_figure.py \
  --input-root "$input_dir" --output-dir "$figure_dir" >>"$log" 2>&1

test -s "$figure_dir/c4_checkpoint_development.source.json"
test -s "$figure_dir/c4_checkpoint_development.pdf"
test -s "$figure_dir/c4_checkpoint_development.png"
python - "$figure_dir/c4_checkpoint_development.source.json" <<'PY' >>"$log" 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert set(payload["curves"]) == {"13", "42", "97"}, payload["curves"].keys()
assert payload["definition"]["checkpoints"] == [250, 500, 750, 1000, 1250, 1500]
for rows in payload["curves"].values():
    assert len(rows) == 6
print("source-data validation passed")
PY
sha256sum \
  "$figure_dir/c4_checkpoint_development.source.json" \
  "$figure_dir/c4_checkpoint_development.pdf" \
  "$figure_dir/c4_checkpoint_development.png" >>"$log"
printf '[%s] development figure complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$log"
touch "$monitor_dir/development_figure.complete"
