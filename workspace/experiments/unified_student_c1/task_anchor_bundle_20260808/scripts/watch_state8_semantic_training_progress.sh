#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
root="$project/results/state8_semantic_candidates_20260806"
controller="$project/results/state8_semantic_training_progress_20260806"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
poll_seconds="${POLL_SECONDS:-120}"
stall_seconds="${STALL_SECONDS:-3600}"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-keyflip-feeds}"

declare -A jobs=(
  [c4_g005]=state8-semantic-causal-g005-seed45
  [c4_g010]=state8-semantic-causal-seed45
  [c4_g025]=state8-semantic-causal-g025-seed45
  [eedi_g005]=state8-eedi-geometry-g005-seed45
  [eedi_g010]=state8-eedi-geometry-g010-seed45
  [eedi_g025]=state8-eedi-geometry-g025-seed45
  [eedi_g050]=state8-eedi-geometry-g050-seed45
  [eedi_g025_single]=state8-eedi-g025-single-seed46
)

mkdir -p "$controller"
cd "$project" || exit 1
exec 9>"$controller/controller.lock"; flock -n 9 || exit 0
echo $$ > "$controller/controller.active.pid"
trap 'rm -f "$controller/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$controller/controller.heartbeat"; }

status_name(){
  "$python_bin" - "$1" <<'PY'
import re,sys
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',sys.argv[1])
values=re.findall(r'\b(preparing|queued|running|paused|resuming|pass|failed|killed|cancelled)\b',text,re.I)
print(values[-1].lower() if values else 'unknown')
PY
}

observe(){
  local slug="$1" state="$root/$1" progress="$controller/$1"
  local experiment status value listing log_file total_rounds=250
  mkdir -p "$progress"
  [[ -s "$state/active_experiment.txt" ]] || return 0
  experiment="$(<"$state/active_experiment.txt")"
  [[ "$slug" == eedi_g025_single ]] && total_rounds=1000
  status="$(timeout 90 amlt status "$experiment" ":${jobs[$slug]}" 2>&1)" || return 0
  value="$(status_name "$status")"
  printf '[%s] slug=%s experiment=%s status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$slug" "$experiment" "$value" \
    >> "$controller/status.log"
  if [[ "$value" != running ]]; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "" \
      --status "$value" --output "$progress/progress.json" \
      --total-rounds "$total_rounds" >/dev/null 2>&1 || true
    rm -f "$progress/stalled.json"
    return 0
  fi
  listing="$(timeout 90 amlt logs list "$experiment" ":${jobs[$slug]}" 2>&1)" || listing=""
  log_file="$("$python_bin" - "$listing" <<'PY'
import re,sys
paths=[]
for line in sys.argv[1].splitlines():
 value=line.strip()
 if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log_process_(?:0|00)\.txt',value):
  retry=re.search(r'retry_(\d+)',value)
  paths.append((int(retry.group(1)) if retry else -1,value))
print(max(paths)[1] if paths else '')
PY
)"
  if [[ -n "$log_file" ]] && timeout 90 amlt logs view -F "$log_file" -n 600 \
      "$experiment" ":${jobs[$slug]}" > "$progress/latest_rank0.log" \
      2> "$progress/latest_rank0.stderr"; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --log "$progress/latest_rank0.log" --experiment "$experiment" \
      --log-file "$log_file" --status "$value" \
      --output "$progress/progress.json" --total-rounds "$total_rounds" \
      --max-run-seconds 172800 >/dev/null 2>&1 || true
  else
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "$log_file" \
      --status "$value" --output "$progress/progress.json" \
      --total-rounds "$total_rounds" >/dev/null 2>&1 || true
  fi
  if "$python_bin" - "$progress/progress.json" "$stall_seconds" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
raise SystemExit(0 if float(x.get('seconds_since_progress') or 0)>=float(sys.argv[2]) else 1)
PY
  then cp "$progress/progress.json" "$progress/stalled.json"
  else rm -f "$progress/stalled.json"; fi
}

while true; do
  heartbeat
  for slug in c4_g005 c4_g010 c4_g025 eedi_g005 eedi_g010 eedi_g025 eedi_g050 eedi_g025_single; do
    observe "$slug"
    heartbeat
  done
  all_selection_arms_complete=true
  for slug in c4_g005 c4_g010 c4_g025 eedi_g005 eedi_g010 eedi_g025 eedi_g050; do
    if [[ ! -f "$root/$slug/complete" ]]; then
      all_selection_arms_complete=false
      break
    fi
  done
  if [[ "$all_selection_arms_complete" == true ]]; then
    touch "$controller/complete"
    heartbeat
    exit 0
  fi
  sleep "$poll_seconds"
done
