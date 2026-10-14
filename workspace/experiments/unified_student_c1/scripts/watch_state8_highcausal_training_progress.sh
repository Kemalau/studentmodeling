#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
screen="$project/results/state8_highcausal_binding_20260806"
controller="$project/results/state8_highcausal_training_progress_20260806"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
poll_seconds="${POLL_SECONDS:-120}"
stall_seconds="${STALL_SECONDS:-3600}"
if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  if AZURE_CONFIG_DIR=/home/v-yuejiali/.azure az account get-access-token \
      --subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895 \
      --query expiresOn -o tsv >/dev/null 2>&1; then
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
  else
    export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds
  fi
fi

declare -A jobs=(
  [joint]=state8-highcausal-joint-seed47
  [updater]=state8-highcausal-updater-seed48
  [mid]=state8-highcausal-mid-seed49
  [eedi]=state8-highcausal-eedi-seed50
  [taxonomy]=state8-taxonomy-eedi-highcausal-seed51
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
  local slug="$1" progress="$controller/$1"
  local experiment status value listing log_file
  mkdir -p "$progress"
  if [[ "$slug" == taxonomy ]]; then
    experiment=state8-taxonomy-eedi-highcausal-seed51-20260807
  elif [[ "$slug" == mid || "$slug" == eedi ]]; then
    [[ -s "$screen/$slug/active_experiment.txt" ]] || return 0
    experiment="$(<"$screen/$slug/active_experiment.txt")"
  else
    [[ -s "$screen/active_experiment.txt" ]] || return 0
    experiment="$(<"$screen/active_experiment.txt")"
  fi
  status="$(timeout 90 amlt status "$experiment" ":${jobs[$slug]}" 2>&1)" || return 0
  value="$(status_name "$status")"
  printf '[%s] slug=%s experiment=%s status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$slug" "$experiment" "$value" \
    >> "$controller/status.log"
  if [[ "$value" != running ]]; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "" \
      --status "$value" --output "$progress/progress.json" \
      --total-rounds 250 >/dev/null 2>&1 || true
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
  if [[ -n "$log_file" ]] && timeout 90 amlt logs view -F "$log_file" -n 1200 \
      "$experiment" ":${jobs[$slug]}" > "$progress/latest_rank0.log" \
      2> "$progress/latest_rank0.stderr"; then
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --log "$progress/latest_rank0.log" --experiment "$experiment" \
      --log-file "$log_file" --status "$value" \
      --output "$progress/progress.json" --total-rounds 250 \
      --max-run-seconds 172800 >/dev/null 2>&1 || true
  else
    "$python_bin" scripts/monitor_layerwise_unified_training_progress.py \
      --awaiting-first-record --experiment "$experiment" --log-file "$log_file" \
      --status "$value" --output "$progress/progress.json" \
      --total-rounds 250 >/dev/null 2>&1 || true
  fi
  if "$python_bin" - "$progress/progress.json" "$stall_seconds" <<'PY'
import json,sys
try: value=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
stalled=float(value.get('seconds_since_progress') or 0)>=float(sys.argv[2])
unhealthy=value.get('health') is not None and not bool(value['health'].get('healthy'))
raise SystemExit(0 if stalled or unhealthy else 1)
PY
  then cp "$progress/progress.json" "$progress/stalled.json"
  else rm -f "$progress/stalled.json"; fi
}

while [[ ! -f "$screen/complete" && ! -f "$screen/needs_attention" ]]; do
  heartbeat
  observe joint
  heartbeat
  observe updater
  heartbeat
  observe mid
  heartbeat
  observe eedi
  heartbeat
  observe taxonomy
  heartbeat
  sleep "$poll_seconds"
done
touch "$controller/complete"
heartbeat
