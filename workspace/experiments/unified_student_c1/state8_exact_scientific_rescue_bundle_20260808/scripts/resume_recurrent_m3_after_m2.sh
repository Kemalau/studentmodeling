#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
config="amlt/feeds_recurrent_student_m3_ab_20260730.yaml"
m2_summary="$project_dir/results/recurrent_m2_factorial_monitor_20260730/summary.json"
m2_attention="$project_dir/results/recurrent_m2_factorial_monitor_20260730/needs_attention"
state_dir="$project_dir/results/recurrent_m3_submit_20260730"
log="$state_dir/submit.log"
lock="$state_dir/submit.lock"
expected_tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
expected_subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
expected_workspace_name="CS-NewsAndFeeds-Singularity"
expected_workspace_resource_group="rg-cs-newsandfeeds-singularity"

mkdir -p "$state_dir"
exec 9>"$lock"
if ! flock -n 9; then
  exit 0
fi
echo "$$" > "$state_dir/controller.active.pid"
cleanup() { rm -f "$state_dir/controller.active.pid"; }
trap cleanup EXIT
cd "$project_dir"

deadline=$(( $(date +%s) + 604800 ))
while [[ ! -f "$m2_summary" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  if [[ -f "$m2_attention" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] M2 needs attention; M3 not submitted" >> "$log"
    touch "$state_dir/blocked_by_m2_execution"
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] M2 summary wait exceeded seven days" >> "$log"
    touch "$state_dir/needs_attention"
    exit 1
  fi
  sleep 120
done

projection="$(python - "$m2_summary" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if not value.get("downstream_rescue_authorized"):
    raise SystemExit(2)
projection = value.get("selected_projection_mode")
if projection not in {"identity", "trainable"}:
    raise SystemExit(3)
print(projection)
PY
)"
projection_rc=$?
if (( projection_rc != 0 )); then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] M2 strict gate failed without an authorized behavioral-rescue arm; M3 not submitted" >> "$log"
  touch "$state_dir/blocked_by_m2_science"
  exit 1
fi
printf '%s\n' "$projection" > "$state_dir/selected_projection.txt"
job_generation="m3-generation-only-${projection}-seed42"
job_kd="m3-generation-kd-${projection}-seed42"
base_experiment="recurrent-m3-ab-${projection}-seed42-fixedu4-v2-20260730"

account=""
while (( $(date +%s) < deadline )); do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  if ! az account set --subscription "$expected_subscription" >/dev/null 2>&1; then
    sleep 60
    continue
  fi
  account="$(az account show --query '{tenant:tenantId,subscription:id,state:state}' -o json 2>/dev/null || true)"
  if [[ -z "$account" ]]; then sleep 60; continue; fi
  if ! python - "$account" "$expected_tenant" "$expected_subscription" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
assert value["tenant"] == sys.argv[2], value
assert value["subscription"] == sys.argv[3], value
assert str(value["state"]).lower() == "enabled", value
PY
  then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] authenticated account mismatch: $account" >> "$log"
    touch "$state_dir/needs_attention"
    exit 1
  fi
  break
done
if [[ -z "$account" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] authentication did not recover" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

exec 8>"$project_dir/results/feeds_submission_global.lock"
flock 8

# A terminal AMLT experiment cannot be resumed under the same name. Select the
# first missing retry name, while reattaching to an already active or completed
# retry if a controller is restarted. All retries retain the same remote RUN_DIR,
# so train_recurrent_student_m3.py resumes from its latest atomic checkpoint.
experiment=""
receipt=""
for attempt in $(seq 0 20); do
  if (( attempt == 0 )); then
    candidate="$base_experiment"
    candidate_receipt="$state_dir/submission_receipt.txt"
  else
    candidate="${base_experiment}-retry${attempt}"
    candidate_receipt="$state_dir/submission_receipt.retry${attempt}.txt"
  fi
  candidate_status="$(amlt status --no-update-db "$candidate" 2>&1)"
  candidate_rc=$?
  candidate_class="$(python - "$candidate_rc" "$candidate_status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)"
  if [[ "$candidate_class" == missing ]]; then
    experiment="$candidate"
    receipt="$candidate_receipt"
    break
  fi
  if [[ "$candidate_class" == ambiguous ]]; then
    echo "$candidate_status" >> "$log"
    touch "$state_dir/needs_attention"
    exit 1
  fi
  if grep -Eqi 'running|queued|preparing|paused|resuming|2 jobs (—|-)[[:space:]]*2 pass' <<< "$candidate_status"; then
    experiment="$candidate"
    receipt="$candidate_receipt"
    break
  fi
  if grep -Eqi 'killed|failed|cancelled|canceled|jobs (—|-)' <<< "$candidate_status"; then
    printf '[%s] skipping terminal experiment=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$candidate" >> "$log"
    continue
  fi
  echo "$candidate_status" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
done
if [[ -z "$experiment" || -z "$receipt" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exhausted M3 retry names" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

python scripts/validate_m3_amlt_config.py --config "$config" >> "$log" 2>&1 || {
  touch "$state_dir/needs_attention"; exit 1;
}
project_output="$(amlt project 2>&1)" || {
  echo "$project_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
workspace_output="$(amlt workspace list --no-update 2>&1)" || {
  echo "$workspace_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
quota_output="$(amlt --json-tables ti sing -t Feeds 2>&1)" || {
  echo "$quota_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
dump_output="$(amlt run "$config" ":$job_generation" ":$job_kd" --dump -t Feeds 2>&1)" || {
  echo "$dump_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
printf '%s\n' "$dump_output" > "$state_dir/amlt_dump.txt"
python scripts/validate_m3_amlt_config.py \
  --config "$state_dir/amlt_dump.txt" \
  --projection "$projection" \
  --selected-only >> "$log" 2>&1 || {
  touch "$state_dir/needs_attention"; exit 1;
}
{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preflight"
  echo "projection=$projection"
  echo "$account"
  echo "$project_output"
  echo "$workspace_output"
  echo "$quota_output"
} >> "$log"
grep -Fq 'project v-yuejiali' <<< "$project_output" || {
  touch "$state_dir/needs_attention"; exit 1;
}
python scripts/feeds_workspace_utils.py \
  --name "$expected_workspace_name" \
  --resource-group "$expected_workspace_resource_group" \
  <<< "$workspace_output" || {
  touch "$state_dir/needs_attention"; exit 1;
}
if ! python - "$quota_output" "$state_dir/quota_before_submission.json" <<'PY'
import json, sys
from scripts.feeds_quota_utils import parse_json_table, select_a100_80gb
text = sys.argv[1]
row = select_a100_80gb(parse_json_table(text))
quotas = row["quotas_(used/limit)"]
standard = quotas["standard"]
available = int(standard["limit"]) - int(standard["used"])
user = quotas["user_max"]
user_available = int(user["limit"]) - int(user.get("used") or 0)
assert available >= 2, {"available": available, "row": row}
assert user_available >= 2, {"user_available": user_available, "row": row}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({
        "selected_row": row,
        "standard_available": available,
        "required_units": 2,
        "user_available": user_available,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
then
  touch "$state_dir/waiting_for_quota"
  exit 1
fi

existing="$(amlt status --no-update-db "$experiment" 2>&1)"
existing_rc=$?
existing_class="$(python - "$existing_rc" "$existing" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]), sys.argv[2]))
PY
)"
if [[ "$existing_class" == ambiguous ]]; then
  echo "$existing" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
elif [[ "$existing_class" == exists ]]; then
  {
    echo "experiment=$experiment"
    echo "projection=$projection"
    echo "status=preexisting_not_resubmitted"
    echo "$existing"
  } > "$receipt"
elif [[ "$existing_class" == missing ]]; then
  submit_output="$(amlt run "$config" ":$job_generation" ":$job_kd" "$experiment" \
    -t Feeds -y \
    --description "M3 unified recurrent state generation-only versus cached behavioral-KD; M2-selected $projection projection" 2>&1)"
  submit_rc=$?
  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit rc=$submit_rc"
    echo "$submit_output"
  } >> "$log"
  if (( submit_rc != 0 )); then
    touch "$state_dir/needs_attention"
    exit "$submit_rc"
  fi
  {
    echo "experiment=$experiment"
    echo "projection=$projection"
    echo "status=submitted"
    echo "$submit_output"
  } > "$receipt"
else
  echo "unexpected experiment status class=$existing_class" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

accepted="$(amlt status --no-update-db "$experiment" 2>&1)" || {
  echo "$accepted" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
echo "$accepted" >> "$log"
grep -Eqi 'queued|preparing|running|pass|failed' <<< "$accepted" || {
  touch "$state_dir/needs_attention"; exit 1;
}
flock -u 8
exec bash scripts/watch_recurrent_m3_ab.sh "$projection" "$experiment"
