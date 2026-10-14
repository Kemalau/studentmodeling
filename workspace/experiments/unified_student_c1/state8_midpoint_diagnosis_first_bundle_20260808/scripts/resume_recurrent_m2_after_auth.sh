#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
config="amlt/feeds_recurrent_student_state_u1_u4_factorial_20260730.yaml"
experiment="recurrent-m2-factorial-seed42-v4-20260730"
state_dir="$project_dir/results/recurrent_m2_factorial_submit_20260730"
log="$state_dir/submit.log"
receipt="$state_dir/submission_receipt.txt"
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
if [[ -f "$receipt" ]]; then
  exit 0
fi
cd "$project_dir"

deadline=$(( $(date +%s) + 86400 ))
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
    {
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] authenticated account does not match fixed Feeds subscription"
      echo "$account"
    } >> "$log"
    touch "$state_dir/needs_attention"
    exit 1
  fi
  break
done
if [[ -z "${account:-}" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] authentication did not recover within 24h" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

az account set --subscription "$expected_subscription" >> "$log" 2>&1 || {
  touch "$state_dir/needs_attention"; exit 1;
}
# Serialize all Feeds submissions from this project so independent controllers
# cannot both make a decision from the same stale quota snapshot.
exec 8>"$project_dir/results/feeds_submission_global.lock"
flock 8
python scripts/validate_recurrent_amlt_config.py --config "$config" >> "$log" 2>&1 || {
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
dump_output="$(amlt run "$config" --dump -t Feeds 2>&1)" || {
  echo "$dump_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
printf '%s\n' "$dump_output" > "$state_dir/amlt_dump.txt"
python scripts/validate_recurrent_amlt_config.py \
  --config "$state_dir/amlt_dump.txt" >> "$log" 2>&1 || {
  touch "$state_dir/needs_attention"; exit 1;
}
{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] preflight"
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
user_limit = int(quotas["user_max"]["limit"])
user_used = int(quotas["user_max"].get("used") or 0)
user_available = user_limit - user_used
assert available >= 4, {"available": available, "row": row}
assert user_available >= 4, {"user_available": user_available, "row": row}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "selected_row": row,
            "standard_available": available,
            "required_units": 4,
            "user_max": user_limit,
            "user_used": user_used,
            "user_available": user_available,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
then
  printf '[%s] quota parse/availability gate failed\n%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$quota_output" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

# A pre-existing experiment is never overwritten or appended to implicitly.
existing="$(amlt status "$experiment" 2>&1)"
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
    echo "status=preexisting_not_resubmitted"
    echo "$existing"
  } > "$receipt"
elif [[ "$existing_class" == missing ]]; then
  submit_output="$(amlt run "$config" "$experiment" -t Feeds -y \
    --description 'Strict M2 recurrent-state u1/u4 identity-vs-low-rank transition gate; fixed v4 protocol' 2>&1)"
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
    echo "status=submitted"
    echo "$submit_output"
  } > "$receipt"
else
  echo "unexpected experiment status class=$existing_class" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

accepted="$(amlt status "$experiment" 2>&1)" || {
  echo "$accepted" >> "$log"; touch "$state_dir/needs_attention"; exit 1;
}
echo "$accepted" >> "$log"
grep -Eqi 'queued|preparing|running|pass|failed' <<< "$accepted" || {
  touch "$state_dir/needs_attention"; exit 1;
}
flock -u 8
exec bash scripts/watch_recurrent_m2_factorial.sh
