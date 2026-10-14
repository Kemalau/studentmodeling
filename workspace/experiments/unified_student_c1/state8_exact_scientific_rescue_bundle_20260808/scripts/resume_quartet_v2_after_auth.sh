#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
config="amlt/feeds_paired_state_quartet_v2_seed42_20260730.yaml"
experiment="paired-state-quartet-v2-maskfix-seed42-20260730"
state_dir="$project_dir/results/paired_state_quartet_v2_maskfix_submit_20260730"
log="$state_dir/submit.log"
receipt="$state_dir/submission_receipt.txt"
expected_tenant="72f988bf-86f1-41af-91ab-2d7cd011db47"
expected_subscription="b6dc87f3-c479-49c8-8cb5-7896da3ff895"
expected_workspace_name="CS-NewsAndFeeds-Singularity"
expected_workspace_resource_group="rg-cs-newsandfeeds-singularity"

mkdir -p "$state_dir"
exec 9>"$state_dir/submit.lock"
flock -n 9 || exit 0
echo "$$" > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
[[ -f "$receipt" ]] && exit 0
cd "$project_dir"

deadline=$(( $(date +%s) + 86400 ))
while (( $(date +%s) < deadline )); do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"
  az account set --subscription "$expected_subscription" >/dev/null 2>&1 || { sleep 60; continue; }
  account="$(az account show --query '{tenant:tenantId,subscription:id,state:state}' -o json 2>/dev/null || true)"
  [[ -n "$account" ]] || { sleep 60; continue; }
  if ! python - "$account" "$expected_tenant" "$expected_subscription" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
assert value["tenant"] == sys.argv[2]
assert value["subscription"] == sys.argv[3]
assert str(value["state"]).lower() == "enabled"
PY
  then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] account mismatch: $account" >> "$log"
    touch "$state_dir/needs_attention"
    exit 1
  fi
  break
done
[[ -n "${account:-}" ]] || { touch "$state_dir/needs_attention"; exit 1; }

# M2 owns submission priority. Wait for its acceptance receipt, then serialize
# the fresh quota check and this submission under the same project-wide lock.
m2_receipt="$project_dir/results/recurrent_m2_factorial_submit_20260730/submission_receipt.txt"
while [[ ! -f "$m2_receipt" ]]; do
  sleep 60
done
exec 8>"$project_dir/results/feeds_submission_global.lock"
flock 8

python scripts/validate_quartet_amlt_config.py --config "$config" >> "$log" 2>&1 || {
  touch "$state_dir/needs_attention"; exit 1;
}
project_output="$(amlt project 2>&1)" || { echo "$project_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1; }
workspace_output="$(amlt workspace list --no-update 2>&1)" || { echo "$workspace_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1; }
grep -Fq 'project v-yuejiali' <<< "$project_output" || { touch "$state_dir/needs_attention"; exit 1; }
python scripts/feeds_workspace_utils.py \
  --name "$expected_workspace_name" \
  --resource-group "$expected_workspace_resource_group" \
  <<< "$workspace_output" || { touch "$state_dir/needs_attention"; exit 1; }

while true; do
  quota_output="$(amlt --json-tables ti sing -t Feeds 2>&1)" || {
    echo "$quota_output" >> "$log"; sleep 120; continue;
  }
  if python - "$quota_output" "$state_dir/quota_before_submission.json" <<'PY'
import json, sys
text=sys.argv[1]; start=text.find('[')
assert start >= 0
rows=json.loads(text[start:])
matches=[r for r in rows if r.get('target_name')=='Feeds' and r.get('series')=='NDAMv4' and (r.get('accelerator') or {}).get('type')=='A100' and int((r.get('accelerator') or {}).get('memory_gb'))==80]
assert len(matches)==1
row=matches[0]; q=row['quotas_(used/limit)']; available=int(q['standard']['limit'])-int(q['standard']['used']); user_limit=int(q['user_max']['limit']); user_used=int(q['user_max'].get('used') or 0); user_available=user_limit-user_used
with open(sys.argv[2],'w') as f: json.dump({'selected_row':row,'standard_available':available,'required_units':3,'user_max':user_limit,'user_used':user_used,'user_available':user_available},f,indent=2)
assert available >= 3 and user_available >= 3
PY
  then break; fi
  sleep 120
done

dump_output="$(amlt run "$config" --dump -t Feeds 2>&1)" || { echo "$dump_output" >> "$log"; touch "$state_dir/needs_attention"; exit 1; }
printf '%s\n' "$dump_output" > "$state_dir/amlt_dump.txt"
python scripts/validate_quartet_amlt_config.py --config "$state_dir/amlt_dump.txt" >> "$log" 2>&1 || {
  touch "$state_dir/needs_attention"; exit 1;
}

existing="$(amlt status "$experiment" 2>&1)"; rc=$?
if (( rc == 0 )) && grep -Eqi 'Experiment:|jobs —|jobs -' <<< "$existing"; then
  { echo "experiment=$experiment"; echo "status=preexisting_not_resubmitted"; echo "$existing"; } > "$receipt"
elif grep -Eqi 'not found|cannot find an experiment|does not exist|no experiment' <<< "$existing"; then
  submit="$(amlt run "$config" "$experiment" -t Feeds -y --description 'Quartet-v2 native-B versus patched-donor-A causal identification; control/all/latehalf' 2>&1)"
  rc=$?
  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit rc=$rc"; echo "$submit"; } >> "$log"
  (( rc == 0 )) || { touch "$state_dir/needs_attention"; exit "$rc"; }
  { echo "experiment=$experiment"; echo "status=submitted"; echo "$submit"; } > "$receipt"
else
  echo "$existing" >> "$log"
  touch "$state_dir/needs_attention"
  exit 1
fi

accepted="$(amlt status "$experiment" 2>&1)" || { echo "$accepted" >> "$log"; touch "$state_dir/needs_attention"; exit 1; }
echo "$accepted" >> "$log"
grep -Eqi 'queued|preparing|running|pass|failed' <<< "$accepted" || { touch "$state_dir/needs_attention"; exit 1; }
flock -u 8
exec bash scripts/watch_paired_state_quartet_v2.sh
