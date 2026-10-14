#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_LRFIX_V4_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}
experiment=state8-profile-ownership-lrfix-v4-world4-hashfix-20260821
state="$project/results/state8_profile_ownership_lrfix_v4_world4_hashfix_20260821"
original="$project/results/state8_profile_ownership_lrfix_v4_world4_diagnostic_20260821"
promotion="$project/results/state8_profile_ownership_lrfix_v4_formal_promotion_20260820"
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state" "$original" "$promotion"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT
rm -f "$state/complete" "$state/needs_attention" "$state/promoted_original_world4"
export AZURE_CONFIG_DIR="$auth"

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  status="$(timeout 120 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" "$python" - <<'PY'
import json, os
rows = json.loads(os.environ["STATUS"])
done = {"pass", "passed", "failed", "fail", "killed", "cancelled", "canceled", "stopped"}
states = [str(row.get("status") or "").casefold() for row in rows]
print("yes" if states and all(value in done for value in states) else "no")
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 60
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout 1800 amlt results download "$experiment" --status pass \
  --include '*.json' --output "$state/download" > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" "$python" - <<'PY'
import json, os, pathlib, shutil
state = pathlib.Path(os.environ["STATE"])
rows = json.loads(os.environ["STATUS"])
passed = bool(rows) and all(str(row.get("status") or "").casefold() in {"pass", "passed"} for row in rows)
artifacts = sorted(state.joinpath("download").rglob("distributed_diagnostic_completion.json"))
failures = []
if not passed:
    failures.append("remote hash-fix job did not pass")
if len(artifacts) != 1:
    failures.append(f"expected one distributed diagnostic artifact, found {len(artifacts)}")
payload = json.loads(artifacts[0].read_text()) if len(artifacts) == 1 else None
if payload is not None:
    checks = {
        "diagnostic_complete": payload.get("status") == "diagnostic_complete",
        "world_size_4": payload.get("distributed_world_size") == 4,
        "micro_steps_8": payload.get("micro_steps") == 8,
        "positions_0_through_7": payload.get("global_positions_seen") == list(range(8)),
        "rank_assignment_consistent": payload.get("rank_assignment_consistent") is True,
        "history_integrity": payload.get("history_integrity_pass") is True,
    }
    failures += [name for name, ok in checks.items() if not ok]
summary = {
    "stage": "state8_profile_ownership_lrfix_v4_world4_diagnostic_watcher_v1",
    "recovery_mode": "source_hash_fix_only",
    "remote_passed": passed,
    "rows": rows,
    "failures": failures,
    "distributed_diagnostic": payload,
}
(state / "terminal_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if payload is not None:
    shutil.copy2(artifacts[0], state / "distributed_diagnostic_completion.json")
(state / ("complete" if not failures else "needs_attention")).touch()
raise SystemExit(0 if not failures else 1)
PY
audit_rc=$?
if (( audit_rc != 0 )); then
  timeout 180 amlt logs view -n 500 --no-pager "$experiment" > "$state/terminal_logs.txt" 2>&1 || true
  exit "$audit_rc"
fi

# Replace only the failed execution receipt. The training/evaluation protocol
# and the promotion gate remain unchanged.
cp "$original/terminal_summary.json" "$original/terminal_summary.hash_drift_failure.json" 2>/dev/null || true
cp "$state/terminal_summary.json" "$original/terminal_summary.json.partial"
mv "$original/terminal_summary.json.partial" "$original/terminal_summary.json"
cp "$state/distributed_diagnostic_completion.json" "$original/distributed_diagnostic_completion.json.partial"
mv "$original/distributed_diagnostic_completion.json.partial" "$original/distributed_diagnostic_completion.json"
rm -f "$original/needs_attention"
touch "$original/complete" "$state/promoted_original_world4"

if [[ ! -f "$promotion/promoted" && ! -f "$promotion/controller.pid" ]]; then
  setsid env PROFILE_OWNERSHIP_LRFIX_V4_AUTH="$auth" \
    bash scripts/promote_state8_profile_ownership_lrfix_v4_formal.sh \
    >> "$promotion/controller.nohup.log" 2>&1 < /dev/null &
  echo $! > "$promotion/controller.launch.pid"
fi
