#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
auth=${PROFILE_OWNERSHIP_LRFIX_V4_AUTH:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}
smoke_state="${PROFILE_OWNERSHIP_LRFIX_V4_SMOKE_STATE:-$project/results/state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_20260820}"
world4_state="${PROFILE_OWNERSHIP_LRFIX_V4_WORLD4_STATE:-$project/results/state8_profile_ownership_lrfix_v4_world4_diagnostic_20260821}"
state="$project/results/state8_profile_ownership_lrfix_v4_formal_promotion_20260820"
formal_state="$project/results/state8_profile_ownership_lrfix_v4_formal_20260820"
config="$project/amlt/feeds_state8_profile_ownership_lrfix_v4_formal_world4_20260821.yaml"
experiment=state8-profile-ownership-lrfix-v4-formal-20260820
python=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state" "$formal_state"
cd "$project" || exit 1
test -d "$auth" || {
  printf 'Browser-login Azure cache does not exist: %s\n' "$auth" >&2
  exit 2
}
exec 9>"$state/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/controller.pid"
trap 'rm -f "$state/controller.pid"' EXIT
rm -f "$state/promoted" "$state/suppressed" "$state/needs_attention"

while [[ ! -f "$smoke_state/complete" && ! -f "$smoke_state/needs_attention" ]] \
   || [[ ! -f "$world4_state/complete" && ! -f "$world4_state/needs_attention" ]]; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
  sleep 60
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"
if [[ ! -f "$smoke_state/complete" ]]; then
  printf 'lrfix-v4 smoke did not complete its artifact contract; formal promotion suppressed\n' \
    > "$state/suppressed"
  cp "$smoke_state/terminal_summary.json" "$state/smoke_terminal_summary.json" 2>/dev/null || true
  exit 2
fi
if [[ ! -f "$world4_state/complete" ]]; then
  printf 'four-rank execution diagnostic did not pass its artifact contract; formal promotion suppressed\n' \
    > "$state/suppressed"
  cp "$world4_state/terminal_summary.json" "$state/world4_terminal_summary.json" 2>/dev/null || true
  exit 2
fi

"$python" - "$world4_state/terminal_summary.json" "$state/world4_gate.json" <<'PY'
import json, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
value = json.loads(source.read_text())
diagnostic = value.get("distributed_diagnostic") or {}
checks = {
    "watcher_stage": value.get("stage") == "state8_profile_ownership_lrfix_v4_world4_diagnostic_watcher_v1",
    "remote_passed": value.get("remote_passed") is True,
    "artifact_contract": not value.get("failures"),
    "diagnostic_complete": diagnostic.get("status") == "diagnostic_complete",
    "world_size_4": diagnostic.get("distributed_world_size") == 4,
    "micro_steps_8": diagnostic.get("micro_steps") == 8,
    "positions_0_through_7": diagnostic.get("global_positions_seen") == list(range(8)),
    "rank_assignment_consistent": diagnostic.get("rank_assignment_consistent") is True,
    "history_integrity": diagnostic.get("history_integrity_pass") is True,
}
result = {"status": "pass" if all(checks.values()) else "fail", "checks": checks}
target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
raise SystemExit(0 if all(checks.values()) else 2)
PY
world4_gate_rc=$?
if (( world4_gate_rc != 0 )); then
  printf 'four-rank execution diagnostic receipt failed validation; formal promotion suppressed\n' \
    > "$state/suppressed"
  exit "$world4_gate_rc"
fi

"$python" scripts/validate_state8_profile_ownership_lrfix_v4_promotion.py \
  --smoke-summary "$smoke_state/terminal_summary.json" \
  --output "$state/promotion_gate.json" \
  > "$state/promotion_gate.log" 2>&1
gate_rc=$?
if (( gate_rc != 0 )); then
  printf 'DBE development ownership gate did not pass; formal promotion suppressed\n' \
    > "$state/suppressed"
  exit "$gate_rc"
fi

"$python" scripts/validate_state8_profile_ownership_lrfix_v4_formal_amlt.py \
  --config "$config" > "$state/config_validation.txt" 2>&1 || {
  touch "$state/needs_attention"
  exit 3
}

export AZURE_CONFIG_DIR="$auth"
amlt --json-tables ti sing -t Feeds -a > "$state/quota.json" || {
  touch "$state/needs_attention"
  exit 4
}
amlt run "$config" --dump -t Feeds > "$state/preflight.dump.yaml" \
  2> "$state/preflight.stderr" || {
  touch "$state/needs_attention"
  exit 5
}
"$python" scripts/validate_state8_profile_ownership_lrfix_v4_formal_amlt.py \
  --config "$state/preflight.dump.yaml" > "$state/preflight_validation.txt" 2>&1 || {
  touch "$state/needs_attention"
  exit 6
}

existing="$(timeout 120 amlt --json-tables status "$experiment" \
  --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null || true)"
if [[ -n "$existing" && "$existing" != '[]' ]]; then
  printf '%s\n' "$existing" > "$state/status.existing.json"
else
  exec 8>"$project/results/feeds_submission_global.lock"
  flock -w 300 8 || {
    touch "$state/needs_attention"
    exit 7
  }
  amlt run "$config" "$experiment" -t Feeds -y \
    --description 'State8 lrfix-v4 formal DBE development run on four synchronized A100 ranks after positive significant 200-step ownership gate; exact 1000-row epoch and 250 updates; strict State8-only inference; frozen XES profile test remains untouched' \
    > "$state/submission.txt" 2>&1 || {
    touch "$state/needs_attention"
    exit 8
  }
  flock -u 8
fi

timeout 120 amlt --json-tables status "$experiment" --hide-urls \
  --force-status-update --no-calculate-result-size > "$state/status.initial.json" \
  2> "$state/status.initial.stderr" || {
  touch "$state/needs_attention"
  exit 9
}

rm -f "$formal_state/watcher.lock" "$formal_state/watcher.pid"
setsid env PROFILE_READOUT_EXPERIMENT="$experiment" \
  PROFILE_READOUT_STATE="$formal_state" PROFILE_READOUT_AUTH="$auth" \
  bash scripts/watch_state8_student_profile_readout.sh \
  > "$formal_state/watcher.nohup.log" 2>&1 < /dev/null &
echo $! > "$formal_state/watcher.launch.pid"
touch "$state/promoted"
