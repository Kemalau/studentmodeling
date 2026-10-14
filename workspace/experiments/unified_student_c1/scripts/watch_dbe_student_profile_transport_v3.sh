#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment="${EXPERIMENT:-dbe-state8-student-profile-transport-v3-20260817}"
state="${STATE_DIR:-$project/results/dbe_student_profile_transport_v3_20260817}"
auth=/home/v-yuejiali/.azure-studentmodeling-feeds-v2

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/watcher.lock"
flock -n 9 || exit 0
echo $$ > "$state/watcher.pid"
trap 'rm -f "$state/watcher.pid"' EXIT

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$state/watcher.heartbeat"
  export AZURE_CONFIG_DIR="$auth"
  status="$(timeout 90 amlt --json-tables status "$experiment" \
    --hide-urls --force-status-update --no-calculate-result-size 2>/dev/null)" || {
      sleep 60
      continue
    }
  printf '%s\n' "$status" > "$state/status.json.partial"
  mv "$state/status.json.partial" "$state/status.json"
  terminal="$(STATUS="$status" /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json
import os

rows = json.loads(os.environ["STATUS"])
active = {"queued", "preparing", "running", "pausing", "resuming", "paused"}
print("no" if any(str(row.get("status") or "").casefold() in active for row in rows) else "yes")
PY
)"
  [[ "$terminal" == yes ]] && break
  sleep 60
done

rm -rf "$state/download"
mkdir -p "$state/download"
timeout 1800 amlt results download "$experiment" --status pass \
  --include 'metrics.json' --include 'readable_profiles.jsonl' \
  --include 'fixed_query_transport.jsonl' --include 'compressed_feature_cache.npz' \
  --output "$state/download" > "$state/download.log" 2>&1 || true

STATUS="$status" STATE="$state" EXPERIMENT="$experiment" \
  /home/v-yuejiali/.venvs/studentmodeling/bin/python - <<'PY'
import json
import os
import pathlib
import shutil
import subprocess

state = pathlib.Path(os.environ["STATE"])
rows = json.loads(os.environ["STATUS"])
passed = bool(rows) and all(
    str(row.get("status") or "").casefold() == "pass" for row in rows
)
files = {
    "metrics.json": sorted(state.joinpath("download").rglob("profile_transport_v3/metrics.json")),
    "readable_profiles.jsonl": sorted(state.joinpath("download").rglob("profile_transport_v3/readable_profiles.jsonl")),
    "fixed_query_transport.jsonl": sorted(state.joinpath("download").rglob("profile_transport_v3/fixed_query_transport.jsonl")),
    "compressed_feature_cache.npz": sorted(state.joinpath("download").rglob("profile_transport_v3/compressed_feature_cache.npz")),
}
complete = passed and all(len(paths) == 1 for paths in files.values())
summary = {
    "experiment": os.environ["EXPERIMENT"],
    "rows": rows,
    "all_passed": passed,
    "artifact_cardinality": {name: len(paths) for name, paths in files.items()},
}
if complete:
    for name, paths in files.items():
        shutil.copy2(paths[0], state / name)
    payload = json.loads((state / "metrics.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "scientific_status": "window_matched_profile_transport_v3_complete",
            "scientific_gate_pass": payload.get("scientific_gate_pass"),
            "gates": payload.get("gates"),
            "metrics": payload.get("metrics"),
            "transport": payload.get("transport"),
            "uncertainty": payload.get("uncertainty"),
        }
    )
elif passed:
    summary["scientific_status"] = "artifact_cardinality_error"
else:
    summary["scientific_status"] = "remote_job_failed"
    for row in rows:
        if str(row.get("status") or "").casefold() == "pass":
            continue
        safe = str(row.get("job_name") or "job").lstrip(":").replace("/", "_")
        with (state / f"{safe}.failure.log").open("w") as handle:
            subprocess.run(
                [
                    "timeout", "180", "amlt", "logs", "view", "-n", "1000",
                    "--no-pager", os.environ["EXPERIMENT"],
                    str(row.get("job_name") or ""),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
(state / "terminal_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
(state / ("complete" if complete else "needs_attention")).touch()
PY
