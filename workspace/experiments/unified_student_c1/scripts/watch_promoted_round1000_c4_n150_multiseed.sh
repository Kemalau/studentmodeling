#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=promoted-round1000-c4-n150-multiseed-20260804
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1000_c4_n150_multiseed"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

rm -f "$state_dir/complete" "$state_dir/needs_attention"
while true; do
  heartbeat
  status="$(timeout 45 amlt status "$experiment" 2>&1)" || { sleep 60; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state_dir/status.log"
  if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued' <<< "$status"; then
    sleep 60
    continue
  fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1], 3) else 1)
PY
  then
    break
  fi
  amlt show "$experiment" > "$state_dir/show.txt" 2>&1 || true
  amlt logs view -n 500 "$experiment" > "$state_dir/failure.log" 2>&1 || true
  fail "multiseed n150 C4 reached terminal non-pass state"
done

rm -rf "$state_dir/download"
mkdir -p "$state_dir/download"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.txt' --output "$state_dir/download" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "could not download multiseed n150 C4 results"

readarray -t summaries < <(find "$state_dir/download" -type f -path '*/eval/c4/summary.json' | sort)
[[ ${#summaries[@]} -eq 3 ]] || fail "downloaded multiseed C4 summary cardinality drift"
"$python_bin" - "$state_dir/summary.json" "${summaries[@]}" <<'PY' \
  || fail "downloaded multiseed n150 C4 audit failed"
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1]); rows=[json.load(open(path)) for path in sys.argv[2:]]
assert {row['pair_seed'] for row in rows}=={13,42,97},rows
assert all(row['checkpoint_round']==1000 and row['all_pairs_n']==150 for row in rows),rows
rows=sorted(rows,key=lambda row:row['pair_seed'])
values=[row['D_specific'] for row in rows]
summary={
  'stage':'layerwise_checkpoint_c4_multiseed',
  'checkpoint_round':1000,
  'diagnostic_only':True,
  'pair_seeds':[13,42,97],
  'rows':rows,
  'D_specific_mean':statistics.mean(values),
  'D_specific_min':min(values),
  'scientific_gate_pass_count':sum(bool(row['scientific_gate_pass']) for row in rows),
}
summary['two_of_three_gate_pass']=summary['scientific_gate_pass_count']>=2
temporary=out.with_suffix('.json.partial')
temporary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
temporary.replace(out)
print(json.dumps(summary,indent=2,sort_keys=True))
PY
touch "$state_dir/complete"
heartbeat
log "promoted round-1000 multiseed n150 C4 complete and audited"
