#!/usr/bin/env bash
set -uo pipefail
project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1";state_dir="$project_dir/results/recurrent_m4_pipeline_20260730/single_layer_state_bridge";experiment="recurrent-cf1-single-vector-layer-bridge-c4-seed42-20260731"
mkdir -p "$state_dir/download";cd "$project_dir";exec 9>"$state_dir/controller.lock";flock -n 9||exit 0;echo $$>"$state_dir/controller.active.pid";trap 'rm -f "$state_dir/controller.active.pid"' EXIT;rm -f "$state_dir/complete" "$state_dir/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ>"$state_dir/controller.heartbeat";};log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"|tee -a "$state_dir/controller.log";};fail(){ log "ERROR: $*";touch "$state_dir/needs_attention";exit 1;}
while true;do heartbeat;status="$(amlt status "$experiment" 2>&1)";printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status">>"$state_dir/status.log";if grep -Eqi 'authentication|connection|timed out|permission'<<<"$status";then sleep 120;continue;fi;if grep -Eqi 'running|queued|preparing|paused|resuming|pre-queued'<<<"$status";then sleep 120;continue;fi;if python - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],5) else 1)
PY
then break;fi;amlt show "$experiment">"$state_dir/show.txt" 2>&1||true;timeout 180 amlt logs view -n 1200 "$experiment">"$state_dir/failure.log" 2>&1||true;fail 'single-layer bridge jobs did not pass';done
rm -rf "$state_dir/download";mkdir -p "$state_dir/download";amlt results download "$experiment" --status pass --include single_layer_state_bridge.json --include scope.txt --output "$state_dir/download">>"$state_dir/controller.log" 2>&1||fail 'download failed'
python - "$state_dir/download" "$state_dir/summary.json" <<'PY'
import json,pathlib,sys
root,out=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]);paths=sorted(root.rglob('single_layer_state_bridge.json'));assert len(paths)==5,paths;rows=[]
for p in paths:
 d=json.load(open(p));s=d['summary'];assert d['state_shape']==[5120] and d['optimizer_steps']==0 and d['raw_history_visible_to_decoder'] is False and d['persistent_history_kv_cache'] is False
 rows.append({'layer':d['layer'],'D_specific':s['D_specific'],'D_damage':s['D_damage'],'clean_n':s['clean_n'],'swap_flip_to_A':s['clean']['swap']['flip_to_A_rate'],'random_flip_to_A':s['clean']['random']['flip_to_A_rate'],'same_rule_flip_to_A':s['clean']['samemal']['flip_to_A_rate'],'noop_flip_to_A':s['clean']['noop']['flip_to_A_rate'],'scientific_gate_pass':s['scientific_gate_pass'],'source':str(p)})
payload={'stage':'single_hidden_vector_same_layer_bridge','scientific_gate_pass':any(r['scientific_gate_pass'] for r in rows),'rows':sorted(rows,key=lambda r:r['layer'])};out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2,sort_keys=True))
PY
touch "$state_dir/complete";log 'single-layer state bridge complete'
