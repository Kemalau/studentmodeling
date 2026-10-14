#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/task_retention_promoted_final_fit/round1000_c4_n150_multiseed"
local_dir="$state_dir/local_seed97"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/local_seed97_controller.lock"
flock -n 9 || exit 0
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/local_seed97_controller.heartbeat"; }
fail(){ printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$state_dir/local_seed97_controller.log"; touch "$state_dir/local_seed97_needs_attention"; exit 1; }
rm -f "$state_dir/local_seed97_needs_attention" "$state_dir/local_seed97_complete"

while [[ ! -f "$local_dir/exit_code" ]]; do heartbeat; sleep 30; done
[[ "$(cat "$local_dir/exit_code")" == 0 ]] || fail "local seed97 C4 exited nonzero"
test -s "$local_dir/recurrent_causal_gate.json" || fail "local seed97 C4 result missing"

"$python_bin" - "$local_dir/recurrent_causal_gate.json" "$local_dir/summary.json" <<'PY' || exit 1
import json,pathlib,sys
x=json.load(open(sys.argv[1]))['summary']; c=x['clean']
assert x['all_pairs_n']==150 and x['clean_n']>0,x
out={'stage':'layerwise_checkpoint_c4','artifact_label':'promoted-eedi-full-open-t3-state-path-lr2e6',
     'checkpoint_round':1000,'pair_seed':97,'diagnostic_only':True,'not_a_formal_gate':True,
     'all_pairs_n':x['all_pairs_n'],'clean_n':x['clean_n'],
     'swap':c['swap']['flip_to_A_rate'],'random':c['random']['flip_to_A_rate'],
     'same_rule':c['samemal']['flip_to_A_rate'],'noop':c['noop']['flip_to_A_rate'],
     'D_specific':x['D_specific'],'D_damage':x['D_damage'],
     'scientific_gate_pass':x['scientific_gate_pass'],'checks':x['checks']}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

mapfile -t remote < <(find "$state_dir/all_pass_download" -type f -path '*/eval/c4/summary.json' | sort)
[[ ${#remote[@]} -eq 2 ]] || fail "expected two passed remote summaries"
"$python_bin" - "$state_dir/summary.json" "${remote[@]}" "$local_dir/summary.json" <<'PY' || exit 1
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1]); rows=[json.load(open(path)) for path in sys.argv[2:]]
assert {row['pair_seed'] for row in rows}=={13,42,97},rows
assert all(row['checkpoint_round']==1000 and row['all_pairs_n']==150 for row in rows),rows
rows=sorted(rows,key=lambda row:row['pair_seed']); values=[row['D_specific'] for row in rows]
summary={'stage':'layerwise_checkpoint_c4_multiseed','checkpoint_round':1000,
 'diagnostic_only':True,'pair_seeds':[13,42,97],'rows':rows,
 'D_specific_mean':statistics.mean(values),'D_specific_min':min(values),
 'scientific_gate_pass_count':sum(bool(row['scientific_gate_pass']) for row in rows)}
summary['two_of_three_gate_pass']=summary['scientific_gate_pass_count']>=2
temporary=out.with_suffix('.json.partial')
temporary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); temporary.replace(out)
PY
touch "$state_dir/local_seed97_complete"
rm -f "$state_dir/needs_attention"
touch "$state_dir/complete"
heartbeat
