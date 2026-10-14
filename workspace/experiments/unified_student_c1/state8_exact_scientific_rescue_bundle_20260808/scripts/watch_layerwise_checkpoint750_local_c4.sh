#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
checkpoint="$pipeline/checkpoint750_local_audit"
state_dir="$checkpoint/c4_full"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
model="/home/v-yuejiali/studentmodeling/models/Qwen3-32B"
mkdir -p "$state_dir/progress"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

heartbeat
python - "$checkpoint/checkpoint_completion.json" "$checkpoint/sha256.txt" <<'PY' \
  || fail "round-750 local artifact receipt/hash audit failed"
import json,pathlib,sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
assert receipt.get('status')=='complete' and receipt.get('round')==750,receipt
assert receipt.get('world_size')==16,receipt
lines=pathlib.Path(sys.argv[2]).read_text(encoding='utf-8').splitlines()
observed={line.split()[1]:line.split()[0] for line in lines if line.split()}
adapter='results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/adapter/adapter_model.safetensors'
state='results/layerwise_unified_final_pipeline_20260731/checkpoint750_local_audit/recurrent_state.pt'
assert observed[adapter]==receipt['sha256']['adapter_model'],observed
assert observed[state]==receipt['sha256']['recurrent_state'],observed
PY
python scripts/audit_layerwise_distributed_checkpoint.py \
  --checkpoint "$checkpoint/recurrent_state.pt" --expected-round 750 \
  --output "$checkpoint/checkpoint_resume_audit.json" \
  >> "$state_dir/controller.log" 2>&1 \
  || fail "round-750 exact-resume audit failed"

if [[ ! -f "$state_dir/recurrent_causal_gate.json" ]]; then
  heartbeat
  (
    while true; do
      heartbeat
      sleep 60
    done
  ) &
  heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts/frozen_layerwise_finalfit:scripts \
    "$python_bin" -u -m evaluate_recurrent_causal_gate \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise \
      --jsonl data/processed/malrule_c1_clean/test_same_template.jsonl \
      --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
      --output-json "$state_dir/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' --max-length 1536 \
      --num-pairs 150 --seed 42 \
      --progress-dir "$state_dir/progress" \
      --progress-key 'round750-adapter428d32bc-state598901f8' \
      >> "$state_dir/live.log" 2>&1
  eval_rc=$?
  if [[ -n "${heartbeat_pid:-}" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  ((eval_rc==0)) || fail "round-750 local C4 evaluation failed"
fi

python - "$state_dir/recurrent_causal_gate.json" "$state_dir/summary.json" <<'PY' \
  || fail "round-750 C4 summary invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); s=x['summary']
assert s['all_pairs_n']==150 and s['clean_n']>0,s
assert set(s['clean'])=={'noop','swap','random','samemal'},s
out={
 'stage':'layerwise_finalfit_checkpoint750_local_c4',
 'diagnostic_only':True,
 'checkpoint_round':750,
 'clean_n':s['clean_n'],
 'swap':s['clean']['swap']['flip_to_A_rate'],
 'random':s['clean']['random']['flip_to_A_rate'],
 'same_rule':s['clean']['samemal']['flip_to_A_rate'],
 'noop':s['clean']['noop']['flip_to_A_rate'],
 'D_specific':s['D_specific'],
 'scientific_gate_pass_at_checkpoint':s['scientific_gate_pass'],
}
open(sys.argv[2],'w',encoding='utf-8').write(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
PY

python - "$pipeline" "$state_dir/checkpoint_causal_curve.json" <<'PY' \
  || fail "checkpoint causal curve construction failed"
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
paths={
 250:root/'checkpoint250_local/c4_full/recurrent_causal_gate.json',
 500:root/'checkpoint500_c4/download/layerwise-finalfit-checkpoint500-c4-seed42-20260801-recovery1/layerwise-finalfit-checkpoint500-c4-seed42/eval/c4/recurrent_causal_gate.json',
 750:root/'checkpoint750_local_audit/c4_full/recurrent_causal_gate.json',
}
rows=[]
for step,path in paths.items():
 x=json.load(open(path,encoding='utf-8'))['summary']; c=x['clean']
 rows.append({'round':step,'clean_n':x['clean_n'],'swap':c['swap']['flip_to_A_rate'],
              'random':c['random']['flip_to_A_rate'],'same_rule':c['samemal']['flip_to_A_rate'],
              'noop':c['noop']['flip_to_A_rate'],'D_specific':x['D_specific']})
open(sys.argv[2],'w',encoding='utf-8').write(json.dumps({'stage':'checkpoint_causal_curve','rows':rows},indent=2,sort_keys=True)+'\n')
print(json.dumps(rows,indent=2))
PY
touch "$state_dir/complete"
heartbeat
log "round-750 local C4 and checkpoint causal curve complete"
