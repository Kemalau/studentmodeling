#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
source_dir="$pipeline/task_retention_promoted_final_fit/local_round1250_trajectory"
checkpoint="$source_dir/checkpoint-round-1250"
state_dir="$pipeline/task_retention_promoted_final_fit/round1250_c4_n150_multiseed/local_seed42"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
jsonl="$project_dir/data/processed/malrule_c1_clean/test_same_template.jsonl"
pair_manifest="$project_dir/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"

mkdir -p "$state_dir/progress"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail(){ printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; touch "$state_dir/needs_attention"; exit 1; }
rm -f "$state_dir/needs_attention" "$state_dir/complete"

"$python_bin" - "$checkpoint" "$jsonl" "$pair_manifest" <<'PY' \
  || fail "round-1250 local C4 artifact audit failed"
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); receipt=json.load(open(root/'checkpoint_completion.json'))
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),
          'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt=={'status':'complete','round':1250,'world_size':16,'sha256':observed},(receipt,observed)
assert hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()=='079f7634cc54dfd55dfc234503bb5925ab734b1ca74aa2f2a9a17cc4375a2c2f'
assert hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()=='edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8'
manifest=json.load(open(sys.argv[3])); assert manifest.get('seed')==42 and len(manifest.get('pairs') or [])==150,manifest
PY

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do heartbeat; sleep 30; done

progress_key="round1250-local-n150-seed42:adapter=fbe5b392de372e84bf89caef6c1f98f452c842a5a478c739bece84e6547d1d60:state=95f2721a6b933f9d0ef029a3853cd607e9510c49c5ef1745a97f8ff8a6e883b0"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
    --model "$model" --adapter "$checkpoint/adapter" \
    --recurrent-state "$checkpoint/recurrent_state.pt" --state-architecture layerwise \
    --jsonl "$jsonl" --pair-manifest "$pair_manifest" \
    --output-json "$state_dir/recurrent_causal_gate.json" \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 --seed 42 \
    --progress-dir "$state_dir/progress" --progress-key "$progress_key" \
    >> "$state_dir/live.log" 2>&1 || fail "round-1250 local seed42 C4 failed"
flock -u 10

"$python_bin" - "$state_dir/recurrent_causal_gate.json" "$state_dir/summary.json" <<'PY' \
  || fail "round-1250 local seed42 summary failed"
import json,pathlib,sys
x=json.load(open(sys.argv[1]))['summary']; c=x['clean']
assert x['all_pairs_n']==150 and x['clean_n']>0,x
out={'stage':'layerwise_checkpoint_c4','artifact_label':'promoted-eedi-full-open-t3-state-path-lr2e6',
     'checkpoint_round':1250,'pair_seed':42,'diagnostic_only':True,'not_a_formal_gate':True,
     'all_pairs_n':x['all_pairs_n'],'clean_n':x['clean_n'],
     'swap':c['swap']['flip_to_A_rate'],'random':c['random']['flip_to_A_rate'],
     'same_rule':c['samemal']['flip_to_A_rate'],'noop':c['noop']['flip_to_A_rate'],
     'D_specific':x['D_specific'],'D_damage':x['D_damage'],
     'scientific_gate_pass':x['scientific_gate_pass'],'checks':x['checks']}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state_dir/complete"; heartbeat
