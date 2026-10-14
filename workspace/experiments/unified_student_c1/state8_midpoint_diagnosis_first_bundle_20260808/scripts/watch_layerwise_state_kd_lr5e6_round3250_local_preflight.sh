#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_state_kd_lr_scan/local_lr5e6_round3250_preflight"
checkpoint="$state_dir/checkpoint-round-3250"
adapter="$checkpoint/adapter"
model="/home/v-yuejiali/studentmodeling/models/Qwen3-32B"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
c4_jsonl="$project_dir/data/processed/malrule_c1_clean/test_same_template.jsonl"
pair_manifest="$project_dir/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
account=csnewsandfeeds4150361735
container=amulet
prefix="studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_full/state_kd/lr5e-6_start12000_stop4250_97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440/checkpoint-round-3250"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
mkdir -p "$state_dir" "$adapter"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }
download_blob(){
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$1" --file "$2" --overwrite -o none \
    >> "$state_dir/download.log" 2>&1
}

receipt="$checkpoint/checkpoint_completion.json"
while [[ ! -s "$receipt" ]]; do
  heartbeat
  exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$prefix/checkpoint_completion.json" \
    --query exists -o tsv 2>/dev/null || true)"
  if [[ "$exists" == true ]]; then
    mkdir -p "$checkpoint"
    download_blob "$prefix/checkpoint_completion.json" "$receipt" || { sleep 120; continue; }
  else
    sleep 120
  fi
done

"$python_bin" - "$receipt" <<'PY' || fail "round-3250 receipt invalid"
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('status')=='complete' and x.get('round')==3250,x
assert x.get('world_size')==4,x
assert set(x.get('sha256') or {})=={'adapter_model','recurrent_state'},x
PY

[[ -s "$adapter/adapter_model.safetensors" ]] || \
  download_blob "$prefix/adapter/adapter_model.safetensors" "$adapter/adapter_model.safetensors" || fail "adapter download failed"
[[ -s "$adapter/adapter_config.json" ]] || \
  download_blob "$prefix/adapter/adapter_config.json" "$adapter/adapter_config.json" || fail "adapter config download failed"
[[ -s "$checkpoint/recurrent_state.pt" ]] || \
  download_blob "$prefix/recurrent_state.pt" "$checkpoint/recurrent_state.pt" || fail "state download failed"

"$python_bin" - "$receipt" "$adapter/adapter_model.safetensors" "$checkpoint/recurrent_state.pt" "$state_dir/artifact_audit.json" <<'PY' \
  || fail "round-3250 artifact hash audit failed"
import hashlib,json,pathlib,sys
receipt=json.load(open(sys.argv[1],encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
observed={'adapter_model':digest(sys.argv[2]),'recurrent_state':digest(sys.argv[3])}
assert observed==receipt['sha256'],(observed,receipt)
path=pathlib.Path(sys.argv[4]); path.write_text(json.dumps({'stage':'state_kd_lr5e6_round3250_local_artifact_audit','status':'passed','round':3250,'world_size':4,'sha256':observed},indent=2,sort_keys=True)+'\n')
PY

test -f "$model/model.safetensors.index.json" || fail "local model missing"
test -f "$eval_jsonl" || fail "formal manifest missing"
test -f "$c4_jsonl" || fail "C4 data missing"
test -f "$pair_manifest" || fail "C4 pair manifest missing"

if [[ ! -s "$state_dir/c4/recurrent_causal_gate.json" ]]; then
  mkdir -p "$state_dir/c4/progress"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
      --model "$model" --adapter "$adapter" --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --jsonl "$c4_jsonl" --pair-manifest "$pair_manifest" \
      --output-json "$state_dir/c4/recurrent_causal_gate.json" --state-token-alias '<|vision_pad|>' \
      --max-length 1536 --num-pairs 150 --seed 42 --progress-dir "$state_dir/c4/progress" \
      --progress-key state-kd-lr5e6-round3250-local-n150 >> "$state_dir/c4/live.log" 2>&1 \
    || fail "local C4 failed"
fi

if [[ ! -s "$state_dir/formal_n10/metrics.json" ]]; then
  mkdir -p "$state_dir/formal_n10"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
      --model "$model" --adapter "$adapter" --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" --output-dir "$state_dir/formal_n10" \
      --state-token-alias '<|vision_pad|>' --max-samples-per-group 10 --max-length 1536 \
      --max-new-tokens 128 --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
      --save-unredacted-predictions >> "$state_dir/formal_n10/live.log" 2>&1 \
    || fail "local formal n=10 failed"
fi

"$python_bin" - "$state_dir/c4/recurrent_causal_gate.json" "$state_dir/formal_n10/metrics.json" \
  "$state_dir/formal_n10/predictions.jsonl" "$state_dir/summary.json" <<'PY' || fail "summary failed"
import collections,json,pathlib,sys
c4=json.load(open(sys.argv[1],encoding='utf-8'))['summary']; formal=json.load(open(sys.argv[2],encoding='utf-8'))
pred=[json.loads(x) for x in open(sys.argv[3],encoding='utf-8') if x.strip()]
t3=[x for x in pred if x.get('task')=='diagnose_misconception']; outputs=collections.Counter(str(x.get('raw_output','')) for x in t3)
clean=c4['clean']; out={'stage':'state_kd_lr5e6_round3250_local_preflight','diagnostic_only':True,'not_a_formal_gate':True,'checkpoint_round':3250,'formal_sampled_rows':formal.get('sampled_rows'),'metrics_by_group':formal.get('metrics_by_group'),'T3_output_diversity':{'support':len(t3),'unique_outputs':len(outputs),'top_output_share':outputs.most_common(1)[0][1]/len(t3) if t3 else None},'C4':{'clean_n':c4['clean_n'],'swap':clean['swap']['flip_to_A_rate'],'random':clean['random']['flip_to_A_rate'],'same_rule':clean['samemal']['flip_to_A_rate'],'noop':clean['noop']['flip_to_A_rate'],'D_specific':c4['D_specific'],'scientific_gate_pass':c4['scientific_gate_pass']}}
path=pathlib.Path(sys.argv[4]); path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state_dir/complete"
heartbeat
log "state-KD lr5e-6 round-3250 local preflight complete"
