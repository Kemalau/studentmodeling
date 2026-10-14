#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
pipeline="$project_dir/results/layerwise_unified_final_pipeline_20260731"
state_dir="$pipeline/eedi_open_t3_t6_repair/recovered_balanced_lr5e7/local_round4500_t6_full"
checkpoint="$state_dir/checkpoint-round-4500"
remote_prefix=studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_full_open_t3/state_path/lr5e-7_start17000_stop4500_08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a/checkpoint-round-4500
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
account=csnewsandfeeds4150361735
container=amulet
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"

mkdir -p "$checkpoint/adapter" "$state_dir/eval"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
fail(){ printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; touch "$state_dir/needs_attention"; exit 1; }
rm -f "$state_dir/needs_attention" "$state_dir/complete"

download(){
  local remote="$1" local_path="$2" partial="$2.partial.$$"
  [[ -s "$local_path" ]] && return 0
  rm -f "$partial"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$remote" --file "$partial" \
    --overwrite -o none >> "$state_dir/download.log" 2>&1 || return 1
  mv "$partial" "$local_path"
}
for target in checkpoint_completion.json adapter/adapter_config.json adapter/adapter_model.safetensors recurrent_state.pt; do
  download "$remote_prefix/$target" "$checkpoint/$target" || fail "download failed target=$target"
done
"$python_bin" - "$checkpoint" <<'PY' || fail "round-4500 artifact audit failed"
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); receipt=json.load(open(root/'checkpoint_completion.json'))
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
observed={'adapter_model':digest(root/'adapter/adapter_model.safetensors'),'recurrent_state':digest(root/'recurrent_state.pt')}
assert receipt=={'status':'complete','round':4500,'world_size':4,'sha256':observed},(receipt,observed)
PY
test "$($python_bin - <<'PY'
import hashlib
print(hashlib.sha256(open('data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl','rb').read()).hexdigest())
PY
)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525 || fail "frozen eval drift"

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do heartbeat; sleep 30; done
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
    --model "$model" --adapter "$checkpoint/adapter" \
    --recurrent-state "$checkpoint/recurrent_state.pt" --state-architecture layerwise \
    --eval-jsonl "$eval_jsonl" --output-dir "$state_dir/eval" \
    --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 128 \
    --state-batch-size 2 --candidate-row-batch-size 2 \
    --tasks predict_error_persistence --seed 42 --save-unredacted-predictions \
    >> "$state_dir/eval/live.log" 2>&1 || fail "full T6 evaluation failed"
flock -u 10
"$python_bin" - "$state_dir/eval/metrics.json" "$state_dir/summary.json" <<'PY' || fail "T6 summary failed"
import json,pathlib,sys
x=json.load(open(sys.argv[1])); groups=x['metrics_by_group']; selected=[v for k,v in groups.items() if k.startswith('predict_error_persistence/')]
support=sum(v['n'] for v in selected); score=sum(v['macro_f1']*v['n'] for v in selected)/support
assert support==976,support
out={'stage':'balanced_lr5e7_round4500_T6_full','support':support,'macro_f1_default_threshold':score,'calibrated_threshold_pending':True}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
touch "$state_dir/complete"; heartbeat
