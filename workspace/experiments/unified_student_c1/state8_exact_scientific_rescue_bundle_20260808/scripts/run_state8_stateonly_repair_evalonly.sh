#!/usr/bin/env bash
set -euo pipefail

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-repair-evalonly-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed44_logits_v1
export RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export RECURRENT_EVAL=/tmp/state8_repair_evalonly_formal_u4/test.jsonl
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
export FORMAL_OUT="$AMLT_OUTPUT_DIR/eval/formal_t3_300"
export DIAGNOSIS_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_50"

mkdir -p "$HF_HUB_CACHE" "$(dirname "$RECURRENT_EVAL")" \
  "$FORMAL_OUT" "$DIAGNOSIS_OUT" "$AMLT_OUTPUT_DIR/metrics"
test "$(df --output=avail /tmp | tail -1)" -ge 83886080
for path in \
  "$RUN_DIR/final_adapter/adapter_model.safetensors" \
  "$RUN_DIR/recurrent_state.pt" \
  "$RUN_DIR/completion.json" \
  "$RUN_DIR/run_config.json" \
  "$RUN_DIR/state8_stateonly_teacher_repair.json" \
  "$RAW_EVAL" "$C4_DATA/test_same_template.jsonl" "$PAIRS"; do
  test -f "$path"
done
test "$(sha256sum "$RUN_DIR/final_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = 5d701f9259f6a5958229d98f0dcbfc39a0f8f718681882886a7449935af7befe
test "$(sha256sum "$RUN_DIR/recurrent_state.pt" | cut -d' ' -f1)" = 8b1a3051993f323af6ae8aec85d41e97172823869bdb5677dfce6865574d4d35
test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum "$C4_DATA/test_same_template.jsonl" | cut -d' ' -f1)" = 079f7634cc54dfd55dfc234503bb5925ab734b1ca74aa2f2a9a17cc4375a2c2f
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d' ' -f1)" = 80d9f7f5320b997f656bf5bc61ee10c7c31ee2d3de2c7e7b06bba66dd5dbac4a

python - "$RUN_DIR" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
completion=json.load(open(root/'completion.json'))
config=json.load(open(root/'run_config.json'))
repair=json.load(open(root/'state8_stateonly_teacher_repair.json'))
assert completion['status']=='complete' and completion['diagnostic'] is False,completion
assert completion['micro_steps']==1000 and completion['optimizer_steps']==250,completion
assert completion['all_causal_pairs_covered'] is True,completion
assert completion['sha256']['adapter_model']=='5d701f9259f6a5958229d98f0dcbfc39a0f8f718681882886a7449935af7befe',completion
assert completion['sha256']['recurrent_state']=='8b1a3051993f323af6ae8aec85d41e97172823869bdb5677dfce6865574d4d35',completion
assert config['state_architecture']=='layerwise' and config['state_token_count']==8,config
assert config['final_inference_route']=='strict_state_only',config
assert config['history_dropout_schedule']==[1.0],config
assert config['online_teacher_scope']=='qualified_external_full_history_answer_logits',config
assert float(config['online_kd_weight'])==1.0 and float(config['label_auxiliary_weight'])==4.0,config
assert config['output_schema']=='label_diagnosis_evidence_v1',config
assert repair['trajectory_weight']==0.0 and repair['state_token_count']==8,repair
PY

python -m pytest -q tests/test_evaluate_recurrent_formal_protocols.py \
  tests/test_evaluate_recurrent_diagnosis_c4.py
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json /tmp/state8_repair_evalonly_formal_u4/audit.json \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/evaluate_recurrent_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --eval-jsonl "$RECURRENT_EVAL" --output-dir "$FORMAL_OUT" \
  --state-token-alias '<|vision_pad|>' --tasks diagnose_misconception \
  --sources eedi_mining malrule --max-samples-per-group 100 \
  --max-length 1536 --max-new-tokens 160 --state-batch-size 2 \
  --candidate-row-batch-size 2 --seed 44 --save-unredacted-predictions --t3-natural-schema

python -u scripts/evaluate_recurrent_diagnosis_c4.py \
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \
  --jsonl "$C4_DATA/test_same_template.jsonl" --pair-manifest "$PAIRS" \
  --output-dir "$DIAGNOSIS_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 160 --num-pairs 50 --seed 42 \
  --expected-adapter-sha256 5d701f9259f6a5958229d98f0dcbfc39a0f8f718681882886a7449935af7befe \
  --expected-recurrent-state-sha256 8b1a3051993f323af6ae8aec85d41e97172823869bdb5677dfce6865574d4d35

python - "$FORMAL_OUT/metrics.json" "$DIAGNOSIS_OUT/metrics.json" \
  "$AMLT_OUTPUT_DIR/metrics/evalonly_receipt.json" <<'PY'
import json,pathlib,sys
formal=json.load(open(sys.argv[1])); c4=json.load(open(sys.argv[2])); out=pathlib.Path(sys.argv[3])
expected={'adapter_model':'5d701f9259f6a5958229d98f0dcbfc39a0f8f718681882886a7449935af7befe',
          'recurrent_state':'8b1a3051993f323af6ae8aec85d41e97172823869bdb5677dfce6865574d4d35'}
assert formal['sampled_rows']==300 and formal['artifact_sha256']==expected,formal
assert formal['state_shape']==[64,8,5120] and formal['raw_history_visible_to_decoder'] is False,formal
assert c4['num_pairs']==50 and c4['artifact_sha256']==expected,c4
assert c4['state_shape']==[64,8,5120] and c4['raw_history_visible_to_decoder'] is False,c4
receipt={'stage':'state8_stateonly_repair_evalonly_v1','status':'complete',
         'repair_arm':'logits','formal_rows':300,'c4_pairs':50,
         'artifact_sha256':expected,'raw_history_visible_to_decoder':False}
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
cp /tmp/state8_repair_evalonly_formal_u4/audit.json "$AMLT_OUTPUT_DIR/metrics/formal_manifest_audit.json"
