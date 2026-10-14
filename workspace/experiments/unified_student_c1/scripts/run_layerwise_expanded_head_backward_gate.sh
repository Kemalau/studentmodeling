#!/usr/bin/env bash
set -euo pipefail

: "${ROW_INDEX:?ROW_INDEX is required}"
: "${EXPECTED_TOKENS:?EXPECTED_TOKENS is required}"
: "${EXPECTED_SUFFIX:?EXPECTED_SUFFIX is required}"
case "$ROW_INDEX:$EXPECTED_TOKENS:$EXPECTED_SUFFIX" in
  177752:549:*:233:1|110499:653:*:0002079:malrule_identity|68865:721:*:0002130:malrule_identity) ;;
  *) echo "Unregistered backward-gate row" >&2; exit 2 ;;
esac

export PATH="/tmp/studentmodeling-layerwise-expanded-head-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export RAW_FULL=/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl
export RAW_SPLIT=/tmp/layerwise_expanded_head_full_trainvalid_v1
export RECURRENT_SPLIT=/tmp/layerwise_expanded_head_recurrent_full259845_u4_v1
export FINAL_JSONL="$RECURRENT_SPLIT/final_fit_trainvalid.jsonl"
export C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
export CAUSAL_DATA=/tmp/layerwise_expanded_head_cf4_u4_v1
export INIT_ADAPTER=/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter
export CACHE_ROOT=/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way
export SELECTED_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/layerwise_trajectory_curriculum_r256_scale_seed42_n200_20260731
export RUN_DIR="/mnt/amulet/studentmodeling/diagnostics/layerwise_expanded_head_gate_${EXPECTED_TOKENS}_20260802"

mkdir -p "$HF_HUB_CACHE" "$RAW_SPLIT" "$RECURRENT_SPLIT" "$CAUSAL_DATA" \
  "$AMLT_OUTPUT_DIR/model" "$AMLT_OUTPUT_DIR/metrics"
test "$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)" = b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440
test "$(sha256sum scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py | cut -d' ' -f1)" = c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca
test "$(sha256sum tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py | cut -d' ' -f1)" = a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274

finalize_gate() {
  python - "$RUN_DIR" "$EXPECTED_TOKENS" "$EXPECTED_SUFFIX" "$ROW_INDEX" <<'PY'
import hashlib, json, pathlib, sys
root, tokens, suffix, row_index = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
required = [
    'trainer_history.json', 'final_adapter/adapter_model.safetensors',
    'recurrent_state.pt', 'validation_metrics.json',
    'causal_validation_metrics.json', 'run_config.json', 'gradient_audit.json',
]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise ValueError(f'incomplete backward-gate artifacts: {missing}')
history = json.load(open(root / 'trainer_history.json', encoding='utf-8'))
config = json.load(open(root / 'run_config.json', encoding='utf-8'))
assert len(history) == 1 and history[0]['example_id'].endswith(suffix), history
assert history[0]['direct_offload_used'] is True, history[0]
assert history[0]['direct_tokens'] == tokens, history[0]
assert history[0]['peak_cuda_allocated_gib'] < 79.0, history[0]
assert history[0]['peak_cuda_reserved_gib'] < 79.0, history[0]
assert history[0].get('causal_pair_id') is None, history[0]
assert config['diagnostic_real_index'] == row_index, config
assert config['diagnostic_max_micro_steps'] == 1, config
assert config['planned_causal_exposures'] == 0, config
assert config['causal_train_pairs'] == 1051, config
assert config['causal_repeat_fraction'] == 0.05, config
assert config['causal_weight'] == 1.0, config
completion = {
    'status': 'diagnostic_complete',
    'diagnostic': True,
    'gate_scope': 'single_real_row_backward_only',
    'expected_noncausal_row': True,
    'formal_causal_audit_waived': False,
    'micro_steps': 1,
    'optimizer_steps': 1,
    'real_examples_seen': 1,
    'causal_pairs_seen': 0,
    'contains_causal_pair': False,
    'history_rows': 1,
    'history_integrity_pass': True,
}
(root / 'completion.json').write_text(json.dumps(completion, indent=2, sort_keys=True) + '\n', encoding='utf-8')
runtime = pathlib.Path('scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py')
receipt = {
    'status': 'passed', 'row_index': row_index,
    'example_id': history[0]['example_id'],
    'direct_tokens': history[0]['direct_tokens'],
    'direct_offload_used': history[0]['direct_offload_used'],
    'peak_cuda_allocated_gib': history[0]['peak_cuda_allocated_gib'],
    'peak_cuda_reserved_gib': history[0]['peak_cuda_reserved_gib'],
    'runtime_sha256': hashlib.sha256(runtime.read_bytes()).hexdigest(),
    'gate_scope': 'single_real_row_backward_only',
}
(root / 'expanded_head_backward_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
  cp "$RUN_DIR/expanded_head_backward_receipt.json" \
    "$RUN_DIR/trainer_history.json" "$RUN_DIR/completion.json" \
    "$AMLT_OUTPUT_DIR/metrics/"
}

existing_complete=true
for artifact in trainer_history.json final_adapter/adapter_model.safetensors \
  recurrent_state.pt validation_metrics.json causal_validation_metrics.json \
  run_config.json gradient_audit.json; do
  [[ -f "$RUN_DIR/$artifact" ]] || existing_complete=false
done
if [[ "${FINALIZE_EXISTING:-false}" == true || "$existing_complete" == true ]]; then
  finalize_gate
  exit 0
fi

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -f "$RAW_FULL"
test -f "$C4_DATA/train.jsonl"
test -f "$C4_DATA/val.jsonl"
test -f "$INIT_ADAPTER/adapter_model.safetensors"
test -f "$CACHE_ROOT/integrity.json"
test -f "$CACHE_ROOT/manifest.json"
test -f "$SELECTED_RUN_DIR/completion.json"
test -f "$SELECTED_RUN_DIR/final_adapter/adapter_model.safetensors"
test -f "$SELECTED_RUN_DIR/recurrent_state.pt"
cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py
cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py
python -m unittest discover -s tests -p 'test_train_recurrent_student_m4.py' -v
python -m unittest discover -s tests -p 'test_layerwise_recurrent_student_model.py' -v
python -m unittest discover -s tests -p 'test_verify_sdpa_aligned_mask_equivalence.py' -v

python scripts/split_full_mix_by_original_split.py \
  --input-jsonl "$RAW_FULL" --output-dir "$RAW_SPLIT"
python scripts/build_unified_recurrent_manifest.py \
  --train-jsonl "$RAW_SPLIT/train.jsonl" --valid-jsonl "$RAW_SPLIT/valid.jsonl" \
  --output-dir "$RECURRENT_SPLIT" --max-updates 4 --seed 20260730
python scripts/merge_recurrent_full_corpus.py \
  --train-jsonl "$RECURRENT_SPLIT/train.jsonl" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --output-jsonl "$FINAL_JSONL" --manifest "$RECURRENT_SPLIT/final_fit_manifest.json"
python scripts/build_recurrent_causal_pairs.py \
  --train-source "$C4_DATA/train.jsonl" --valid-source "$C4_DATA/val.jsonl" \
  --train-pairs manifests/cf4_balanced_paired_state_swap_train_seed20260731_v1.json \
  --valid-pairs manifests/cf4_balanced_paired_state_swap_valid_seed20260801_v1.json \
  --output-dir "$CAUSAL_DATA" --max-updates 4
test "$(sha256sum "$FINAL_JSONL" | cut -d' ' -f1)" = 9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd

python - "$FINAL_JSONL" "$ROW_INDEX" "$EXPECTED_TOKENS" "$EXPECTED_SUFFIX" <<'PY'
import json, pathlib, sys
from transformers import AutoTokenizer
from train_recurrent_student_m4 import direct_supervised_batch
path, index, tokens, suffix = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with path.open(encoding='utf-8') as handle:
    rows = [json.loads(line) for line in handle]
row = rows[index]
assert str(row['example_id']).endswith(suffix), row['example_id']
tokenizer = AutoTokenizer.from_pretrained('/mnt/amulet/studentmodeling/models/Qwen3-32B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
batch = direct_supervised_batch(tokenizer, row['direct_text'], row['target'], '<|vision_pad|>', 1536, 'cpu')
assert int(batch['input_ids'].shape[1]) == tokens, batch['input_ids'].shape
PY

rm -rf "$LOCAL_MODEL"; mkdir -p "$LOCAL_MODEL"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

set +e
python -u scripts/train_recurrent_student_m4.py \
  --model "$LOCAL_MODEL" --init-adapter "$INIT_ADAPTER" \
  --init-run-dir "$SELECTED_RUN_DIR" \
  --state-architecture layerwise --layerwise-operation-checkpointing \
  --layerwise-transition-rank 256 \
  --train-jsonl "$FINAL_JSONL" --valid-jsonl "$RECURRENT_SPLIT/valid.jsonl" \
  --causal-train-jsonl "$CAUSAL_DATA/train.jsonl" \
  --causal-valid-jsonl "$CAUSAL_DATA/valid.jsonl" \
  --teacher-cache "$CACHE_ROOT" --behavior-kd-weight 0.5 --behavior-kd-temperature 1.0 \
  --output-dir "$RUN_DIR" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --direct-offload-min-tokens 512 \
  --learning-rate 5e-7 --lr-scheduler cosine --warmup-ratio 0.03 \
  --generation-weight 1.0 --consistency-weight 0.25 --direct-replay-weight 0.25 \
  --causal-weight 1.0 --causal-margin 0.5 \
  --causal-preference-reduction sum --causal-objective-mode specificity_triplet \
  --causal-repeat-fraction 0.05 --normalize-causal-exposure \
  --gradient-accumulation-steps 1 --eval-limit 1 --causal-eval-limit 1 \
  --diagnostic-max-micro-steps 1 --diagnostic-real-index "$ROW_INDEX" \
  --save-steps 1 --seed 42 2>&1 | tee "$RUN_DIR/backward_gate_train.log"
train_rc=${PIPESTATUS[0]}
set -e
if (( train_rc != 0 )); then
  test "$train_rc" -eq 1
  test "$(grep -Fc 'RuntimeError: M4 diagnostic missed causal path:' "$RUN_DIR/backward_gate_train.log")" -eq 1
fi
finalize_gate
