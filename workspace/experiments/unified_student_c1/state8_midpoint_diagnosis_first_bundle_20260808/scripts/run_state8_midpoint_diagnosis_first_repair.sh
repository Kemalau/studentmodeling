#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MIDPOINT_DIAGNOSIS_FIRST_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${STATE8_CHECKPOINT_ROOT:?STATE8_CHECKPOINT_ROOT is required}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?EXPECTED_STUDENT_ADAPTER_SHA256 is required}"
: "${EXPECTED_STUDENT_STATE_SHA256:?EXPECTED_STUDENT_STATE_SHA256 is required}"

expected_root=/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4/checkpoint-2000
test "$STATE8_CHECKPOINT_ROOT" = "$expected_root"
test "$EXPECTED_STUDENT_ADAPTER_SHA256" = af09ff4b886fefe42b99d60edf467f5c3c7830b8caaa909668b189b12d9ae6f5
test "$EXPECTED_STUDENT_STATE_SHA256" = 2bcd1cf600b699010e2d7de8bd8fe3b310fa500c38ba66080fa1f5a54b1ede7f

export STUDENT_ROOT="$STATE8_CHECKPOINT_ROOT"
export STUDENT_MODEL_DIR="$STATE8_CHECKPOINT_ROOT/adapter"
export STUDENT_STATE_PATH="$STATE8_CHECKPOINT_ROOT/recurrent_state.pt"
source_receipt="$STATE8_CHECKPOINT_ROOT/checkpoint_completion.json"
normalized_root=/tmp/state8-midpoint-diagnosis-first-source
export STUDENT_RECEIPT="$normalized_root/source_receipt.json"
mkdir -p "$normalized_root"

python_bin="${VENV_ROOT:?VENV_ROOT is required}/bin/python"
"$python_bin" - "$source_receipt" "$STUDENT_MODEL_DIR/adapter_model.safetensors" \
  "$STUDENT_STATE_PATH" "$STUDENT_RECEIPT" <<'PY'
import collections, hashlib, json, pathlib, sys, torch
receipt, adapter, state, output = map(pathlib.Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
observed = {'adapter_model': digest(adapter), 'recurrent_state': digest(state)}
expected = {
    'adapter_model': 'af09ff4b886fefe42b99d60edf467f5c3c7830b8caaa909668b189b12d9ae6f5',
    'recurrent_state': '2bcd1cf600b699010e2d7de8bd8fe3b310fa500c38ba66080fa1f5a54b1ede7f',
}
x = json.load(receipt.open())
assert x == {'sha256': expected, 'status': 'complete', 'step': 2000}, (x, expected)
assert observed == expected, (observed, expected)
p = torch.load(state, map_location='cpu', weights_only=False, mmap=True)
c = p['config']; history = p['history']
assert p['step'] == 2000 and len(history) == 2000, (p['step'], len(history))
assert c['state_architecture'] == 'layerwise' and c['state_token_count'] == 8, c
assert c['final_inference_route'] == 'strict_state_only', c
assert c['taxonomy_label_contract'] == 'source_aware_no_answer_labels_v1', c
assert c['semantic_judge_authoritative'] is True, c
assert c['exact_label_metric_authoritative'] is False, c
assert len({int(row['global_position']) for row in history}) == 2000
dropout = collections.Counter(float(row['history_dropout']) for row in history)
assert dropout == {0.0: 1000, 0.25: 1000}, dropout
normalized = {
    'stage': 'state8_midpoint_step2000_weight_initialization_v1',
    'status': 'complete',
    'micro_steps': 2000,
    'real_examples_seen_once': True,
    'state_shape': [64, 8, 5120],
    'raw_history_visible_to_decoder': False,
    'persistent_history_kv_cache': False,
    'history_dropout_counts': {'0.0': 1000, '0.25': 1000},
    'initialization_only_not_complete_curriculum': True,
    'sha256': observed,
}
output.write_text(json.dumps(normalized, indent=2, sort_keys=True) + '\n')
PY

bash scripts/run_state8_stateonly_repair_ab.sh

answer_out="$AMLT_OUTPUT_DIR/eval/answer_c4_150"
mkdir -p "$answer_out"
test -f "$AMLT_OUTPUT_DIR/model/adapter_model.safetensors"
test -f "$AMLT_OUTPUT_DIR/model/recurrent_state.pt"
test -f /tmp/Qwen3-32B/model.safetensors.index.json

"$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
  --model /tmp/Qwen3-32B --adapter "$AMLT_OUTPUT_DIR/model" \
  --recurrent-state "$AMLT_OUTPUT_DIR/model/recurrent_state.pt" \
  --state-architecture layerwise \
  --jsonl /mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl \
  --pair-manifest manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json \
  --output-json "$answer_out/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --num-pairs 150 --seed 42

"$python_bin" - "$answer_out/recurrent_causal_gate.json" \
  "$AMLT_OUTPUT_DIR/metrics/publish_receipt.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1])); receipt = json.load(open(sys.argv[2]))
assert result['state_shape'] == [64, 8, 5120], result
assert result['raw_history_visible_to_decoder'] is False, result
assert result['persistent_history_kv_cache'] is False, result
assert result['artifact_sha256'] == receipt['artifact_sha256'], (result, receipt)
assert result['summary']['all_pairs_n'] == 150, result
PY
cp "$STUDENT_RECEIPT" "$AMLT_OUTPUT_DIR/metrics/midpoint_source_receipt.json"
