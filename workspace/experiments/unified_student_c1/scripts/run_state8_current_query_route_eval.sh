#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_CURRENT_QUERY_ROUTE_EVAL_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${ROUTE_MODE:?ROUTE_MODE is required}"
: "${CHECKPOINT_ADAPTER_SHA256:?CHECKPOINT_ADAPTER_SHA256 is required}"
: "${CHECKPOINT_STATE_SHA256:?CHECKPOINT_STATE_SHA256 is required}"
: "${CHECKPOINT_RECEIPT_SHA256:?CHECKPOINT_RECEIPT_SHA256 is required}"

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-current-query-route-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

case "$ROUTE_MODE" in
  state_plus_current_item)
    evaluator=scripts/evaluate_recurrent_formal_protocols_diagnosis_first_current_query.py
    evaluator_sha=1b892ad41e8c053a7990ed85e3707fbe5aa463b8cb6fb97028fcde80e14db966
    ;;
  current_item_only)
    evaluator=scripts/evaluate_recurrent_formal_protocols_diagnosis_first_current_query_item_only.py
    evaluator_sha=27c95d0eb2469b11c38da4dc75d03a4a7d26946ee96393bd7507848c8929782f
    ;;
  *) echo "Unknown ROUTE_MODE=$ROUTE_MODE" >&2; exit 2 ;;
esac

CHECKPOINT=/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed3072_taskanchor_source_anchor_updater_mse1_v1/checkpoint-100
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
RAW_EVAL=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
RECURRENT_EVAL=/tmp/state8_current_query_route_eval/test.jsonl
OUTPUT="$AMLT_OUTPUT_DIR/eval/formal_t3_75"
mkdir -p "$MODEL_LOCAL" "$(dirname "$RECURRENT_EVAL")" "$OUTPUT" "$AMLT_OUTPUT_DIR/metrics"

test "$(sha256sum "$RAW_EVAL" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test "$(sha256sum scripts/state8_current_query_routing.py | cut -d' ' -f1)" = 8663ae9e7f66e19cf983aeffa6a244ab24d9a25b672874a3a645f3265607cb5a
test "$(sha256sum "$evaluator" | cut -d' ' -f1)" = "$evaluator_sha"
python scripts/audit_state8_source_anchor_checkpoint.py \
  --checkpoint-root "$CHECKPOINT" --arm updater_mse1 --step 100 \
  --adapter-sha256 "$CHECKPOINT_ADAPTER_SHA256" \
  --state-sha256 "$CHECKPOINT_STATE_SHA256" \
  --receipt-sha256 "$CHECKPOINT_RECEIPT_SHA256" \
  --output "$AMLT_OUTPUT_DIR/metrics/artifact_audit.json"
python scripts/build_recurrent_evaluation_manifest.py \
  --input-jsonl "$RAW_EVAL" --output-jsonl "$RECURRENT_EVAL" \
  --audit-json "$AMLT_OUTPUT_DIR/metrics/manifest_audit.json" \
  --max-updates 4 --seed 20260730
test "$(sha256sum "$RECURRENT_EVAL" | cut -d' ' -f1)" = a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525
python - "$RECURRENT_EVAL" <<'PY'
import json,re,sys
from state8_current_query_routing import apply_current_query_routing
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
diagnosis=[row for row in rows if row.get('task')=='diagnose_misconception']
audit=apply_current_query_routing(diagnosis)
assert audit['current_query_routed_rows']==1494,audit
assert audit['current_query_source_eedi_mining']==894,audit
assert audit['current_query_source_malrule']==600,audit
for row in diagnosis:
  prompt=row['decode_text']
  assert '<student_history>' not in prompt.lower(),row['example_id']
  current=re.search(r'<current_item>(.*?)</current_item>',prompt,re.S)
  assert current,row['example_id']
  item=json.loads(current.group(1))
  assert not ({'interaction_id','question_id','student_answer','student_response',
               'incorrect_answer','target_label'} & set(item)),item
  if row.get('source')=='eedi_mining':
    assert '### Incorrect' not in str(item.get('question_text','')),row['example_id']
PY

rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
python scripts/copy_model_parallel.py \
  --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
python -u "$evaluator" \
  --model "$MODEL_LOCAL" --adapter "$CHECKPOINT/adapter" \
  --recurrent-state "$CHECKPOINT/recurrent_state.pt" \
  --state-architecture layerwise --eval-jsonl "$RECURRENT_EVAL" \
  --output-dir "$OUTPUT" --state-token-alias '<|vision_pad|>' \
  --max-samples-per-group 25 --max-length 1536 --max-new-tokens 160 \
  --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
  --tasks diagnose_misconception --sources eedi_mining malrule \
  --save-unredacted-predictions --t3-natural-schema

python - "$AMLT_OUTPUT_DIR" "$ROUTE_MODE" \
  "$CHECKPOINT_ADAPTER_SHA256" "$CHECKPOINT_STATE_SHA256" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
expected={'adapter_model':sys.argv[3],'recurrent_state':sys.argv[4]}
metrics=json.load(open(root/'eval/formal_t3_75/metrics.json'))
pred=[json.loads(x) for x in open(root/'eval/formal_t3_75/predictions.jsonl') if x.strip()]
assert metrics['sampled_rows']==75 and metrics['state_shape']==[64,8,5120],metrics
assert metrics['artifact_sha256']==expected,metrics
assert metrics['raw_history_visible_to_decoder'] is False,metrics
assert metrics['persistent_history_kv_cache'] is False,metrics
eedi=[x for x in pred if x['source']=='eedi_mining']
assert len(eedi)==25,eedi
receipt={
 'stage':'state8_current_query_route_counterfactual_v1','status':'complete',
 'route_mode':mode,'checkpoint_step':100,'state_token_count':8,
 'raw_history_visible_to_decoder':False,
 'current_item_visible_to_decoder':True,
 'student_response_visible_to_decoder':False,
 'semantic_judgment_pending':True,'exact_match_authoritative':False,
 'artifact_sha256':expected,'protocol':{'t3_rows':75,'seed':42},
}
(root/'metrics/route_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
