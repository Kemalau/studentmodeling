#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CURRICULUM_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export STATE8_REPLAY_VARIANT=t3_heavy
export STATE8_KD_VARIANT=highkd
export STATE8_CURRICULUM_NAMESPACE=taxonomy_semantic_v1
export STATE8_SINGLE_DEVICE_FALLBACK=1
export STATE8_SAVE_STEPS=200
export STATE8_LEARNING_RATE=5e-7
export STATE8_RUN_SUFFIX=diagnosis_first_matched_teacher_seed67
export TRAIN_SEED=67
export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py
export FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py
export STATE8_CURRICULUM_LABEL_WEIGHT=1.0
export STATE8_TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1

export STATE8_STUDENT_ROOT=/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_full_open_t3/state_path/lr5e-7_start17000_stop4500_08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a/checkpoint-round-4500
export STATE8_STUDENT_MODEL_DIR="$STATE8_STUDENT_ROOT/adapter"
export STATE8_STUDENT_STATE_PATH="$STATE8_STUDENT_ROOT/recurrent_state.pt"
export STATE8_STUDENT_RECEIPT="$STATE8_STUDENT_ROOT/checkpoint_completion.json"
export STATE8_STUDENT_INIT_CAPACITY=1
export STATE8_STUDENT_INIT_TAG=strong1_round4500
export STATE8_STUDENT_INIT_ADAPTER_SHA256=827532591bbcc5587bc78a0108463dac43ceab8c702ee6f60e120c68a2afe482
export STATE8_STUDENT_INIT_STATE_SHA256=701ed585383d919bdfe9956ddbbfe870b2dcf323a3ae4adde257cf2365e419da

test -f "$STATE8_TEACHER_ROOT/model/adapter_model.safetensors"
for name in publish_receipt.json teacher_gate.json teacher_provenance_gate.json; do
  test -f "$STATE8_TEACHER_ROOT/metrics/$name"
done
readarray -t matched_hashes < <("$VENV_ROOT/bin/python" - "$STATE8_TEACHER_ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
p=json.load(open(root/'metrics/publish_receipt.json'))
g=json.load(open(root/'metrics/teacher_gate.json'))
v=json.load(open(root/'metrics/teacher_provenance_gate.json'))
assert p['status']=='complete' and p['semantic_teacher_gate_pass'] is True,p
assert p['diagnosis_field_order']==['diagnosis','evidence','label'],p
assert p['adapter_sha256']==digest(root/'model/adapter_model.safetensors'),p
assert g['stage']=='state8_diagnosis_first_full_history_teacher_screen_v1' and g['pass'] is True,g
assert g['support']==75 and g['semantic_judge_authoritative'] is True,g
assert g['exact_match_authoritative'] is False,g
assert v['stage']=='state8_diagnosis_first_teacher_provenance_gate_v1' and v['status']=='complete',v
assert v['teacher_adapter_sha256']==p['adapter_sha256'],(v,p)
assert v['output_schema']=='diagnosis_evidence_label_v1',v
print(p['eval_sha256']); print(digest(root/'metrics/teacher_gate.json'))
print(digest(root/'metrics/teacher_provenance_gate.json'))
PY
)
test "${#matched_hashes[@]}" = 3
export STATE8_MATCHED_TEACHER_EVAL_SHA256="${matched_hashes[0]}"
export STATE8_MATCHED_TEACHER_GATE_SHA256="${matched_hashes[1]}"
export STATE8_MATCHED_TEACHER_PROVENANCE_SHA256="${matched_hashes[2]}"

python - scripts/run_state8_qualified_teacher_curriculum.sh \
  /tmp/run_state8_diagnosis_first_matched_curriculum.rendered.sh <<'PY'
import hashlib,pathlib,sys
source,output=map(pathlib.Path,sys.argv[1:])
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b545496b52c8aff82196c8ac917178b110e304b1ae4880ed24ca039cff626638'
text=raw.decode()
old='export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1'
new='export TEACHER_ROOT="${STATE8_TEACHER_ROOT:?}"'
assert text.count(old)==1; text=text.replace(old,new)
old='''  test "$TRAIN_ENTRYPOINT" = scripts/train_state8_taxonomy_teacher_curriculum.py
  test "$FORMAL_EVALUATOR" = scripts/evaluate_recurrent_formal_protocols_taxonomy.py
  test "${STATE8_CURRICULUM_LABEL_WEIGHT:-}" = 4.0'''
new='''  case "$TRAIN_ENTRYPOINT:$FORMAL_EVALUATOR:${STATE8_CURRICULUM_LABEL_WEIGHT:-}" in
    scripts/train_state8_taxonomy_teacher_curriculum.py:scripts/evaluate_recurrent_formal_protocols_taxonomy.py:4.0) ;;
    scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)
      test "$(sha256sum scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py | cut -d' ' -f1)" = 0f4e305fa5eb48ec2beaa328a3b9c125b649f28327a405a211a1803570498a1a
      test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
      ;;
    *) exit 2 ;;
  esac'''
assert text.count(old)==1; text=text.replace(old,new)
start=text.index("tp=json.load(open(teacher/'metrics/publish_receipt.json'))")
stop=text.index("if 'state_shape' in si:",start)
old=text[start:stop]
new='''tp=json.load(open(teacher/'metrics/publish_receipt.json')); tg=json.load(open(teacher/'metrics/teacher_gate.json'))
pg=json.load(open(teacher/'metrics/teacher_provenance_gate.json'))
si=json.load(open(student_receipt))
teacher_hash=digest(teacher/'model/adapter_model.safetensors')
student_hash=digest(student_model/'adapter_model.safetensors')
state_hash=digest(student_state)
assert tp['status']=='complete' and tp['semantic_teacher_gate_pass'] is True,tp
assert tp['adapter_sha256']==teacher_hash,tp
assert tp['diagnosis_field_order']==['diagnosis','evidence','label'],tp
assert tg['stage']=='state8_diagnosis_first_full_history_teacher_screen_v1' and tg['pass'] is True,tg
assert tg['support']==75 and tg['semantic_judge_authoritative'] is True,tg
assert tg['exact_match_authoritative'] is False,tg
assert pg['stage']=='state8_diagnosis_first_teacher_provenance_gate_v1' and pg['status']=='complete',pg
assert pg['teacher_gate_pass'] is True and pg['teacher_adapter_sha256']==teacher_hash,pg
assert pg['output_schema']=='diagnosis_evidence_label_v1',pg
assert pg['diagnosis_field_order']==['diagnosis','evidence','label'],pg
assert pg['student_history_visible_at_final_inference'] is False,pg
'''
text=text[:start]+new+text[stop:]
text=text.replace("assert cfg['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',cfg",
 "assert cfg['teacher_prompt_provenance_stage']=='state8_diagnosis_first_teacher_provenance_gate_v1',cfg")
text=text.replace("assert cfg['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',cfg",
 "assert cfg['teacher_evaluation_input_sha256']==__import__('os').environ['STATE8_MATCHED_TEACHER_EVAL_SHA256'],cfg")
text=text.replace("assert cfg['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',cfg",
 "assert cfg['teacher_clean_prompt_sha256']==__import__('os').environ['STATE8_MATCHED_TEACHER_EVAL_SHA256'],cfg")
text=text.replace("assert cfg['final_inference_route']=='strict_state_only',cfg",
 "assert cfg['final_inference_route']=='strict_state_only' and cfg.get('raw_history_visible_to_decoder',False) is False,cfg")
text=text.replace(" assert float(cfg['label_auxiliary_weight'])==4.0,cfg",
 " assert float(cfg['label_auxiliary_weight'])==float(__import__('os').environ['STATE8_CURRICULUM_LABEL_WEIGHT']),cfg")
text=text.replace("assert q['teacher_prompt_provenance_stage']=='state8_clean_prompt_teacher_provenance_gate_v2',q",
 "assert q['teacher_prompt_provenance_stage']=='state8_diagnosis_first_teacher_provenance_gate_v1',q")
text=text.replace("assert q['teacher_evaluation_input_sha256']=='78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',q",
 "assert q['teacher_evaluation_input_sha256']==__import__('os').environ['STATE8_MATCHED_TEACHER_EVAL_SHA256'],q")
text=text.replace("assert q['teacher_clean_prompt_sha256']=='c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',q",
 "assert q['teacher_clean_prompt_sha256']==__import__('os').environ['STATE8_MATCHED_TEACHER_EVAL_SHA256'],q")
text=text.replace(" assert float(q['label_auxiliary_weight'])==4.0,q",
 " assert float(q['label_auxiliary_weight'])==float(__import__('os').environ['STATE8_CURRICULUM_LABEL_WEIGHT']),q")
text=text.replace(" 'teacher_prompt_provenance_stage':'state8_clean_prompt_teacher_provenance_gate_v2',",
 " 'teacher_prompt_provenance_stage':'state8_diagnosis_first_teacher_provenance_gate_v1',")
text=text.replace(" 'teacher_evaluation_input_sha256':'78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5',",
 " 'teacher_evaluation_input_sha256':__import__('os').environ['STATE8_MATCHED_TEACHER_EVAL_SHA256'],")
text=text.replace(" 'teacher_clean_prompt_sha256':'c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314',",
 " 'teacher_clean_prompt_sha256':__import__('os').environ['STATE8_MATCHED_TEACHER_EVAL_SHA256'],")
old='''if [[ "$STATE8_CURRICULUM_NAMESPACE" == taxonomy_semantic_v1 ]]; then
  test -f "$RUN_DIR/taxonomy_teacher_curriculum.json"
fi'''
new=old+'''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_teacher_curriculum.json"
  test -f "$RUN_DIR/matched_teacher_curriculum.json"
fi'''
assert text.count(old)>=1; text=text.replace(old,new,1)
for destination in ('"$AMLT_OUTPUT_DIR/metrics/','"$STABLE_ROOT/metrics/'):
 old=f'''if [[ -f "$RUN_DIR/taxonomy_teacher_curriculum.json" ]]; then
  cp "$RUN_DIR/taxonomy_teacher_curriculum.json" {destination}taxonomy_teacher_curriculum.json"
fi'''
 new=old+f'''
if [[ -f "$RUN_DIR/diagnosis_first_teacher_curriculum.json" ]]; then
  cp "$RUN_DIR/diagnosis_first_teacher_curriculum.json" {destination}diagnosis_first_teacher_curriculum.json"
  cp "$RUN_DIR/matched_teacher_curriculum.json" {destination}matched_teacher_curriculum.json"
fi'''
 assert text.count(old)==1; text=text.replace(old,new)
output.write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
PY
bash -n /tmp/run_state8_diagnosis_first_matched_curriculum.rendered.sh
bash /tmp/run_state8_diagnosis_first_matched_curriculum.rendered.sh

# Generate the frozen natural-language C4 panel while the trained checkpoint
# and 32B backbone are already resident on the Feeds node.  Semantic scoring
# remains outside the target model and is performed by the independent TRAPI
# watcher after artifacts are downloaded.
MATCHED_STABLE_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4
MATCHED_C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean/test_same_template.jsonl
MATCHED_C4_PAIRS=manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json
MATCHED_C4_OUT="$AMLT_OUTPUT_DIR/eval/diagnosis_c4_20"
MATCHED_ANSWER_C4_OUT="$AMLT_OUTPUT_DIR/eval/answer_c4_150"
MATCHED_ANSWER_C4_PROGRESS=/mnt/amulet/studentmodeling/eval_progress/state8_diagnosis_first_matched_seed67/answer_c4_seed42
for path in \
  "$MATCHED_STABLE_ROOT/model/adapter_model.safetensors" \
  "$MATCHED_STABLE_ROOT/model/recurrent_state.pt" \
  "$MATCHED_STABLE_ROOT/metrics/immutable_receipt.json" \
  "$MATCHED_C4_DATA" "$MATCHED_C4_PAIRS"; do
  test -f "$path"
done
test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py | cut -d' ' -f1)" = 660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
test "$(sha256sum scripts/evaluate_recurrent_causal_gate.py | cut -d' ' -f1)" = 69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52
test "$(sha256sum "$MATCHED_C4_PAIRS" | cut -d' ' -f1)" = edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8
readarray -t matched_artifact_hashes < <("$VENV_ROOT/bin/python" - "$MATCHED_STABLE_ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
r=json.load(open(root/'metrics/immutable_receipt.json'))
observed={'adapter_model':digest(root/'model/adapter_model.safetensors'),
          'recurrent_state':digest(root/'model/recurrent_state.pt')}
assert r['status']=='complete' and r['state_shape']==[64,8,5120],r
assert r['raw_history_visible_to_decoder'] is False,r
assert r['persistent_history_kv_cache'] is False,r
assert r['semantic_judge_authoritative'] is True,r
assert r['exact_label_metric_authoritative'] is False,r
assert r['sha256']==observed,(r,observed)
print(observed['adapter_model']); print(observed['recurrent_state'])
PY
)
test "${#matched_artifact_hashes[@]}" = 2
rm -rf "$MATCHED_C4_OUT"; mkdir -p "$MATCHED_C4_OUT"
"$VENV_ROOT/bin/python" -u scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py \
  --model /tmp/Qwen3-32B --adapter "$MATCHED_STABLE_ROOT/model" \
  --recurrent-state "$MATCHED_STABLE_ROOT/model/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$MATCHED_C4_DATA" \
  --pair-manifest "$MATCHED_C4_PAIRS" --output-dir "$MATCHED_C4_OUT" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 \
  --max-new-tokens 192 --num-pairs 20 --seed 67 \
  --expected-adapter-sha256 "${matched_artifact_hashes[0]}" \
  --expected-recurrent-state-sha256 "${matched_artifact_hashes[1]}"
"$VENV_ROOT/bin/python" - "$MATCHED_C4_OUT/metrics.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['num_pairs']==20 and x['state_shape']==[64,8,5120],x
assert x['conditions']==['native','noop','samemal','random','swap'],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['semantic_judgment_pending'] is True,x
PY
"$VENV_ROOT/bin/python" - "$MATCHED_C4_OUT/predictions.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows)==100, len(rows)
assert all('For this Malrule diagnosis' in row['decode_prompt'] for row in rows)
assert all('For this Eedi diagnosis' not in row['decode_prompt'] for row in rows)
PY

# Frozen 150-pair answer-level C4 is a separate promotion requirement.  It is
# resumable on remote storage and bound to the exact adapter/state hashes.
mkdir -p "$MATCHED_ANSWER_C4_OUT" "$MATCHED_ANSWER_C4_PROGRESS"
"$VENV_ROOT/bin/python" -u scripts/evaluate_recurrent_causal_gate.py \
  --model /tmp/Qwen3-32B --adapter "$MATCHED_STABLE_ROOT/model" \
  --recurrent-state "$MATCHED_STABLE_ROOT/model/recurrent_state.pt" \
  --state-architecture layerwise --jsonl "$MATCHED_C4_DATA" \
  --pair-manifest "$MATCHED_C4_PAIRS" \
  --output-json "$MATCHED_ANSWER_C4_OUT/recurrent_causal_gate.json" \
  --state-token-alias '<|vision_pad|>' --max-length 1536 --num-pairs 150 \
  --seed 42 --progress-dir "$MATCHED_ANSWER_C4_PROGRESS" \
  --progress-key "adapter=${matched_artifact_hashes[0]}:state=${matched_artifact_hashes[1]}:pairs=150:seed=42:v1"
"$VENV_ROOT/bin/python" - "$MATCHED_ANSWER_C4_OUT/recurrent_causal_gate.json" \
  "${matched_artifact_hashes[0]}" "${matched_artifact_hashes[1]}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['pair_seed']==42 and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False,x
assert x['summary']['all_pairs_n']==150 and x['summary']['clean_n']>0,x
assert x['artifact_sha256']=={'adapter_model':sys.argv[2],'recurrent_state':sys.argv[3]},x
PY
