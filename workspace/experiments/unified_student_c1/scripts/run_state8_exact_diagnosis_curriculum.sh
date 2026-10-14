#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EXACT_CURRICULUM_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export STATE8_REPLAY_VARIANT=t3_heavy
export STATE8_KD_VARIANT=highkd
export STATE8_CURRICULUM_NAMESPACE=taxonomy_semantic_v1
export STATE8_SINGLE_DEVICE_FALLBACK=1
export STATE8_SAVE_STEPS=200
export STATE8_LEARNING_RATE=5e-7
export STATE8_RUN_SUFFIX=exact_diagnosis_first_seed64
export TRAIN_SEED=64
export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_teacher_curriculum.py
export FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py
export STATE8_CURRICULUM_LABEL_WEIGHT=1.0

# Feeds can preempt the long formal-generation tail after the complete
# curriculum has already been atomically published.  Bind every scientific
# artifact before reusing it; a partial or stale publication falls through to
# the ordinary exact-resume path below.
curriculum_root=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4
reuse_receipt="$AMLT_OUTPUT_DIR/metrics/immutable_curriculum_reuse.json"
mkdir -p "$(dirname "$reuse_receipt")"
if "$VENV_ROOT/bin/python" scripts/validate_state8_exact_immutable_curriculum.py \
    --root "$curriculum_root" --output "$reuse_receipt"; then
  printf 'STATE8_CURRICULUM_STAGE=immutable_reuse time=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi
rm -f "$reuse_receipt"

export STATE8_STUDENT_ROOT=/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_full_open_t3/state_path/lr5e-7_start17000_stop4500_08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a/checkpoint-round-4500
export STATE8_STUDENT_MODEL_DIR="$STATE8_STUDENT_ROOT/adapter"
export STATE8_STUDENT_STATE_PATH="$STATE8_STUDENT_ROOT/recurrent_state.pt"
export STATE8_STUDENT_RECEIPT="$STATE8_STUDENT_ROOT/checkpoint_completion.json"
export STATE8_STUDENT_INIT_CAPACITY=1
export STATE8_STUDENT_INIT_TAG=strong1_round4500
export STATE8_STUDENT_INIT_ADAPTER_SHA256=827532591bbcc5587bc78a0108463dac43ceab8c702ee6f60e120c68a2afe482
export STATE8_STUDENT_INIT_STATE_SHA256=701ed585383d919bdfe9956ddbbfe870b2dcf323a3ae4adde257cf2365e419da

curriculum_sha="${STATE8_DIAGNOSIS_FIRST_CURRICULUM_SHA256:?}"
test "$(sha256sum scripts/train_state8_diagnosis_first_teacher_curriculum.py | cut -d' ' -f1)" = "$curriculum_sha"

# The production launcher deliberately whitelists only its established
# taxonomy entrypoint.  Render an immutable diagnosis-first variant without
# mutating that already-running path or weakening its fail-closed checks.
python - scripts/run_state8_qualified_teacher_curriculum.sh \
  /tmp/run_state8_exact_diagnosis_curriculum.rendered.sh \
  "$curriculum_sha" <<'PY'
import hashlib, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:3])
curriculum_sha = sys.argv[3]
expected = "b545496b52c8aff82196c8ac917178b110e304b1ae4880ed24ca039cff626638"
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
text = raw.decode()
old = """  test \"$TRAIN_ENTRYPOINT\" = scripts/train_state8_taxonomy_teacher_curriculum.py
  test \"$FORMAL_EVALUATOR\" = scripts/evaluate_recurrent_formal_protocols_taxonomy.py
  test \"${STATE8_CURRICULUM_LABEL_WEIGHT:-}\" = 4.0"""
new = """  case \"$TRAIN_ENTRYPOINT:$FORMAL_EVALUATOR:${STATE8_CURRICULUM_LABEL_WEIGHT:-}\" in
    scripts/train_state8_taxonomy_teacher_curriculum.py:scripts/evaluate_recurrent_formal_protocols_taxonomy.py:4.0) ;;
    scripts/train_state8_diagnosis_first_teacher_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)
      test \"$(sha256sum scripts/train_state8_diagnosis_first_teacher_curriculum.py | cut -d' ' -f1)\" = \"${STATE8_DIAGNOSIS_FIRST_CURRICULUM_SHA256:?}\"
      test \"$(sha256sum scripts/train_state8_diagnosis_first_causal_repair.py | cut -d' ' -f1)\" = 7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216
      test \"$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py | cut -d' ' -f1)\" = a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
      ;;
    *) exit 2 ;;
  esac"""
assert text.count(old) == 1
text = text.replace(old, new)
old = "assert float(cfg['label_auxiliary_weight'])==4.0,cfg"
new = "assert float(cfg['label_auxiliary_weight'])==float(__import__('os').environ['STATE8_CURRICULUM_LABEL_WEIGHT']),cfg"
assert text.count(old) == 1
text = text.replace(old, new)
old = """if [[ \"$STATE8_CURRICULUM_NAMESPACE\" == taxonomy_semantic_v1 ]]; then
  test -f \"$RUN_DIR/taxonomy_teacher_curriculum.json\"
fi"""
new = old + """
if [[ \"$TRAIN_ENTRYPOINT\" == scripts/train_state8_diagnosis_first_teacher_curriculum.py ]]; then
  test -f \"$RUN_DIR/diagnosis_first_teacher_curriculum.json\"
fi"""
assert text.count(old) == 1
text = text.replace(old, new)
old = "assert float(q['label_auxiliary_weight'])==4.0,q"
new = "assert float(q['label_auxiliary_weight'])==float(__import__('os').environ['STATE8_CURRICULUM_LABEL_WEIGHT']),q"
assert text.count(old) == 1
text = text.replace(old, new)
old = """ assert taxonomy_receipt['exact_label_metric_authoritative'] is False,taxonomy_receipt
receipt={"""
new = """ assert taxonomy_receipt['exact_label_metric_authoritative'] is False,taxonomy_receipt
diagnosis_first=(root/'diagnosis_first_teacher_curriculum.json')
if diagnosis_first.is_file():
 d=json.load(open(diagnosis_first))
 assert d['field_order']==['diagnosis','evidence','label'],d
 assert d['canonical_label_role']=='auxiliary_ordinary_weight',d
 assert d['history_dropout_schedule']==[0.0,0.25,0.5,0.75,1.0],d
 assert d['semantic_judge_authoritative'] is True,d
 assert d['exact_label_metric_authoritative'] is False,d
receipt={"""
assert text.count(old) == 1
text = text.replace(old, new)
old = """ 'label_auxiliary_weight':float(cfg.get('label_auxiliary_weight',1.0)),
 'taxonomy_label_contract':"""
new = """ 'label_auxiliary_weight':float(cfg.get('label_auxiliary_weight',1.0)),
 'diagnosis_field_order':cfg.get('diagnosis_field_order'),
 'canonical_label_role':cfg.get('canonical_label_role','auxiliary_weighted'),
 'taxonomy_label_contract':"""
assert text.count(old) == 1
text = text.replace(old, new)
old = """if [[ -f \"$RUN_DIR/taxonomy_teacher_curriculum.json\" ]]; then
  cp \"$RUN_DIR/taxonomy_teacher_curriculum.json\" \"$AMLT_OUTPUT_DIR/metrics/taxonomy_teacher_curriculum.json\"
fi"""
new = old + """
if [[ -f \"$RUN_DIR/diagnosis_first_teacher_curriculum.json\" ]]; then
  cp \"$RUN_DIR/diagnosis_first_teacher_curriculum.json\" \"$AMLT_OUTPUT_DIR/metrics/diagnosis_first_teacher_curriculum.json\"
fi"""
assert text.count(old) == 1
text = text.replace(old, new)
old = """if [[ -f \"$RUN_DIR/taxonomy_teacher_curriculum.json\" ]]; then
  cp \"$RUN_DIR/taxonomy_teacher_curriculum.json\" \"$STABLE_ROOT/metrics/taxonomy_teacher_curriculum.json\"
fi"""
new = old + """
if [[ -f \"$RUN_DIR/diagnosis_first_teacher_curriculum.json\" ]]; then
  cp \"$RUN_DIR/diagnosis_first_teacher_curriculum.json\" \"$STABLE_ROOT/metrics/diagnosis_first_teacher_curriculum.json\"
fi"""
assert text.count(old) == 1
text = text.replace(old, new)
old = """ 'label_auxiliary_weight':q.get('label_auxiliary_weight',1.0),
 'taxonomy_label_contract':"""
new = """ 'label_auxiliary_weight':q.get('label_auxiliary_weight',1.0),
 'diagnosis_field_order':q.get('diagnosis_field_order'),
 'canonical_label_role':q.get('canonical_label_role','auxiliary_weighted'),
 'taxonomy_label_contract':"""
assert text.count(old) == 1
text = text.replace(old, new)
output.write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
PY
bash -n /tmp/run_state8_exact_diagnosis_curriculum.rendered.sh
bash /tmp/run_state8_exact_diagnosis_curriculum.rendered.sh
