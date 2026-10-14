#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_SOURCE_ANCHOR_SCREEN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${SOURCE_ANCHOR_MODE:?SOURCE_ANCHOR_MODE is required}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?EXPECTED_STUDENT_ADAPTER_SHA256 is required}"
: "${EXPECTED_STUDENT_STATE_SHA256:?EXPECTED_STUDENT_STATE_SHA256 is required}"

source_launcher=scripts/run_state8_stateonly_repair_ab.sh
rendered=/tmp/run_state8_source_anchor_screen.rendered.sh
source_hash=2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7
test "$(sha256sum "$source_launcher" | cut -d' ' -f1)" = "$source_hash"

case "$SOURCE_ANCHOR_MODE" in
  full_mse1)
    export TRAIN_SEED=3071
    export RUN_TAG=source_anchor_full_mse1_v1
    export STATE8_LAYERWISE_TRANSITION_ONLY=0
    export STATE8_UPDATER_ONLY=0
    export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=1.0
    ;;
  updater_mse1)
    export TRAIN_SEED=3072
    export RUN_TAG=source_anchor_updater_mse1_v1
    export STATE8_LAYERWISE_TRANSITION_ONLY=1
    export STATE8_UPDATER_ONLY=1
    export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=1.0
    ;;
  full_mse2)
    export TRAIN_SEED=3073
    export RUN_TAG=source_anchor_full_mse2_v1
    export STATE8_LAYERWISE_TRANSITION_ONLY=0
    export STATE8_UPDATER_ONLY=0
    export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=2.0
    ;;
  *) echo "Unknown SOURCE_ANCHOR_MODE=$SOURCE_ANCHOR_MODE" >&2; exit 2 ;;
esac

export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_source_anchored_repair.py
export REPAIR_ARM=taskanchor
export REQUESTED_WORLD_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export STATE8_REPAIR_SAVE_STEPS=20
export STATE8_REPAIR_RECOVERY_SAVE_STEPS=20
export STATE8_REPAIR_LABEL_WEIGHT=1.0
export STATE8_CAUSAL_LABEL_WEIGHT=1.0
export STATE8_RANDOM_NEUTRAL=1
export STATE8_GEOMETRY_WEIGHT=0.10
export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
export STATE8_DIFFERENT_MAX_COSINE=0.90
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_CAUSAL_REPEAT_FRACTION=0.50
export STATE8_CAUSAL_WEIGHT=2.0
export STATE8_CAUSAL_MARGIN=0.5
export STATE8_FORMAL_SCOPE=unified
export STATE8_EEDI_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_MALRULE_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_OTHER_DIAGNOSIS_ANCHOR_SCHEDULE=1.0,0.5,0.25
export STATE8_T6_ANCHOR_SCHEDULE=0.5,0.75,1.0
export STATE8_OTHER_ANCHOR_WEIGHT=0.1
export STATE8_TRAJECTORY_COSINE_WEIGHT=1.0
export STATE8_REPAIR_MANIFEST=manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json
export STATE8_REPAIR_MANIFEST_SHA256=a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557
export STUDENT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed3066_trajectory01_matched_cp3000_semantic_preserve_v1
export STUDENT_MODEL_DIR="$STUDENT_ROOT/model"
export STUDENT_STATE_PATH="$STUDENT_ROOT/model/recurrent_state.pt"
export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/publish_receipt.json"

python - "$source_launcher" "$rendered" "$source_hash" <<'PY'
import hashlib
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:3])
expected = sys.argv[3]
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
text = raw.decode()

def replace(old: str, new: str, count: int = 1) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

replace(
    "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = 7d5e4966a5317733fe5d5f32cc6730eb82f4ad78dedd8fa6a6967ee4aefb4e6d",
)
replace(
    "test \"$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)\" = 262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
    "test \"$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | cut -d' ' -f1)\" = 208edcc8c623101667844910ad3e85c79d82491a08e370acc468ad617ab62522",
)
replace(
    "test \"$(sha256sum \"$TRAIN_ENTRYPOINT\" | cut -d' ' -f1)\" = 14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a ;;",
    "test \"$(sha256sum \"$TRAIN_ENTRYPOINT\" | cut -d' ' -f1)\" = 26e33e8aa9a2cc2e77d96299562fc13ee247ca007ff0e9b4657b0f0a2e905b31 ;;",
)
replace(
    "export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1",
    "export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_diagnosis_first_full_history_teacher_v1",
)
replace(
    '  "$TEACHER_ROOT/model/adapter_model.safetensors" \\\n'
    '  "$TEACHER_ROOT/metrics/teacher_gate.json" "$MANIFEST" "$PAIRS"; do',
    '  "$TEACHER_ROOT/model/adapter_model.safetensors" \\\n'
    '  "$TEACHER_ROOT/metrics/teacher_gate.json" \\\n'
    '  "$TEACHER_ROOT/metrics/teacher_provenance_gate.json" "$MANIFEST" "$PAIRS"; do',
)
replace(
    "assert config['output_schema']=='label_diagnosis_evidence_v1',config",
    "assert config['output_schema'] in {'label_diagnosis_evidence_v1','diagnosis_evidence_label_v1'},config",
)
replace(
    "assert tg['stage']=='state8_independent_full_history_teacher_gate_v1' and tg['pass'] is True,tg\n"
    "assert tg['support']==1494 and tg['exact_match_is_authoritative'] is False,tg",
    "assert tg['stage']=='state8_diagnosis_first_full_history_teacher_screen_v1' and tg['status']=='complete' and tg['pass'] is True,tg\n"
    "assert tg['support']==75 and sum(x['support'] for x in tg['by_group'].values())==75,tg\n"
    "assert all(x['pass'] is True for x in tg['by_group'].values()),tg\n"
    "assert tg['judge_backend']=='trapi' and tg['semantic_judge_authoritative'] is True,tg\n"
    "assert tg['exact_match_authoritative'] is False,tg",
)

diagnosis_case = '''  scripts/train_state8_diagnosis_first_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256:?}" 
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0 ;;'''
source_case = diagnosis_case + '''
  scripts/train_state8_diagnosis_first_source_anchored_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_SOURCE_ANCHOR_ENTRYPOINT_SHA256:?}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT:-}" != 0 ;;'''
replace(diagnosis_case, source_case)

needle = '|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then'
replacement = (
    '|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\\n'
    '      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then'
)
replace(needle, replacement, 4)

needle = 'if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then'
replacement = (
    'if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\\n'
    '      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then'
)
replace(needle, replacement, 3)

replace(
    "                        'train_state8_diagnosis_first_causal_repair.py')):",
    "                        'train_state8_diagnosis_first_causal_repair.py',\n"
    "                        'train_state8_diagnosis_first_source_anchored_repair.py')):",
)
replace(
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_causal_repair.py
fi''',
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_causal_repair.py
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_source_anchored_repair.py
fi''',
)
replace(
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
fi''',
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
fi
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_source_anchored_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_source_anchored_repair.json"
fi''',
)
replace(
    '--consistency-weight "$STATE8_REPAIR_TRAJECTORY_WEIGHT" --direct-replay-weight 0.0 \\\n',
    '--consistency-weight "$STATE8_REPAIR_TRAJECTORY_WEIGHT" --direct-replay-weight 0.0 \\\n'
    '  --trajectory-cosine-weight "$STATE8_TRAJECTORY_COSINE_WEIGHT" \\\n'
    '  --trajectory-relative-mse-weight "$STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT" \\\n',
)
output.write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
PY

export STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256=7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216
export STATE8_SOURCE_ANCHOR_ENTRYPOINT_SHA256=f9b96fe182641b9563911463ef6eacd8ffef7131ad613ec6c6865043c2b964ff
export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9
export STATE8_DIAGNOSIS_FIRST_C4_SHA256=660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca

bash -n "$rendered"
bash "$rendered"
