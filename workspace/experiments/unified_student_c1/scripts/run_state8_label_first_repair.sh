#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_LABEL_FIRST_REPAIR_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${SOURCE_ROOT:?}" "${EXPECTED_STUDENT_ADAPTER_SHA256:?}"
: "${EXPECTED_STUDENT_STATE_SHA256:?}" "${REPAIR_REMOTE_ROOT:?}"
: "${STATE8_REPAIR_MANIFEST:?}" "${STATE8_REPAIR_MANIFEST_SHA256:?}"

test "$SOURCE_ROOT" = /mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401
test "$REPAIR_REMOTE_ROOT" = /mnt/amulet/studentmodeling/diagnostics/state8_label_first_repair_seed8501_v1
test "$STATE8_REPAIR_MANIFEST" = manifests/state8_label_first_repair_seed8501_v1.json
test "$STATE8_REPAIR_MANIFEST_SHA256" = 4a5e31add8732a16ccc109ad9ff3d1956e9c0dcfa4608a0dea3800d7ee295f9c
test "$(sha256sum "$STATE8_REPAIR_MANIFEST" | cut -d' ' -f1)" = "$STATE8_REPAIR_MANIFEST_SHA256"
target_remote_root=$REPAIR_REMOTE_ROOT

base=scripts/run_state8_interleaved_retention_branch.sh
base_sha=ad07c6e2ea60175f6b86664cf754853eddc28872374faf44ab27c32344f2a6ea
test "$(sha256sum "$base" | cut -d' ' -f1)" = "$base_sha"

export REPAIR_BRANCH=early_label_high
export REPAIR_SEED=8401
export SOURCE_CHECKPOINT_ARM=curriculum_current_query
export SOURCE_CHECKPOINT_STEP=3000
export SOURCE_ANCHOR_MODE=full_mse1
export SOURCE_CHECKPOINT_ROOT=/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-3000
export REPAIR_REMOTE_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401
export STATE8_T6_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_EEDI_CAUSAL_TWINS=200
export STATE8_CAUSAL_REPEAT_FRACTION=0.75
export STATE8_CAUSAL_WEIGHT=0.5
export STATE8_CAUSAL_MARGIN=0.5
export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
export STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT=0.15
export STATE8_CAUSAL_TEACHER_COSINE_WEIGHT=1.0
export STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT=1.0
export STATE8_EARLY_LABEL_BINDING_WEIGHT=0.5
export STATE8_INTERLEAVED_RENDER_ONLY=1

source_manifest=$STATE8_REPAIR_MANIFEST
source_manifest_sha=$STATE8_REPAIR_MANIFEST_SHA256
# Render the proven launcher with its original immutable inputs, then bind the
# final script to the new source checkpoint and 500-row suffix.
export STATE8_REPAIR_MANIFEST=manifests/state8_interleaved_retention_seed8401_v1.json
export STATE8_REPAIR_MANIFEST_SHA256=bb502f9bbbcdd8beb92d474da16ed8f803eb388840107ba1df1841fd4fcff85c
bash "$base"

rendered=/tmp/run_state8_label_first_repair.final.sh
python - /tmp/run_state8_interleaved_retention.final.sh "$rendered" \
  "$SOURCE_ROOT" "$target_remote_root" "$source_manifest" \
  "$source_manifest_sha" <<'PY'
from pathlib import Path
import hashlib, sys

source, output = map(Path, sys.argv[1:3])
source_root, stable_root, manifest, manifest_sha = sys.argv[3:]
value = source.read_text(encoding="utf-8")

def replace(old: str, new: str, count: int = 1) -> None:
    global value
    observed = value.count(old)
    if observed != count:
        raise ValueError((old, observed, count))
    value = value.replace(old, new)

replace('export RUN_TAG="${RUN_TAG:-v1}"', 'export RUN_TAG=label_first_state_only_v1')
replace('export TRAIN_SEED="${TRAIN_SEED:-44}"', 'export TRAIN_SEED=8501')
replace(
    'export MANIFEST="${STATE8_REPAIR_MANIFEST:-manifests/state8_stateonly_repair_seed44_v1.json}"',
    f'export MANIFEST={manifest}',
)
replace(
    'export MANIFEST_SHA256="${STATE8_REPAIR_MANIFEST_SHA256:-5aee46644236170613142c688ec0c05d14e5b965656167c3e1d059ef493d4cf3}"',
    f'export MANIFEST_SHA256={manifest_sha}',
)
replace(
    'export STUDENT_ROOT="${STUDENT_ROOT:-/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_v1_immutable}"',
    f'export STUDENT_ROOT={source_root}',
)
replace(
    'export STUDENT_MODEL_DIR="${STUDENT_MODEL_DIR:-$STUDENT_ROOT/model}"',
    'export STUDENT_MODEL_DIR="$STUDENT_ROOT/model"',
)
replace(
    'export STUDENT_STATE_PATH="${STUDENT_STATE_PATH:-$STUDENT_MODEL_DIR/recurrent_state.pt}"',
    'export STUDENT_STATE_PATH="$STUDENT_ROOT/model/recurrent_state.pt"',
)
replace(
    '''if [[ -z "${STUDENT_RECEIPT:-}" ]]; then
  if [[ -f "$STUDENT_ROOT/metrics/immutable_receipt.json" ]]; then
    export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/immutable_receipt.json"
  elif [[ -f "$STUDENT_ROOT/metrics/publish_receipt.json" ]]; then
    export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/publish_receipt.json"
  else
    export STUDENT_RECEIPT="$STUDENT_ROOT/completion.json"
  fi
fi''',
    'export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/publish_receipt.json"',
)
replace(
    'export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/state8_stateonly_repair_seed${TRAIN_SEED}_${REPAIR_ARM}_${RUN_TAG}"',
    'export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/state8_label_first_repair_seed8501_v1"',
)
replace(
    'export STABLE_ROOT="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed${TRAIN_SEED}_${REPAIR_ARM}_${RUN_TAG}"',
    f'export STABLE_ROOT={stable_root}',
)
replace('assert s[\'rows\']==1000 and s[\'formal_example_id_overlap\']==0,s',
        'assert s[\'rows\']==500 and s[\'formal_example_id_overlap\']==0,s')
replace("assert x['status']=='passed' and x['rows']==1000,x",
        "assert x['status']=='passed' and x['rows']==500,x")
replace('--output-jsonl "$CAUSAL_SUBSET/train.jsonl" --twins 125 --seed 44',
        '--output-jsonl "$CAUSAL_SUBSET/train.jsonl" --twins 50 --seed 44')
replace('--twins 200 --seed 20260811 --offset 0',
        '--twins 150 --seed 20260811 --offset 0')
replace('--twins 25 --seed 20260811 --offset 200',
        '--twins 25 --seed 20260811 --offset 150')
replace("assert train['output_pairs']==400 and train['offset']==0,train",
        "assert train['output_pairs']==300 and train['offset']==0,train")
replace("assert valid['output_pairs']==50 and valid['offset']==200,valid",
        "assert valid['output_pairs']==50 and valid['offset']==150,valid")
replace("assert merged['rows']==900,merged",
        "assert merged['rows']==500,merged")
replace("assert merged['source_counts']=={'eedi_mining':400,'malrule':500},merged",
        "assert merged['source_counts']=={'eedi_mining':300,'malrule':200},merged")
replace("assert c['train']['input_pairs']==250 and c['train']['output_pairs']==500,c",
        "assert c['train']['input_pairs']==100 and c['train']['output_pairs']==200,c")
replace("assert c['status']=='complete' and c['micro_steps']==1000,c",
        "assert c['status']=='complete' and c['micro_steps']==500,c")
replace("assert c['optimizer_steps']==250,c",
        "assert c['optimizer_steps']==125,c")
replace("expected_pairs=250 if os.environ['STATE8_CAUSAL_DIAGNOSIS_ONLY']=='1' else 900",
        "expected_pairs=100 if os.environ['STATE8_CAUSAL_DIAGNOSIS_ONLY']=='1' else 500")
replace('expected_exposures=max(expected_pairs,round(1000*float(os.environ[\'STATE8_CAUSAL_REPEAT_FRACTION\'])))',
        'expected_exposures=max(expected_pairs,round(500*float(os.environ[\'STATE8_CAUSAL_REPEAT_FRACTION\'])))')
replace("assert collections.Counter(float(x['history_dropout']) for x in h)=={1.0:1000},h[:2]",
        "assert collections.Counter(float(x['history_dropout']) for x in h)=={1.0:500},h[:2]")
replace(
    "  --model \"$LOCAL_MODEL\" --init-adapter \"$STUDENT_MODEL_DIR\" \"${INIT_ARGS[@]}\" \\",
    "  --model \"$LOCAL_MODEL\" --init-adapter \"$STUDENT_MODEL_DIR\" \"${INIT_ARGS[@]}\" \\",
)

insertion = '''export STATE8_DIAGNOSIS_FIELD_ORDER=label_first
export STATE8_EARLY_LABEL_QUERY_MODE=state_only
export STATE8_EARLY_LABEL_BINDING_WEIGHT=1.0
export STATE8_EARLY_LABEL_RANDOM_WEIGHT=0.5
export STATE8_EARLY_LABEL_SAME_RULE_WEIGHT=0.5
export REPAIR_ARM=taskanchor
export REQUESTED_WORLD_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4
export STATE8_REPAIR_LABEL_WEIGHT=1.0
export STATE8_CAUSAL_LABEL_WEIGHT=1.0
export STATE8_RANDOM_NEUTRAL=1
export STATE8_GEOMETRY_WEIGHT=0.10
export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
export STATE8_DIFFERENT_MAX_COSINE=0.90
export STATE8_CAUSAL_DIAGNOSIS_ONLY=0
export STATE8_CAUSAL_WEIGHT=1.0
export STATE8_CAUSAL_REPEAT_FRACTION=1.0
export STATE8_CAUSAL_MARGIN=0.5
export STATE8_FORMAL_SCOPE=unified
export STATE8_EEDI_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_MALRULE_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_OTHER_DIAGNOSIS_ANCHOR_SCHEDULE=1.0,0.5,0.25
export STATE8_T6_ANCHOR_SCHEDULE=1.0,1.0,1.0
export STATE8_OTHER_ANCHOR_WEIGHT=0.1
export STATE8_TRAJECTORY_COSINE_WEIGHT=1.0
export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=1.0
export STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT=0.15
export STATE8_CAUSAL_TEACHER_COSINE_WEIGHT=1.0
export STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT=1.0
export STATE8_CURRENT_QUERY_ROUTING=eedi_only
export STATE8_REPAIR_SAVE_STEPS=100
export STATE8_REPAIR_RECOVERY_SAVE_STEPS=100
export STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256=e38c93ac1f44779716fb9f82ad9ced47a88d1cd91c1319354f47168b1088911e
export STATE8_SOURCE_ANCHOR_ENTRYPOINT_SHA256=f3f2954cb93df4d312078f727648a3563c85f71ce95a9799a4c9a75bdeb25265
export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f
export STATE8_DIAGNOSIS_FIRST_C4_SHA256=660df80664c6012f6eac1d379cc1d611c9f085c91496d1c9cc8e7f7bcb30d7ca
'''
replace('export STATE8_RENDER_ONLY=1\n', insertion + 'export STATE8_RENDER_ONLY=1\n', 0)
# The rendered file no longer contains the wrapper's render-only marker. Add
# the intervention contract immediately before model/data execution.
replace('export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n',
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' + insertion)
replace(
    'test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d\' \' -f1)" = e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664',
    'test "$(sha256sum scripts/evaluate_recurrent_diagnosis_c4.py | cut -d\' \' -f1)" = e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664\n'
    'test "$(sha256sum scripts/evaluate_state8_eedi_label_readout_c4.py | cut -d\' \' -f1)" = fb550be116d2587d52d6900cc1075d2ea47a32d5ebe38230f816a4910d46f3b6',
)
replace(
    " assert s['final_inference_route']=='strict_state_only',s",
    " assert s['final_inference_route']=='strict_state_only',s\n"
    " assert s['early_label_query_mode']=='state_only' and s['early_label_current_query_visible'] is False,s\n"
    " assert s['early_label_random_weight']==0.5 and s['early_label_same_rule_weight']==0.5,s\n"
    " d=json.load(open(root/'diagnosis_first_causal_repair.json'))\n"
    " assert d['field_order']==['label','diagnosis','evidence'] and d['state_identity_first'] is True,d",
)
replace(
    'rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval"',
    '''export LABEL_READOUT_OUT="$AMLT_OUTPUT_DIR/eval/eedi_label_readout_25"
python -u scripts/evaluate_state8_eedi_label_readout_c4.py \\
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \\
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --causal-jsonl "$EEDI_CAUSAL_VALID" \\
  --output-dir "$LABEL_READOUT_OUT" --state-architecture layerwise \\
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 24 \\
  --num-pairs 25 --seed 20260811 \\
  --expected-adapter-sha256 "${ARTIFACT_SHA256[0]}" \\
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA256[1]}"

rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval"''',
)
output.write_text(value, encoding="utf-8")
print(hashlib.sha256(value.encode()).hexdigest())
PY

bash -n "$rendered"
if [[ "${STATE8_LABEL_FIRST_RENDER_ONLY:-0}" == 1 ]]; then
  exit 0
fi
bash "$rendered"
