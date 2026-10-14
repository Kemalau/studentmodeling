#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_CURRICULUM_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

# Start from the qualified full-history teacher curriculum.  The rendered
# launcher changes only the student entrypoint, causal data, final-phase
# schedule, immutable namespace, and frozen evaluation panel.
export STATE8_RUN_SUFFIX=diagnosis_first_eedi_curriculum_seed83
export TRAIN_SEED=83
export STATE8_CAUSAL_REPEAT_FRACTION=0.10
export STATE8_EEDI_STABLE_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_diagnosis_first_eedi_curriculum_seed83_single_ga4
# The qualified parent launcher runs in a child shell, so its exported RUN_DIR
# does not propagate back here.  Reconstruct the same immutable checkpoint path
# from the already-bound suffix before running the Eedi-only publication tail.
export STATE8_EEDI_RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_${STATE8_RUN_SUFFIX}_single_ga4"

source_runner=scripts/run_state8_diagnosis_first_matched_curriculum.sh
rendered=/tmp/run_state8_diagnosis_first_eedi_curriculum.rendered.sh
expected_source=1bb2bb9e7d3ec12b74d54a3aed5bbd8093395721bc9a62e0bd31cf09fdd2f925
test "$(sha256sum "$source_runner" | cut -d' ' -f1)" = "$expected_source"
test -f scripts/train_state8_diagnosis_first_eedi_curriculum.py
test -f scripts/evaluate_state8_eedi_same_question_c4.py

python - "$source_runner" "$rendered" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")

def replace(old: str, new: str, count: int = 1) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

replace(
    "export STATE8_RUN_SUFFIX=diagnosis_first_matched_teacher_seed67",
    'export STATE8_RUN_SUFFIX="${STATE8_RUN_SUFFIX:?}"',
)
replace("export TRAIN_SEED=67", 'export TRAIN_SEED="${TRAIN_SEED:?}"')
replace(
    "export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py",
    "export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_eedi_curriculum.py",
)
replace(
    "scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)",
    "scripts/train_state8_diagnosis_first_eedi_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)",
)
replace(
    '''      test "$(sha256sum scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py | cut -d' ' -f1)" = 0f4e305fa5eb48ec2beaa328a3b9c125b649f28327a405a211a1803570498a1a''',
    '''      test -f scripts/train_state8_diagnosis_first_eedi_curriculum.py
      test -f scripts/build_state8_eedi_same_question_causal_twins.py''',
)
replace(
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_matched_teacher_curriculum.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_teacher_curriculum.json"
  test -f "$RUN_DIR/matched_teacher_curriculum.json"
fi''',
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_eedi_curriculum.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_teacher_curriculum.json"
  test -f "$RUN_DIR/matched_teacher_curriculum.json"
  test -f "$RUN_DIR/eedi_curriculum.json"
  test -f "$RUN_DIR/eedi_causal_merge.json"
  test -f "$RUN_DIR/eedi_same_question_valid.jsonl"
fi''',
)
replace(
    "bash /tmp/run_state8_diagnosis_first_matched_curriculum.rendered.sh",
    r'''if [[ -f "${STATE8_EEDI_STABLE_ROOT:?}/metrics/immutable_receipt.json" \
   && -f "${STATE8_EEDI_RUN_DIR:?}/completion.json" ]]; then
  # Recover a completed train/formal/publish attempt without repeating any
  # optimizer steps or the 300-row formal generation pass.
  "$VENV_ROOT/bin/python" - "$STATE8_EEDI_RUN_DIR" "$STATE8_EEDI_STABLE_ROOT" <<'PY_TAIL_RECOVERY'
import hashlib, json, pathlib, sys
run, stable = map(pathlib.Path, sys.argv[1:])
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()
completion = json.load(open(run / 'completion.json'))
receipt = json.load(open(stable / 'metrics/immutable_receipt.json'))
observed = {
    'adapter_model': digest(stable / 'model/adapter_model.safetensors'),
    'recurrent_state': digest(stable / 'model/recurrent_state.pt'),
}
assert completion['status'] == 'complete' and completion['micro_steps'] == 5000, completion
assert completion['real_examples_seen_once'] is True, completion
assert completion['all_causal_pairs_covered'] is True, completion
assert completion['causal_exposure_count_matches_plan'] is True, completion
assert receipt['status'] == 'complete' and receipt['state_shape'] == [64, 8, 5120], receipt
assert receipt['raw_history_visible_to_decoder'] is False, receipt
assert receipt['persistent_history_kv_cache'] is False, receipt
assert receipt['sha256'] == observed, (receipt, observed)
for path in (
    run / 'eedi_curriculum.json',
    run / 'eedi_causal_merge.json',
    run / 'eedi_same_question_valid.jsonl',
    stable / 'eval/formal_screen/metrics.json',
    stable / 'eval/formal_screen/predictions.jsonl',
):
    assert path.is_file() and path.stat().st_size > 0, path
PY_TAIL_RECOVERY
  mkdir -p "$AMLT_OUTPUT_DIR/eval/formal_screen" "$AMLT_OUTPUT_DIR/metrics"
  cp "$STATE8_EEDI_STABLE_ROOT/eval/formal_screen/metrics.json" \
    "$AMLT_OUTPUT_DIR/eval/formal_screen/metrics.json"
  cp "$STATE8_EEDI_STABLE_ROOT/eval/formal_screen/predictions.jsonl" \
    "$AMLT_OUTPUT_DIR/eval/formal_screen/predictions.jsonl"
else
  bash /tmp/run_state8_diagnosis_first_matched_curriculum.rendered.sh
fi''',
)
replace(
    '''text=text.replace("assert cfg['final_inference_route']=='strict_state_only',cfg",''',
    '''text=text.replace(
 "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
 "7d5e4966a5317733fe5d5f32cc6730eb82f4ad78dedd8fa6a6967ee4aefb4e6d")
text=text.replace(
 "--causal-objective-mode specificity_triplet --causal-repeat-fraction 0.05 --normalize-causal-exposure",
 "--causal-objective-mode specificity_triplet --causal-repeat-fraction ${STATE8_CAUSAL_REPEAT_FRACTION:?} --normalize-causal-exposure")
text=text.replace("assert cfg['final_inference_route']=='strict_state_only',cfg",''',
)

# Replace the legacy MalRule-only tail evaluation with an identical-query Eedi
# intervention panel bound to the newly trained checkpoint.
start = text.index("# Generate the frozen natural-language C4 panel")
text = text[:start] + r'''# Publish the final-phase scientific receipts omitted by the parent launcher.
RUN_DIR="${STATE8_EEDI_RUN_DIR:?}"
for name in eedi_curriculum.json eedi_causal_merge.json; do
  cp "$RUN_DIR/$name" "$AMLT_OUTPUT_DIR/metrics/$name"
  cp "$RUN_DIR/$name" "$STATE8_EEDI_STABLE_ROOT/metrics/$name"
done

# Generate the frozen same-question Eedi C4 panel on the trained checkpoint.
EEDI_C4_OUT="$AMLT_OUTPUT_DIR/eval/eedi_same_question_c4_25"
MATCHED_STABLE_ROOT="$STATE8_EEDI_STABLE_ROOT"
MODEL_REMOTE=/mnt/amulet/studentmodeling/models/Qwen3-32B
MODEL_LOCAL=/tmp/Qwen3-32B
# A tail-only exact-resume attempt can land on a fresh node after training and
# formal generation have already completed remotely.  In that case the parent
# launcher's training branch is intentionally skipped, so its node-local model
# copy does not exist.  Materialize the immutable remote backbone here as a
# prerequisite of C4 instead of letting Transformers interpret the missing
# absolute path as a Hugging Face repository ID.
if [[ ! -f "$MODEL_LOCAL/model.safetensors.index.json" ]]; then
  test -f "$MODEL_REMOTE/model.safetensors.index.json"
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  "$VENV_ROOT/bin/python" scripts/copy_model_parallel.py \
    --source "$MODEL_REMOTE" --destination "$MODEL_LOCAL" --workers 16
fi
test -f "$MODEL_LOCAL/model.safetensors.index.json"
readarray -t artifact_hashes < <("$VENV_ROOT/bin/python" - "$MATCHED_STABLE_ROOT" <<'PY_HASH'
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
assert r['state_shape']==[64,8,5120] and r['raw_history_visible_to_decoder'] is False,r
assert r['sha256']==observed,(r,observed)
print(observed['adapter_model']); print(observed['recurrent_state'])
PY_HASH
)
test "${#artifact_hashes[@]}" = 2
rm -rf "$EEDI_C4_OUT"; mkdir -p "$EEDI_C4_OUT"
"$VENV_ROOT/bin/python" -u scripts/evaluate_state8_eedi_same_question_c4.py \
  --model "$MODEL_LOCAL" --adapter "$MATCHED_STABLE_ROOT/model" \
  --recurrent-state "$MATCHED_STABLE_ROOT/model/recurrent_state.pt" \
  --state-architecture layerwise --causal-jsonl "$RUN_DIR/eedi_same_question_valid.jsonl" \
  --output-dir "$EEDI_C4_OUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --max-new-tokens 192 --num-pairs 25 --seed 20260812 \
  --expected-adapter-sha256 "${artifact_hashes[0]}" \
  --expected-recurrent-state-sha256 "${artifact_hashes[1]}"
"$VENV_ROOT/bin/python" - "$EEDI_C4_OUT/metrics.json" <<'PY_C4'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['num_pairs']==25 and x['state_shape']==[64,8,5120],x
assert x['conditions']==['native','noop','samemal','random','swap'],x
assert x['same_question_query_identical'] is True,x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['semantic_judgment_pending'] is True,x
PY_C4
'''
output.write_text(text, encoding="utf-8")
PY

bash -n "$rendered"
if [[ "${STATE8_RENDER_ONLY:-0}" == 1 ]]; then
  exit 0
fi
bash "$rendered"
