#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_SAME_QUESTION_CAUSAL_REPAIR_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

# Keep the qualified State8 repair recipe immutable and introduce one isolated
# variable: natural Eedi causal pairs whose decoder query is byte-identical.
source_runner=scripts/run_state8_source_anchor_screen.sh
patched_runner=/tmp/run_state8_eedi_same_question_source_anchor.rendered.sh
expected_source=9340b550330af6536576313f6b017983351a55f59466424be66b51e3a641f4f5
test "$(sha256sum "$source_runner" | cut -d' ' -f1)" = "$expected_source"
test "$(sha256sum scripts/build_state8_eedi_same_question_causal_twins.py | cut -d' ' -f1)" = fcb25027ea7a35eb3a67a2cb3f2fceeae22f1e10a7ce98b0a7c243c2a3ae7c4d
test "$(sha256sum scripts/merge_state8_causal_training_rows.py | cut -d' ' -f1)" = 408e89a0dc1cb803a274cb714f7ce5bbb50b1ffc7b3024b5227b5d655bbc9a1f
test "$(sha256sum scripts/evaluate_state8_eedi_same_question_c4.py | cut -d' ' -f1)" = 4b9911ce6051a04340ce2b1a2e0b8cb1806b4dac20000a7e5a9ed130aac82352
test "$(sha256sum scripts/state8_current_query_routing.py | cut -d' ' -f1)" = b48bfabc2f48df1a422f46d52a9bfda9c915dce3385c50391c4ba3ab623ac875

python - "$source_runner" "$patched_runner" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")

def replace(old: str, new: str, count: int = 1) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

# Bind the current-query entrypoint and its Eedi-aware formal evaluator.
replace(
    "export RUN_TAG=source_anchor_updater_mse1_v1",
    "export RUN_TAG=source_anchor_updater_mse1_eedi_same_question_v1",
)
replace(
    "export RUN_TAG=source_anchor_full_mse1_v1",
    "export RUN_TAG=source_anchor_full_mse1_eedi_same_question_v1",
)
replace(
    "export RUN_TAG=source_anchor_full_mse2_v1",
    "export RUN_TAG=source_anchor_full_mse2_eedi_same_question_v1",
)
replace(
    "export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f",
)

# These replacements are executed inside the source-anchor renderer, against
# the final base repair launcher.  The ordinary rows and MalRule causal rows
# remain unchanged; 100 Eedi training twins and 25 disjoint validation twins
# are appended with a hash-bound receipt.
insertion = r"""
replace(
    'flock -u 8\n',
    '''flock -u 8
export EEDI_CAUSAL_ROOT=/tmp/state8_eedi_same_question_causal
export EEDI_CAUSAL_TRAIN="$EEDI_CAUSAL_ROOT/train.jsonl"
export EEDI_CAUSAL_VALID="$EEDI_CAUSAL_ROOT/valid.jsonl"
export CAUSAL_MERGED=/tmp/state8_repair_c4_with_eedi_same_question
rm -rf "$EEDI_CAUSAL_ROOT" "$CAUSAL_MERGED"
mkdir -p "$EEDI_CAUSAL_ROOT" "$CAUSAL_MERGED"
python scripts/build_state8_eedi_same_question_causal_twins.py \\
  --source-jsonl "$FULL_JSONL" --output-jsonl "$EEDI_CAUSAL_TRAIN" \\
  --receipt "$EEDI_CAUSAL_ROOT/train.receipt.json" \\
  --twins 100 --seed 20260811 --offset 0
python scripts/build_state8_eedi_same_question_causal_twins.py \\
  --source-jsonl "$FULL_JSONL" --output-jsonl "$EEDI_CAUSAL_VALID" \\
  --receipt "$EEDI_CAUSAL_ROOT/valid.receipt.json" \\
  --twins 25 --seed 20260811 --offset 100 \\
  --exclude-jsonl "$EEDI_CAUSAL_TRAIN"
python scripts/merge_state8_causal_training_rows.py \\
  --input "$CAUSAL_DATA/train.jsonl" --input "$EEDI_CAUSAL_TRAIN" \\
  --output-jsonl "$CAUSAL_MERGED/train.jsonl" \\
  --receipt "$CAUSAL_MERGED/train.receipt.json"
python scripts/merge_state8_causal_training_rows.py \\
  --input "$CAUSAL_DATA/valid.jsonl" --input "$EEDI_CAUSAL_VALID" \\
  --output-jsonl "$CAUSAL_MERGED/valid.jsonl" \\
  --receipt "$CAUSAL_MERGED/valid.receipt.json"
export CAUSAL_TRAIN_JSONL="$CAUSAL_MERGED/train.jsonl"
export CAUSAL_VALID_JSONL="$CAUSAL_MERGED/valid.jsonl"
python - "$EEDI_CAUSAL_ROOT/train.receipt.json" \\
  "$EEDI_CAUSAL_ROOT/valid.receipt.json" \\
  "$CAUSAL_MERGED/train.receipt.json" <<'PY_EEDI_CAUSAL_AUDIT'
import json,os,sys
train,valid,merged=(json.load(open(path)) for path in sys.argv[1:])
assert train['output_pairs']==200 and train['offset']==0,train
assert valid['output_pairs']==50 and valid['offset']==100,valid
assert train['same_question_query_identical'] is True,train
assert valid['same_question_query_identical'] is True,valid
assert train['student_response_visible_to_decoder'] is False,train
assert train['raw_history_visible_to_decoder'] is False,train
assert valid['excluded_entity_ids']>0 and valid['excluded_base_ids']>0,valid
train_rows=[json.loads(line) for line in open(os.environ['EEDI_CAUSAL_TRAIN'])]
valid_rows=[json.loads(line) for line in open(os.environ['EEDI_CAUSAL_VALID'])]
fields=('recipient_id','donor_id','same_rule_control_id')
train_entities={row[key] for row in train_rows for key in fields}
valid_entities={row[key] for row in valid_rows for key in fields}
assert not train_entities & valid_entities,(len(train_entities),len(valid_entities))
assert not ({row['same_question_base_id'] for row in train_rows} &
            {row['same_question_base_id'] for row in valid_rows})
assert not ({row['pair_id'] for row in train_rows} & {row['pair_id'] for row in valid_rows})
assert merged['rows']==700,merged
assert merged['source_counts']=={'eedi_mining':200,'malrule':500},merged
PY_EEDI_CAUSAL_AUDIT
''',
)
replace(
    "expected_pairs=250 if os.environ['STATE8_CAUSAL_DIAGNOSIS_ONLY']=='1' else 500",
    "expected_pairs=250 if os.environ['STATE8_CAUSAL_DIAGNOSIS_ONLY']=='1' else 700",
)
replace(
    'cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_augmentation.json"',
    '''cp "$CAUSAL_DATA/augmentation_receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_augmentation.json"
cp "$EEDI_CAUSAL_ROOT/train.receipt.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_train.json"
cp "$EEDI_CAUSAL_ROOT/valid.receipt.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_valid.json"
cp "$CAUSAL_MERGED/train.receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_merged_train.json"
cp "$CAUSAL_MERGED/valid.receipt.json" "$AMLT_OUTPUT_DIR/metrics/causal_merged_valid.json"''',
)
replace(
    'rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval"',
    '''export EEDI_C4_OUT="$AMLT_OUTPUT_DIR/eval/eedi_same_question_c4_25"
python -u scripts/evaluate_state8_eedi_same_question_c4.py \\
  --model "$LOCAL_MODEL" --adapter "$RUN_DIR/final_adapter" \\
  --recurrent-state "$RUN_DIR/recurrent_state.pt" --state-architecture layerwise \\
  --causal-jsonl "$EEDI_CAUSAL_VALID" --output-dir "$EEDI_C4_OUT" \\
  --state-token-alias '<|vision_pad|>' --max-length 1536 --max-new-tokens 160 \\
  --num-pairs 25 --seed 20260811 \\
  --expected-adapter-sha256 "${ARTIFACT_SHA256[0]}" \\
  --expected-recurrent-state-sha256 "${ARTIFACT_SHA256[1]}"

rm -rf "$STABLE_ROOT/model" "$STABLE_ROOT/metrics" "$STABLE_ROOT/eval"''',
)
"""
needle = "output.write_text(text)\nprint(hashlib.sha256(text.encode()).hexdigest())"
replace(needle, insertion + "\n" + needle)

# Match the teacher/student query while retaining strict state-only decoding.
replace(
    'bash -n "$rendered"\nbash "$rendered"',
    '''python - "$rendered" <<'PY_CURRENT_QUERY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
value = path.read_text(encoding="utf-8")
old = "FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py"
new = "FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py"
assert value.count(old) == 1, value.count(old)
path.write_text(value.replace(old, new), encoding="utf-8")
PY_CURRENT_QUERY
bash -n "$rendered"
if [[ "${STATE8_RENDER_ONLY:-0}" == 1 ]]; then
  cp "$rendered" /tmp/run_state8_eedi_same_question_causal_repair.final.sh
  exit 0
fi
bash "$rendered"''',
)
output.write_text(text, encoding="utf-8")
PY

export STATE8_CURRENT_QUERY_ROUTING=eedi_only
bash -n "$patched_runner"
bash "$patched_runner"
