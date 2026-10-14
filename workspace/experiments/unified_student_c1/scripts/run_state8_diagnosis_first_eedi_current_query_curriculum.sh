#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_CURRENT_QUERY_CURRICULUM_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

source_runner=scripts/run_state8_diagnosis_first_eedi_curriculum.sh
rendered=/tmp/run_state8_diagnosis_first_eedi_current_query_curriculum.rendered.sh
expected_source=123733ddbfa7743f910be1ab1bedcdbfd9bb379d19ea106c2c92540c246134fe
test "$(sha256sum "$source_runner" | cut -d' ' -f1)" = "$expected_source"
test "$(sha256sum scripts/state8_current_query_routing.py | cut -d' ' -f1)" = b48bfabc2f48df1a422f46d52a9bfda9c915dce3385c50391c4ba3ab623ac875
test -f scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py

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
    "export STATE8_RUN_SUFFIX=diagnosis_first_eedi_curriculum_seed83",
    "export STATE8_RUN_SUFFIX=diagnosis_first_eedi_current_query_curriculum_seed83",
)
replace(
    "diagnosis_first_eedi_curriculum_seed83_single_ga4",
    "diagnosis_first_eedi_current_query_curriculum_seed83_single_ga4",
    1,
)
replace(
    "test -f scripts/train_state8_diagnosis_first_eedi_curriculum.py\n"
    "test -f scripts/evaluate_state8_eedi_same_question_c4.py",
    "test -f scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py\n"
    "test -f scripts/evaluate_state8_eedi_same_question_c4.py",
    1,
)
replace(
    '"export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_eedi_curriculum.py",',
    '"export TRAIN_ENTRYPOINT=scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py",',
    1,
)
replace(
    '"scripts/train_state8_diagnosis_first_eedi_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)",',
    '"scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)",',
    1,
)
replace(
    '''      test -f scripts/train_state8_diagnosis_first_eedi_curriculum.py
      test -f scripts/build_state8_eedi_same_question_causal_twins.py''',
    '''      test -f scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py
      test -f scripts/build_state8_eedi_same_question_causal_twins.py''',
    1,
)
replace(
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_eedi_curriculum.py ]]; then''',
    '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py ]]; then''',
)
replace(
    '''  test -f "$RUN_DIR/eedi_same_question_valid.jsonl"
fi''',
    '''  test -f "$RUN_DIR/eedi_same_question_valid.jsonl"
  test -f "$RUN_DIR/eedi_current_query_curriculum.json"
fi''',
)
replace(
    '''for name in eedi_curriculum.json eedi_causal_merge.json; do''',
    '''for name in eedi_curriculum.json eedi_causal_merge.json eedi_current_query_curriculum.json; do''',
)
# Keep training and formal-screen information routes identical.  This string
# is executed by the outer renderer against the qualified parent launcher.
replace(
    ''' "scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py:1.0)",''',
    ''' "scripts/train_state8_diagnosis_first_eedi_current_query_curriculum.py:scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py:1.0)",''',
)
replace(
    ''' "--causal-objective-mode specificity_triplet --causal-repeat-fraction 0.05 --normalize-causal-exposure",''',
    ''' "FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py",
 "FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py")
text=text.replace(
 "--causal-objective-mode specificity_triplet --causal-repeat-fraction 0.05 --normalize-causal-exposure",''',
)
replace(
    '''      test -f scripts/build_state8_eedi_same_question_causal_twins.py''',
    '''      test -f scripts/build_state8_eedi_same_question_causal_twins.py
      test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py | cut -d' ' -f1)" = 967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f''',
)
output.write_text(text, encoding="utf-8")
PY

bash -n "$rendered"
if [[ "${STATE8_RENDER_ONLY:-0}" == 1 ]]; then
  exit 0
fi
bash "$rendered"
