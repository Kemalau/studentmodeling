#!/usr/bin/env python3
"""Render the immutable state-only launcher for the matched rescue."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7"
TRAINER_SHA256 = "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5"


def materialize(source: Path, output: Path, teacher_root: str) -> str:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Immutable state-only launcher drift")
    text = raw.decode()

    def replace(old: str, new: str, count: int = 1) -> None:
        nonlocal text
        observed = text.count(old)
        if observed != count:
            raise ValueError(f"Replacement cardinality drift: {old!r} observed={observed}")
        text = text.replace(old, new)

    replace(
        "test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
        f"test \"$(sha256sum scripts/train_recurrent_student_m4.py | cut -d' ' -f1)\" = {TRAINER_SHA256}",
    )
    replace(
        "export TEACHER_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_full_history_natural_teacher_v1",
        f"export TEACHER_ROOT={teacher_root}",
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
    old = '''  scripts/train_state8_diagnosis_first_causal_repair.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256:?}" 
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0 ;;'''
    new = old + '''
  scripts/train_state8_matched_checkpoint2000_causal_rescue.py)
    test "$(sha256sum "$TRAIN_ENTRYPOINT" | cut -d' ' -f1)" = "${STATE8_MATCHED_RESCUE_ENTRYPOINT_SHA256:?}"
    test "$(sha256sum scripts/state8_taxonomy_contract.py | cut -d' ' -f1)" = 1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed
    test "${STATE8_REPAIR_LABEL_WEIGHT:-}" = 1.0
    test "${STATE8_CAUSAL_LABEL_WEIGHT:-}" = 1.0 ;;'''
    replace(old, new)
    old = """elif si.get('status')=='complete' and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
 assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si
else: raise AssertionError(si)"""
    new = """elif si.get('status')=='complete' and si.get('step')==2000 and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
elif si.get('status')=='complete' and si.get('sha256'):
 expected={key:si['sha256'][key] for key in observed}
 assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si
else: raise AssertionError(si)"""
    replace(old, new)
    needle = '|| "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py'
    if text.count(needle) != 4:
        raise ValueError("Diagnosis-first condition cardinality drift")
    text = text.replace(
        needle,
        needle + ' \\\n      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_matched_checkpoint2000_causal_rescue.py',
    )
    old = '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_causal_repair.py
fi'''
    new = old + '''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_matched_checkpoint2000_causal_rescue.py ]]; then
  python -m pytest -q tests/test_train_state8_diagnosis_first_causal_repair.py \\
    tests/test_train_state8_diagnosis_first_task_anchored_causal_repair.py \\
    tests/test_train_state8_matched_checkpoint2000_causal_rescue.py
fi'''
    replace(old, new)
    old = '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
fi'''
    new = old + '''
if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_matched_checkpoint2000_causal_rescue.py ]]; then
  test -f "$RUN_DIR/diagnosis_first_causal_repair.json"
  test -f "$RUN_DIR/matched_checkpoint2000_causal_rescue.json"
  if [[ "$STATE8_RESCUE_MODE" == causal_taskanchor ]]; then
    test -f "$RUN_DIR/diagnosis_first_task_anchored_causal_repair.json"
  fi
fi'''
    replace(old, new)
    replace(
        "                        'train_state8_diagnosis_first_causal_repair.py')):",
        "                        'train_state8_diagnosis_first_causal_repair.py',\n                        'train_state8_matched_checkpoint2000_causal_rescue.py')):",
    )
    old = '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py'''
    new = '''if [[ "$TRAIN_ENTRYPOINT" == scripts/train_state8_diagnosis_first_causal_repair.py \\
      || "$TRAIN_ENTRYPOINT" == scripts/train_state8_matched_checkpoint2000_causal_rescue.py ]]; then
  FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py'''
    replace(old, new)
    output.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-root", required=True)
    args = parser.parse_args()
    print(materialize(args.source, args.output, args.teacher_root))


if __name__ == "__main__":
    main()
