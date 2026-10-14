#!/usr/bin/env python3
"""Render the frozen exact-epoch launcher for promoted same-question State8."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "79d7f9ab7068c4e48224b6daeb51a3d35a34a6dde906b242de9da8d00900ed17"
VARIANTS = (
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
)


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    observed = text.count(old)
    if observed != count:
        raise ValueError(
            f"replacement drift expected={count} observed={observed} old={old!r}"
        )
    return text.replace(old, new)


def render(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise ValueError(f"frozen launcher hash drift: {observed}")
    text = raw.decode("utf-8")

    # Add the two promoted pilot lineages without changing any legacy recipe.
    marker = "  diagnosis_first_matched_cp3000_semantic_seed3066)"
    start = text.index(marker)
    stop = text.index("  *) exit 2 ;;", start)
    block = text[start:stop]
    insertion = '''  same_question_updater_mse1_seed3072|same_question_full_mse1_seed3071|same_question_repair_promoted|same_question_repair_checkpoint_promoted|same_question_curriculum_promoted)
    case "$FINALFIT_VARIANT" in
      same_question_updater_mse1_seed3072)
        export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed3072_taskanchor_source_anchor_updater_mse1_eedi_same_question_v1 ;;
      same_question_full_mse1_seed3071)
        export RESULT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed3071_taskanchor_source_anchor_full_mse1_eedi_same_question_v1 ;;
      same_question_repair_promoted)
        export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_eedi_same_question_v1/promotion.json
        export RESULT_ROOT="$(python - "$PROMOTION_GATE" <<'PY_REPAIR_ROOT'
import json,sys
x=json.load(open(sys.argv[1])); s=x.get('selected') or {}; root=s.get('remote_root')
assert x.get('stage')=='state8_eedi_same_question_joint_promotion_v1' and x.get('pass') is True,x
assert s.get('qualified') is True and s.get('finalfit_variant')=='same_question_repair_promoted',s
assert isinstance(root,str) and root.startswith('/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed'),root
print(root)
PY_REPAIR_ROOT
)" ;;
      same_question_repair_checkpoint_promoted)
        export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_eedi_same_question_v1/promotion.json
        export RESULT_ROOT="$(python - "$PROMOTION_GATE" <<'PY_REPAIR_CHECKPOINT_ROOT'
import json,sys
x=json.load(open(sys.argv[1])); s=x.get('selected') or {}; root=s.get('remote_root')
assert x.get('stage')=='state8_eedi_same_question_joint_promotion_v1' and x.get('pass') is True,x
assert s.get('qualified') is True and s.get('finalfit_variant')=='same_question_repair_checkpoint_promoted',s
assert s.get('artifact_layout')=='repair_checkpoint_snapshot',s
seed=int(s.get('repair_seed',-1)); source_step=int(s.get('source_checkpoint_step',-1))
assert seed in {5201,5202,5203,6201,6202,6203} and s.get('checkpoint_step')==500,s
assert source_step==(4000 if seed//1000==5 else 3000),s
assert isinstance(root,str) and root.startswith('/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_curriculum_repairs_20260812/'),root
print(root)
PY_REPAIR_CHECKPOINT_ROOT
)" ;;
      same_question_curriculum_promoted)
        export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_curriculum_v1/promotion.json
        export RESULT_ROOT="$(python - "$PROMOTION_GATE" <<'PY_CURRICULUM_ROOT'
import json,sys
x=json.load(open(sys.argv[1])); s=x.get('selected') or {}; root=s.get('remote_root')
assert x.get('stage')=='state8_eedi_same_question_joint_promotion_v1' and x.get('pass') is True,x
assert s.get('qualified') is True and s.get('finalfit_variant')=='same_question_curriculum_promoted',s
assert s.get('source_history_mode')=='five_stage_curriculum',s
assert isinstance(root,str) and root.startswith('/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_'),root
print(root)
PY_CURRICULUM_ROOT
)" ;;
    esac
    if [[ "$FINALFIT_VARIANT" == same_question_curriculum_promoted ]]; then
      export EXPECTED_IMMUTABLE_STAGE=state8_curriculum_checkpoint_promoted_v1
      export INIT_HISTORY_MODE=five_stage_curriculum
    elif [[ "$FINALFIT_VARIANT" == same_question_repair_checkpoint_promoted ]]; then
      export EXPECTED_IMMUTABLE_STAGE=state8_repair_checkpoint_promoted_v1
      export INIT_HISTORY_MODE=strict_state_only
    else
      export EXPECTED_IMMUTABLE_STAGE=state8_stateonly_repair_pilot_publish_v1
      export INIT_HISTORY_MODE=strict_state_only
    fi
    export STATE8_FINALFIT_ONLINE_KD_WEIGHT=1.0
    export STATE8_FINALFIT_ENTRYPOINT=scripts/train_state8_eedi_same_question_finalfit.py
    export STATE8_FINALFIT_LEARNING_RATE=5e-7
    export EXPECTED_HISTORY_MODE=strict_state_only
    export STATE8_REPAIR_LABEL_WEIGHT=1.0
    export STATE8_CAUSAL_LABEL_WEIGHT=1.0
    export STATE8_REPAIR_TRAJECTORY_WEIGHT=1.0
    export STATE8_RANDOM_NEUTRAL=1
    export STATE8_CONTROL_RANDOM_SAMPLES=4
    export STATE8_CONTROL_RANDOM_WEIGHT=0.25
    export STATE8_CONTROL_SAME_RULE_WEIGHT=0.50
    export STATE8_GEOMETRY_WEIGHT=0.10
    export STATE8_EEDI_GEOMETRY_WEIGHT=0.05
    export STATE8_DIFFERENT_MAX_COSINE=0.90
    export STATE8_CAUSAL_WEIGHT=2.0
    export STATE8_CAUSAL_MARGIN=0.5
    export STATE8_CAUSAL_REPEAT_FRACTION=0.05
    export STATE8_CURRENT_QUERY_ROUTING=eedi_only
    export STATE8_EEDI_ANCHOR_SCHEDULE=1.0,1.0,1.0
    export STATE8_MALRULE_ANCHOR_SCHEDULE=1.0,1.0,1.0
    export STATE8_OTHER_DIAGNOSIS_ANCHOR_SCHEDULE=1.0,0.5,0.25
    export STATE8_T6_ANCHOR_SCHEDULE=0.5,0.75,1.0
    export STATE8_OTHER_ANCHOR_WEIGHT=0.1
    export STATE8_TRAJECTORY_COSINE_WEIGHT=1.0
    export STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT=1.0
    if [[ "$FINALFIT_VARIANT" == same_question_curriculum_promoted ]]; then
      export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_curriculum_v1/promotion.json
    else
      export PROMOTION_GATE=/mnt/amulet/studentmodeling/gates/state8_eedi_same_question_v1/promotion.json
    fi
    export STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT="$(python - "$PROMOTION_GATE" <<'PY_PROMOTED_BINDING'
import json,sys
s=(json.load(open(sys.argv[1])).get('selected') or {})
branch=str(s.get('repair_branch') or '')
print('0.15' if branch in {'causal_teacher_low','early_label_low','early_label_high'} else ('0.50' if branch=='causal_teacher_high' else '0'))
PY_PROMOTED_BINDING
)"
    export STATE8_CAUSAL_TEACHER_COSINE_WEIGHT=1.0
    export STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT=1.0
    export STATE8_EARLY_LABEL_BINDING_WEIGHT="$(python - "$PROMOTION_GATE" <<'PY_PROMOTED_EARLY_LABEL'
import json,sys
s=(json.load(open(sys.argv[1])).get('selected') or {})
branch=str(s.get('repair_branch') or '')
print('0.25' if branch=='early_label_low' else ('0.50' if branch=='early_label_high' else '0'))
PY_PROMOTED_EARLY_LABEL
)" ;;
'''
    text = text[:stop] + insertion + text[stop:]

    text = replace_exact(
        text,
        "if [[ \"$FINALFIT_VARIANT\" == diagnosis_first_matched_seed67 \\\n      || \"$FINALFIT_VARIANT\" == diagnosis_first_matched_cp3000_semantic_seed3066 \\\n      || \"$FINALFIT_VARIANT\" == diagnosis_first_matched_taskanchor_seed2067 ]]; then",
        "if [[ \"$FINALFIT_VARIANT\" == diagnosis_first_matched_seed67 \\\n      || \"$FINALFIT_VARIANT\" == diagnosis_first_matched_cp3000_semantic_seed3066 \\\n      || \"$FINALFIT_VARIANT\" == diagnosis_first_matched_taskanchor_seed2067 \\\n      || \"$FINALFIT_VARIANT\" == same_question_* ]]; then",
        1,
    )
    text = replace_exact(
        text,
        "if [[ \"$FINALFIT_VARIANT\" == diagnosis_first_matched_taskanchor_seed2067 ]]; then\n  export RUN_DIR=",
        "if [[ \"$FINALFIT_VARIANT\" == diagnosis_first_matched_taskanchor_seed2067 ]]; then\n  export RUN_DIR=",
        1,
    )
    anchor = '''if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_taskanchor_seed2067 ]]; then
  export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_first_matched_taskanchor_unified_full259845_seed42_v1_20260811
  export CANONICAL_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_diagnosis_first_matched_taskanchor_unified_canonical_v1_20260811
fi
'''
    text = replace_exact(
        text,
        anchor,
        anchor
        + '''if [[ "$FINALFIT_VARIANT" == same_question_* ]]; then
  export RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/state8_${FINALFIT_VARIANT}_unified_full259845_seed42_v1_20260811"
  export CANONICAL_RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/state8_eedi_same_question_unified_canonical_v1_20260811
fi
''',
    )

    # Promotion is a hash-bound joint T1--T6, dual-diagnosis-C4, answer-C4 gate.
    promotion_anchor = '''if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_seed67 ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); s=x.get('selected') or {}
assert x['stage']=='state8_diagnosis_first_matched_joint_promotion_v1' and x['pass'] is True,x
assert s['slug']=='diagnosis_first_matched_seed67' and s['qualified'] is True,s
assert s['finalfit_variant']=='diagnosis_first_matched_seed67',s
assert s['checks']['answer_C4_pass'] is True and s['checks']['diagnosis_C4_pass'] is True,s
assert x['semantic_judge_authoritative'] is True and x['exact_match_authoritative'] is False,x
PY
fi
'''
    promotion_new = promotion_anchor + '''if [[ "$FINALFIT_VARIANT" == same_question_* ]]; then
  test -f "$PROMOTION_GATE"
  python - "$PROMOTION_GATE" "$FINALFIT_VARIANT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); variant=sys.argv[2]; s=x.get('selected') or {}
assert x['stage']=='state8_eedi_same_question_joint_promotion_v1' and x['pass'] is True,x
assert s['qualified'] is True and s['finalfit_variant']==variant,s
assert s['answer_C4_pass'] is True and s['answer_C4']['scientific_gate_pass'] is True,s
assert s['checks']['malrule_diagnosis_C4_pass'] is True,s
assert s['checks']['eedi_same_question_C4_pass'] is True,s
assert x['semantic_judge_authoritative'] is True and x['exact_match_authoritative'] is False,x
PY
fi
'''
    text = replace_exact(text, promotion_anchor, promotion_new)

    text = replace_exact(
        text,
        '''if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_seed67 ]]; then
  test -f "$TEACHER_ROOT/metrics/teacher_full_gate.json"
  test -f "$TEACHER_ROOT/metrics/teacher_full_provenance_gate.json"
fi''',
        '''if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_seed67 \
      || "$FINALFIT_VARIANT" == same_question_* ]]; then
  test -f "$TEACHER_ROOT/metrics/teacher_full_gate.json"
  test -f "$TEACHER_ROOT/metrics/teacher_full_provenance_gate.json"
fi''',
    )

    # Bind current source hashes and the new entrypoint.
    current_hashes = {
        "scripts/publish_state8_final_checkpoint.py": "2bec746e79f2a09b3a38dd98228f4bc4a57d4169aff75c507857982f0c42a001",
        "scripts/train_state8_diagnosis_first_source_anchored_repair.py": "f9b96fe182641b9563911463ef6eacd8ffef7131ad613ec6c6865043c2b964ff",
        "scripts/train_state8_eedi_same_question_finalfit.py": "a2b236d2bab2a351fc7bdff15367de0c3e5ec1ee30a86cfcbc22a173b647ee6b",
        "scripts/build_state8_eedi_same_question_causal_twins.py": "fcb25027ea7a35eb3a67a2cb3f2fceeae22f1e10a7ce98b0a7c243c2a3ae7c4d",
        "scripts/merge_state8_causal_training_rows.py": "408e89a0dc1cb803a274cb714f7ce5bbb50b1ffc7b3024b5227b5d655bbc9a1f",
        "scripts/train_state8_semantic_causal_repair.py": "8caf665d2e3e0ff3523179289603cba4c3c9f6f20401447137ccf87c8c928598",
    }
    old_publisher = "test \"$(sha256sum scripts/publish_state8_final_checkpoint.py | cut -d' ' -f1)\" = add0de309dde9723090922dccea4f09fa737336ad4bb7ed3e5710ec341c3a629"
    text = replace_exact(
        text,
        old_publisher,
        f'''test "$(sha256sum scripts/publish_state8_final_checkpoint.py | cut -d' ' -f1)" = {current_hashes["scripts/publish_state8_final_checkpoint.py"]}
test "$(sha256sum scripts/train_state8_diagnosis_first_source_anchored_repair.py | cut -d' ' -f1)" = {current_hashes["scripts/train_state8_diagnosis_first_source_anchored_repair.py"]}
test "$(sha256sum scripts/train_state8_eedi_same_question_finalfit.py | cut -d' ' -f1)" = {current_hashes["scripts/train_state8_eedi_same_question_finalfit.py"]}
test "$(sha256sum scripts/build_state8_eedi_same_question_causal_twins.py | cut -d' ' -f1)" = {current_hashes["scripts/build_state8_eedi_same_question_causal_twins.py"]}
test "$(sha256sum scripts/merge_state8_causal_training_rows.py | cut -d' ' -f1)" = {current_hashes["scripts/merge_state8_causal_training_rows.py"]}''',
    )
    text = replace_exact(
        text,
        'test "$(sha256sum scripts/train_state8_semantic_causal_repair.py | cut -d\' \' -f1)" = 14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a',
        f'test "$(sha256sum scripts/train_state8_semantic_causal_repair.py | cut -d\' \' -f1)" = {current_hashes["scripts/train_state8_semantic_causal_repair.py"]}',
    )

    # Treat these variants as diagnosis-first matched Teacher lineages.  This
    # fragment occurs both inside and outside set literals.  Replace the full
    # quoted sequence; slicing the endpoint formerly removed a quote and made
    # the embedded Python invalid even though ``bash -n`` still passed.
    matched_variants = (
        "'diagnosis_first_matched_seed67',"
        "'diagnosis_first_matched_cp3000_semantic_seed3066'"
    )
    matched_with_same_question = matched_variants + (
        ",'same_question_updater_mse1_seed3072'"
        ",'same_question_full_mse1_seed3071'"
        ",'same_question_repair_promoted'"
        ",'same_question_repair_checkpoint_promoted'"
        ",'same_question_curriculum_promoted'"
    )
    if text.count(matched_variants) <= 0:
        raise ValueError("diagnosis-first matched variant sequence drift")
    text = text.replace(matched_variants, matched_with_same_question)

    text = text.replace(
        "|| variant=='strong1_rescue_taskanchor_c4strong_seed62'\n                else 'trajectory01'",
        "|| variant=='strong1_rescue_taskanchor_c4strong_seed62'\n                || variant.startswith('same_question_')\n                else 'trajectory01'",
    )

    # A long exact epoch must retain the control-robust causal objective that
    # qualified the promoted short repair; otherwise C4 can be forgotten while
    # ordinary task likelihood keeps improving.
    semantic_audit = "assert semantic['eedi_pair_supervision_visible_at_inference'] is False,semantic"
    text = replace_exact(
        text,
        semantic_audit,
        semantic_audit
        + "\n  if variant.startswith('same_question_'):\n"
        + "   assert semantic['control_robust_objective'] is True,semantic\n"
        + "   assert semantic['control_random_samples']==4,semantic\n"
        + "   assert float(semantic['control_random_weight'])==0.25,semantic\n"
        + "   assert float(semantic['control_same_rule_weight'])==0.50,semantic\n"
        + "   selected=(json.load(open(os.environ['PROMOTION_GATE'])).get('selected') or {})\n"
        + "   branch=str(selected.get('repair_branch') or '')\n"
        + "   expected_teacher=(0.15 if branch in {'causal_teacher_low','early_label_low','early_label_high'} else (0.50 if branch=='causal_teacher_high' else 0.0))\n"
        + "   expected_early=(0.25 if branch=='early_label_low' else (0.50 if branch=='early_label_high' else 0.0))\n"
        + "   assert float(config.get('causal_teacher_trajectory_weight',0.0))==expected_teacher,config\n"
        + "   assert float(config.get('early_label_binding_weight',0.0))==expected_early,config\n"
        + "   assert float(semantic.get('early_label_binding_weight',0.0))==expected_early,semantic",
    )
    receipt_anchor = " 'semantic_random_target':config.get('semantic_random_target','legacy_recipient'),"
    text = replace_exact(
        text,
        receipt_anchor,
        receipt_anchor
        + "\n 'control_robust_objective':bool(config.get('control_robust_objective',False)),"
        + "\n 'control_random_samples':int(config.get('control_random_samples',1)),"
        + "\n 'control_random_weight':float(config.get('control_random_weight',0.0)),"
        + "\n 'control_same_rule_weight':float(config.get('control_same_rule_weight',0.0)),"
        + "\n 'causal_teacher_trajectory_weight':float(config.get('causal_teacher_trajectory_weight',0.0)),"
        + "\n 'early_label_binding_weight':float(config.get('early_label_binding_weight',0.0)),"
        + "\n 'binding_constraints_inherited_from_promotion':variant.startswith('same_question_'),",
    )
    text = replace_exact(
        text,
        '      || "$FINALFIT_VARIANT" == diagnosis_first_matched_cp3000_semantic_seed3066 ]]; then\n  python - "$PROMOTION_GATE" "$INITIAL_ADAPTER_SHA256" "$INITIAL_RECURRENT_STATE_SHA256"',
        '      || "$FINALFIT_VARIANT" == diagnosis_first_matched_cp3000_semantic_seed3066 \\\n      || "$FINALFIT_VARIANT" == same_question_* ]]; then\n  python - "$PROMOTION_GATE" "$INITIAL_ADAPTER_SHA256" "$INITIAL_RECURRENT_STATE_SHA256"',
    )
    text = replace_exact(
        text,
        'if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_seed67 \\\n'
        '      || "$FINALFIT_VARIANT" == diagnosis_first_matched_cp3000_semantic_seed3066 ]]; then\n'
        '  readarray -t matched_full_teacher_hashes',
        'if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_seed67 \\\n'
        '      || "$FINALFIT_VARIANT" == diagnosis_first_matched_cp3000_semantic_seed3066 \\\n'
        '      || "$FINALFIT_VARIANT" == same_question_* ]]; then\n'
        '  readarray -t matched_full_teacher_hashes',
    )
    immutable_marker = " if variant=='diagnosis_first_matched_cp3000_semantic_seed3066':\n  assert immutable['train_entrypoint']=='scripts/train_state8_diagnosis_first_causal_repair.py',immutable"
    text = replace_exact(
        text,
        immutable_marker,
        immutable_marker
        + "\n if variant.startswith('same_question_'):\n  assert immutable['train_entrypoint']=='scripts/train_state8_diagnosis_first_source_anchored_repair.py',immutable",
    )

    # A selected curriculum checkpoint is already a diagnosis-first State8
    # artifact, not an old repair publish.  Normalize its immutable snapshot
    # into the layout expected by the exact-epoch launcher without changing
    # weights or claiming that a prefix saw all 5,000 curriculum rows.
    normalization_anchor = '''if [[ "$FINALFIT_VARIANT" == diagnosis_first_matched_taskanchor_seed2067 ]]; then
  source_root="$RESULT_ROOT"'''
    checkpoint_normalization = '''if [[ "$FINALFIT_VARIANT" == same_question_repair_checkpoint_promoted ]]; then
  source_root="$RESULT_ROOT"
  export RESULT_ROOT=/tmp/state8_repair_checkpoint_promoted_initialization
  rm -rf "$RESULT_ROOT"; mkdir -p "$RESULT_ROOT/model" "$RESULT_ROOT/metrics"
  ln -s "$source_root/adapter/adapter_model.safetensors" "$RESULT_ROOT/model/adapter_model.safetensors"
  ln -s "$source_root/adapter/adapter_config.json" "$RESULT_ROOT/model/adapter_config.json"
  ln -s "$source_root/recurrent_state.pt" "$RESULT_ROOT/model/recurrent_state.pt"
  python - "$PROMOTION_GATE" "$source_root/checkpoint_completion.json" "$RESULT_ROOT" <<'PY_REPAIR_CHECKPOINT_NORMALIZE'
import json,pathlib,sys
gate_path,source_path,root=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3])
gate=json.load(open(gate_path)); selected=gate.get('selected') or {}; source=json.load(open(source_path))
step=int(selected['checkpoint_step']); seed=int(selected['repair_seed']); branch=str(selected['repair_branch'])
assert gate['stage']=='state8_eedi_same_question_joint_promotion_v1' and gate['pass'] is True,gate
assert selected['finalfit_variant']=='same_question_repair_checkpoint_promoted' and selected['qualified'] is True,selected
assert selected['artifact_layout']=='repair_checkpoint_snapshot',selected
source_step=int(selected['source_checkpoint_step'])
allowed={(5201,'semantic_binding'),(5202,'specificity_triplet'),(5203,'balanced_continuation'),
 (6201,'semantic_binding'),(6202,'specificity_triplet'),(6203,'balanced_continuation'),
 (8201,'causal_teacher_low'),(8202,'causal_teacher_high'),
 (8301,'early_label_low'),(8302,'early_label_high')}
assert (seed,branch) in allowed,selected
assert source_step==(4000 if seed//1000==5 else 3000),selected
assert step in {20,100,240,500},selected
assert source=={'status':'complete','step':step,'sha256':selected['artifact_sha256']},(source,selected)
completion={'status':'complete','diagnostic':False,'micro_steps':step,
 'real_examples_seen_once':False,'selected_prefix_steps':step,
 'source_history_mode':'five_stage_curriculum_then_strict_state_only','sha256':source['sha256']}
publish={'stage':'state8_repair_checkpoint_promoted_v1','status':'complete',
 'repair_arm':'trajectory01','repair_branch':branch,'repair_seed':seed,
 'source_checkpoint_step':source_step,'selected_prefix_steps':step,
 'state_shape':[64,8,5120],'raw_history_visible_to_decoder':False,
 'history_dropout':1.0,'online_teacher_kd':True,
 'train_entrypoint':'scripts/train_state8_diagnosis_first_source_anchored_repair.py',
 'diagnosis_field_order':['diagnosis','evidence','label'],
 'canonical_label_role':'auxiliary_ordinary_weight',
 'source_output_schema':'label_diagnosis_evidence_v1',
 'promoted_output_schema':'diagnosis_evidence_label_v1',
 'metadata_migration':'field_order_and_targets_audited_diagnosis_first_v1',
 'artifact_sha256':source['sha256']}
(root/'metrics/completion.json').write_text(json.dumps(completion,indent=2,sort_keys=True)+'\\n')
(root/'metrics/publish_receipt.json').write_text(json.dumps(publish,indent=2,sort_keys=True)+'\\n')
PY_REPAIR_CHECKPOINT_NORMALIZE
fi
'''
    normalization = checkpoint_normalization + '''if [[ "$FINALFIT_VARIANT" == same_question_curriculum_promoted ]]; then
  source_root="$RESULT_ROOT"
  export RESULT_ROOT=/tmp/state8_curriculum_promoted_initialization
  rm -rf "$RESULT_ROOT"; mkdir -p "$RESULT_ROOT/model" "$RESULT_ROOT/metrics"
  ln -s "$source_root/adapter/adapter_model.safetensors" "$RESULT_ROOT/model/adapter_model.safetensors"
  ln -s "$source_root/adapter/adapter_config.json" "$RESULT_ROOT/model/adapter_config.json"
  ln -s "$source_root/recurrent_state.pt" "$RESULT_ROOT/model/recurrent_state.pt"
  python - "$PROMOTION_GATE" "$source_root/checkpoint_completion.json" "$RESULT_ROOT" <<'PY_CURRICULUM_NORMALIZE'
import json,pathlib,sys
gate_path,source_path,root=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3])
gate=json.load(open(gate_path)); selected=gate.get('selected') or {}; source=json.load(open(source_path))
step=int(selected['checkpoint_step']); arm=str(selected['curriculum_arm'])
assert gate['stage']=='state8_eedi_same_question_joint_promotion_v1' and gate['pass'] is True,gate
assert selected['finalfit_variant']=='same_question_curriculum_promoted' and selected['qualified'] is True,selected
assert selected['source_history_mode']=='five_stage_curriculum',selected
assert arm in {'generic','current_query'} and step in {2000,3000,4000,5000},selected
assert source=={'status':'complete','step':step,'sha256':selected['artifact_sha256']},(source,selected)
completion={'status':'complete','diagnostic':False,'micro_steps':step,
 'real_examples_seen_once':step==5000,'selected_prefix_steps':step,
 'source_history_mode':'five_stage_curriculum','sha256':source['sha256']}
immutable={'stage':'state8_curriculum_checkpoint_promoted_v1','status':'complete',
 'replay_variant':'t3_heavy','source_curriculum_arm':arm,'source_checkpoint_step':step,
 'source_history_mode':'five_stage_curriculum','history_dropout_schedule':[0.0,0.25,0.5,0.75,1.0],
 'state_shape':[64,8,5120],'raw_history_visible_to_decoder':False,
 'persistent_history_kv_cache':False,'semantic_judge_authoritative':True,
 'exact_label_metric_authoritative':False,'sha256':source['sha256']}
(root/'metrics/completion.json').write_text(json.dumps(completion,indent=2,sort_keys=True)+'\\n')
(root/'metrics/immutable_receipt.json').write_text(json.dumps(immutable,indent=2,sort_keys=True)+'\\n')
PY_CURRICULUM_NORMALIZE
fi
'''
    text = replace_exact(
        text,
        normalization_anchor,
        normalization + normalization_anchor,
    )

    # The curriculum initialization has its own transparent prefix receipt.
    receipt_assertion = "assert receipt['status']=='complete' and receipt['diagnostic'] is False,receipt\nexpected_steps=1000 if history_mode=='strict_state_only' else 5000\nassert receipt['micro_steps']==expected_steps and receipt['real_examples_seen_once'] is True,receipt"
    receipt_replacement = """assert receipt['status']=='complete' and receipt['diagnostic'] is False,receipt
if variant=='same_question_curriculum_promoted':
 expected_steps=int(immutable['source_checkpoint_step'])
 assert history_mode=='five_stage_curriculum',history_mode
 assert receipt['micro_steps']==expected_steps and receipt['selected_prefix_steps']==expected_steps,receipt
 assert receipt['real_examples_seen_once'] is (expected_steps==5000),receipt
elif variant=='same_question_repair_checkpoint_promoted':
 expected_steps=int(immutable['selected_prefix_steps'])
 assert history_mode=='strict_state_only',history_mode
 assert expected_steps in {20,100,240,500} and receipt['micro_steps']==expected_steps,receipt
 assert receipt['selected_prefix_steps']==expected_steps,receipt
 assert receipt['real_examples_seen_once'] is False,receipt
else:
 expected_steps=1000 if history_mode=='strict_state_only' else 5000
 assert receipt['micro_steps']==expected_steps and receipt['real_examples_seen_once'] is True,receipt"""
    text = replace_exact(text, receipt_assertion, receipt_replacement)

    curriculum_immutable_anchor = "elif variant=='diagnosis_first_matched_seed67':\n assert immutable['stage']==expected_stage and immutable['status']=='complete',immutable"
    curriculum_immutable = """elif variant=='same_question_repair_checkpoint_promoted':
 assert immutable['stage']==expected_stage and immutable['status']=='complete',immutable
 seed=int(immutable['repair_seed']); source_step=int(immutable['source_checkpoint_step'])
 assert seed in {5201,5202,5203,6201,6202,6203,8201,8202,8301,8302},immutable
 assert source_step==(4000 if seed//1000==5 else 3000),immutable
 assert immutable['selected_prefix_steps'] in {20,100,240,500},immutable
 assert immutable['diagnosis_field_order']==['diagnosis','evidence','label'],immutable
 assert immutable['canonical_label_role']=='auxiliary_ordinary_weight',immutable
 assert immutable['source_output_schema']=='label_diagnosis_evidence_v1',immutable
 assert immutable['promoted_output_schema']=='diagnosis_evidence_label_v1',immutable
 assert immutable['metadata_migration']=='field_order_and_targets_audited_diagnosis_first_v1',immutable
 assert immutable['artifact_sha256']==observed and immutable['state_shape']==[64,8,5120],immutable
 assert immutable['raw_history_visible_to_decoder'] is False,immutable
 assert immutable['history_dropout']==1.0 and immutable['online_teacher_kd'] is True,immutable
elif variant=='same_question_curriculum_promoted':
 assert immutable['stage']==expected_stage and immutable['status']=='complete',immutable
 assert immutable['source_history_mode']=='five_stage_curriculum',immutable
 assert immutable['history_dropout_schedule']==[0.0,0.25,0.5,0.75,1.0],immutable
 assert immutable['sha256']==observed and immutable['state_shape']==[64,8,5120],immutable
 assert immutable['raw_history_visible_to_decoder'] is False,immutable
 assert immutable['persistent_history_kv_cache'] is False,immutable
 assert immutable['semantic_judge_authoritative'] is True,immutable
 assert immutable['exact_label_metric_authoritative'] is False,immutable
elif variant=='diagnosis_first_matched_seed67':
 assert immutable['stage']==expected_stage and immutable['status']=='complete',immutable"""
    text = replace_exact(text, curriculum_immutable_anchor, curriculum_immutable)

    # The initial state retains the five-stage curriculum provenance while the
    # final exact epoch remains strictly state-only.
    state_config_anchor = "elif variant=='diagnosis_first_matched_seed67':\n assert config['teacher_prompt_provenance_stage']=='state8_diagnosis_first_teacher_provenance_gate_v1',config"
    state_config_replacement = """elif variant=='same_question_curriculum_promoted':
 assert config['history_dropout_schedule']==[0.0,0.25,0.5,0.75,1.0],config
 assert config['teacher_prompt_provenance_stage']=='state8_diagnosis_first_teacher_provenance_gate_v1',config
 assert config['diagnosis_field_order']==['diagnosis','evidence','label'],config
 assert config['canonical_label_role']=='auxiliary_ordinary_weight',config
 assert config['external_state_teacher_sha256']==teacher_hash,config
elif variant=='diagnosis_first_matched_seed67':
 assert config['teacher_prompt_provenance_stage']=='state8_diagnosis_first_teacher_provenance_gate_v1',config"""
    text = replace_exact(text, state_config_anchor, state_config_replacement)

    # All same-question exact fits use the configured consistency coefficient.
    text = text.replace(
        "if variant in {'diagnosis_first_matched_seed67','diagnosis_first_matched_cp3000_semantic_seed3066','same_question_updater_mse1_seed3072','same_question_full_mse1_seed3071','same_question_repair_promoted','same_question_repair_checkpoint_promoted','same_question_curriculum_promoted'}: expected_trajectory=0.1",
        "if variant in {'diagnosis_first_matched_seed67','diagnosis_first_matched_cp3000_semantic_seed3066'}: expected_trajectory=0.1\n if variant.startswith('same_question_'): expected_trajectory=1.0",
    )

    # Merge the exact same Eedi causal twins into the full-epoch causal stream.
    data_anchor = '''test "$(sha256sum "$CAUSAL_DATA/train.jsonl" | cut -d' ' -f1)" = 1c8916da482c316dee92982aa3afaa0d2a123f1574140789b14ba5e1e92aee95
'''
    data_insert = data_anchor + '''if [[ "$FINALFIT_VARIANT" == same_question_* ]]; then
  export SAME_QUESTION_CAUSAL=/tmp/state8_finalfit_eedi_same_question
  export SAME_QUESTION_MERGED=/tmp/state8_finalfit_answer_diagnosis_same_question
  rm -rf "$SAME_QUESTION_CAUSAL" "$SAME_QUESTION_MERGED"
  mkdir -p "$SAME_QUESTION_CAUSAL" "$SAME_QUESTION_MERGED"
  python scripts/build_state8_eedi_same_question_causal_twins.py \
    --source-jsonl "$FINAL_JSONL" --output-jsonl "$SAME_QUESTION_CAUSAL/train.jsonl" \
    --receipt "$SAME_QUESTION_CAUSAL/train.receipt.json" --twins 100 --seed 20260811 --offset 0
  python scripts/build_state8_eedi_same_question_causal_twins.py \
    --source-jsonl "$FINAL_JSONL" --output-jsonl "$SAME_QUESTION_CAUSAL/valid.jsonl" \
    --receipt "$SAME_QUESTION_CAUSAL/valid.receipt.json" --twins 25 --seed 20260811 --offset 100 \
    --exclude-jsonl "$SAME_QUESTION_CAUSAL/train.jsonl"
  python scripts/merge_state8_causal_training_rows.py \
    --input "$CAUSAL_DATA/train.jsonl" --input "$SAME_QUESTION_CAUSAL/train.jsonl" \
    --output-jsonl "$SAME_QUESTION_MERGED/train.jsonl" --receipt "$SAME_QUESTION_MERGED/train.receipt.json"
  python scripts/merge_state8_causal_training_rows.py \
    --input "$CAUSAL_DATA/valid.jsonl" --input "$SAME_QUESTION_CAUSAL/valid.jsonl" \
    --output-jsonl "$SAME_QUESTION_MERGED/valid.jsonl" --receipt "$SAME_QUESTION_MERGED/valid.receipt.json"
  export FINAL_CAUSAL_TRAIN_JSONL="$SAME_QUESTION_MERGED/train.jsonl"
  export FINAL_CAUSAL_VALID_JSONL="$SAME_QUESTION_MERGED/valid.jsonl"
fi
'''
    text = replace_exact(text, data_anchor, data_insert)
    text = replace_exact(
        text,
        "causal_counts=={\n 'diagnosis_natural_schema':2102,\n 'prospective_answer':2102,\n},causal_counts",
        "causal_counts==({\n 'diagnosis_natural_schema':2302,\n 'prospective_answer':2102,\n} if variant.startswith('same_question_') else {\n 'diagnosis_natural_schema':2102,\n 'prospective_answer':2102,\n}),causal_counts",
    )

    # Persist all same-question receipts into the final and canonical payload.
    semantic_marker = (
        'if [[ -f "$RUN_DIR/diagnosis_first_matched_finalfit.json" ]]; then\n'
        '  cp "$RUN_DIR/diagnosis_first_matched_finalfit.json" \\\n'
        '    "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_matched_finalfit.json"\n'
        'fi\n'
    )
    text = replace_exact(
        text,
        semantic_marker,
        semantic_marker
        + '''if [[ "$FINALFIT_VARIANT" == same_question_* ]]; then
  test -f "$RUN_DIR/diagnosis_first_source_anchored_repair.json"
  test -f "$RUN_DIR/eedi_same_question_finalfit.json"
  cp "$RUN_DIR/diagnosis_first_source_anchored_repair.json" "$AMLT_OUTPUT_DIR/metrics/diagnosis_first_source_anchored_repair.json"
  cp "$RUN_DIR/eedi_same_question_finalfit.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_finalfit.json"
  cp "$SAME_QUESTION_CAUSAL/train.receipt.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_train.json"
  cp "$SAME_QUESTION_CAUSAL/valid.receipt.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_valid.json"
  cp "$SAME_QUESTION_MERGED/train.receipt.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_merged_train.json"
  cp "$SAME_QUESTION_MERGED/valid.receipt.json" "$AMLT_OUTPUT_DIR/metrics/eedi_same_question_merged_valid.json"
fi
''',
    )

    # World2/global-batch-16 topology, preserving the exact epoch/order.
    text = replace_exact(
        text,
        'export TOTAL_ROUNDS=16241\nexport MILESTONE_ROUNDS="3249 6497 9745 12993"',
        'export TOTAL_ROUNDS=129923\nexport TOTAL_OPTIMIZER_STEPS=16241\nexport MILESTONE_ROUNDS="25992 51976 77960 103944"',
    )
    text = replace_exact(text, 'test "$WORLD_SIZE" = 16', 'test "$WORLD_SIZE" = 2')
    text = replace_exact(text, "--expected-world-size 16", "--expected-world-size 2")
    text = replace_exact(
        text,
        '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"',
        '--expected-planned-optimizer-steps "$TOTAL_OPTIMIZER_STEPS"',
    )
    text = replace_exact(text, "--save-steps 50 --recovery-save-steps 25)", "--save-steps 400 --recovery-save-steps 200)")
    text = replace_exact(text, "--normalize-causal-exposure --gradient-accumulation-steps 1", "--normalize-causal-exposure --gradient-accumulation-steps 8")
    text = replace_exact(text, "--save-steps 50 --checkpoint-retention 2", "--save-steps 400 --checkpoint-retention 2")
    text = replace_exact(text, "--distributed-mpi --distributed-world-size 16 --seed 42", "--distributed-mpi --distributed-world-size 2 --seed 42")
    text = replace_exact(
        text,
        "assert config['distributed_world_size']==16 and config['distributed_rounds']==16241,config",
        "assert config['distributed_world_size']==2 and config['distributed_rounds']==129923,config\n"
        "assert config['gradient_accumulation_steps']==8,config\n"
        "assert config['planned_optimizer_steps']==16241,config\n"
        "assert config['distributed_world_size']*config['gradient_accumulation_steps']==16,config\n"
        "assert completion['optimizer_steps']==16241,completion",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.source, args.output))


if __name__ == "__main__":
    main()
