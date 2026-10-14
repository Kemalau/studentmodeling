#!/usr/bin/env python3
"""Final completion audit for the layerwise unified student world model."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any

from summarize_layerwise_unified_final_eval import (
    FROZEN_GROUP_SUPPORT,
    FROZEN_TOTAL_ROWS,
)
from validate_t3_semantic_result import (
    EXPECTED_SUPPORT as T3_EXPECTED_SUPPORT,
    MIN_RETENTION_RATIO,
)

EXPECTED_REAL_EXAMPLES = 259845
EXPECTED_OPTIMIZER_STEPS = 16241
EXPECTED_CAUSAL_EXPOSURES = 12992
EXPECTED_UNIQUE_CAUSAL_PAIRS = 1051
EXPECTED_WORLD_SIZE = 16
EXPECTED_DATA_COUNTS = {
    'rows': 259845,
    'source/dbe_kt22': 10941,
    'source/eedi_mining': 3476,
    'source/eedi_response': 33440,
    'source/foundationalassist': 147191,
    'source/malrule': 9900,
    'source/xes3g5m': 54897,
    'source_split/train': 244331,
    'source_split/valid': 15514,
    'task/diagnose_misconception': 6776,
    'task/predict_correctness': 110525,
    'task/predict_error_persistence': 8375,
    'task/predict_prospective_error': 78600,
    'task/predict_response': 55569,
}
EXPECTED_TRAIN_SHA256 = '2a2f650bb077196a79deaa7eac363d5224fd5936e9c51e60d3fd5f27775282c2'
EXPECTED_VALID_SHA256 = '78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14'
EXPECTED_FINAL_CORPUS_SHA256 = '9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd'
EXPECTED_TASKS = {
    'predict_correctness',
    'predict_response',
    'diagnose_misconception',
    'predict_prospective_error',
    'predict_error_persistence',
}
EXPECTED_EVAL_CHECKS = {
    'C4_architecture',
    'C4_closed',
    'C4_support',
    'C4_specificity',
    'C4_noop',
    'C4_same_rule',
    'C4_replication',
    'all_tasks_present',
    'all_protocol_groups_present',
    'all_protocol_support_exact',
    'full_formal_row_support',
    'task_performance_retained',
}

def optimizer_plan_is_exact(config:dict[str,Any])->bool:
    planned=config.get('planned_optimizer_steps')
    if planned==EXPECTED_OPTIMIZER_STEPS:return True
    source=config.get('promoted_real_order_source_round')
    return (
        isinstance(source,int) and not isinstance(source,bool) and source==750
        and config.get('promoted_objective_transition_source_round')==source
        and config.get('distributed_rounds')==EXPECTED_OPTIMIZER_STEPS
        and bool(config.get('resume_from_checkpoint'))
        and config.get('diagnostic_weight_only_warm_start') in (None,False)
        and config.get('validation_only_from_distributed_checkpoint') in (None,False)
        and planned==EXPECTED_OPTIMIZER_STEPS-source
    )
def read(path:Path)->dict[str,Any]:
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict):raise ValueError(path)
    return x
def one(root:Path,name:str)->Path:
    x=list(root.rglob(name))
    if len(x)!=1:raise ValueError(f'Expected one {name}: {x}')
    return x[0]
def validate(root:Path)->dict[str,Any]:
    markers=['final_fit/complete','final_fit/scientific_gate_pass','final_eval/complete','final_eval/scientific_gate_pass','t3_semantic_judge/complete','t3_semantic_judge/scientific_gate_pass']
    missing=[x for x in markers if not (root/x).is_file()]
    if missing:raise ValueError(f'Missing completion markers: {missing}')
    fit=read(one(root/'final_fit/download','final_fit_receipt.json'))
    completion=read(one(root/'final_fit/download','completion.json'))
    config=read(one(root/'final_fit/download','run_config.json'))
    data_audit=read(one(root/'final_fit/download','data_audit.json'))
    final_manifest=read(one(root/'final_fit/download','final_fit_manifest.json'))
    evaluation=read(root/'final_eval/summary.json')
    semantic=read(root/'t3_semantic_judge/t3_semantic_gate.json')
    artifacts=read(root/'final_fit/artifact_manifest.json')
    eval_checks=evaluation.get('checks') or {}
    eval_groups=evaluation.get('metrics_by_group') or {}
    performance=evaluation.get('performance_retention') or {}
    c4=evaluation.get('C4') or {}
    c4_pair_seeds=c4.get('pair_seeds') or []
    task_rows=evaluation.get('task_rows') or []
    semantic_accuracy=float(semantic.get('semantic_accuracy',float('nan')))
    semantic_ratio=float(semantic.get('retention_ratio',float('nan')))
    semantic_rubric=str(semantic.get('judge_rubric_sha256') or '')
    checks={
        'final_fit_receipt':fit.get('status')=='final_fit_complete' and fit.get('stage')=='layerwise_unified_final_fit',
        'layerwise_state':fit.get('state_architecture')=='layerwise' and fit.get('state_shape')==[64,5120],
        'closed_state':fit.get('raw_history_visible_to_decoder') is False and fit.get('persistent_history_kv_cache') is False,
        'exact_epoch':(
            fit.get('real_examples')==EXPECTED_REAL_EXAMPLES
            and fit.get('one_epoch_exact') is True
            and fit.get('optimizer_steps')==EXPECTED_OPTIMIZER_STEPS
            and completion.get('status')=='complete'
            and completion.get('diagnostic') is False
            and completion.get('micro_steps')==EXPECTED_REAL_EXAMPLES
            and completion.get('optimizer_steps')==EXPECTED_OPTIMIZER_STEPS
            and completion.get('real_examples_seen')==EXPECTED_REAL_EXAMPLES
            and completion.get('real_examples_seen_once') is True
            and completion.get('history_rows')==EXPECTED_REAL_EXAMPLES
            and completion.get('history_unique_global_positions')==EXPECTED_REAL_EXAMPLES
            and completion.get('history_global_positions_contiguous') is True
            and completion.get('history_unique_examples')==EXPECTED_REAL_EXAMPLES
            and completion.get('history_integrity_pass') is True
            and completion.get('exact_history_support') is True
        ),
        'frozen_training_recipe':(
            config.get('state_architecture')=='layerwise'
            and config.get('layerwise_operation_checkpointing') is True
            and config.get('layerwise_transition_rank')==256
            and config.get('distributed') is True
            and config.get('distributed_world_size')==EXPECTED_WORLD_SIZE
            and config.get('real_train_rows')==EXPECTED_REAL_EXAMPLES
            and optimizer_plan_is_exact(config)
            and config.get('planned_causal_exposures')==EXPECTED_CAUSAL_EXPOSURES
            and config.get('causal_train_pairs')==EXPECTED_UNIQUE_CAUSAL_PAIRS
            and config.get('causal_objective_mode')=='specificity_triplet'
            and config.get('causal_repeat_fraction')==0.05
            and config.get('normalize_causal_exposure') is True
            and config.get('state_token_alias')=='<|vision_pad|>'
            and config.get('causal_branch_dropout_mode')=='eval_deterministic'
        ),
        'frozen_data_provenance':(
            data_audit.get('status')=='passed'
            and set(data_audit.get('required_tasks') or [])==EXPECTED_TASKS
            and data_audit.get('train_valid_example_overlap')==0
            and all(value==0 for value in (data_audit.get('train_valid_real_student_overlap') or {}).values())
            and data_audit.get('train',{}).get('rows')==244331
            and data_audit.get('valid',{}).get('rows')==15514
            and data_audit.get('train',{}).get('decoder_history_leaks')==0
            and data_audit.get('valid',{}).get('decoder_history_leaks')==0
            and data_audit.get('train',{}).get('carrier_errors')==0
            and data_audit.get('valid',{}).get('carrier_errors')==0
            and data_audit.get('train',{}).get('sha256')==EXPECTED_TRAIN_SHA256
            and data_audit.get('valid',{}).get('sha256')==EXPECTED_VALID_SHA256
            and final_manifest.get('purpose')=='post_selection_final_fit_train_plus_validation'
            and final_manifest.get('unique_examples')==EXPECTED_REAL_EXAMPLES
            and final_manifest.get('counts')==EXPECTED_DATA_COUNTS
            and final_manifest.get('inputs',{}).get('train',{}).get('sha256')==EXPECTED_TRAIN_SHA256
            and final_manifest.get('inputs',{}).get('valid',{}).get('sha256')==EXPECTED_VALID_SHA256
            and final_manifest.get('output',{}).get('sha256')==EXPECTED_FINAL_CORPUS_SHA256
            and config.get('train_sha256')==EXPECTED_FINAL_CORPUS_SHA256
        ),
        'causal_exposure':(
            completion.get('causal_pairs_seen')==EXPECTED_CAUSAL_EXPOSURES
            and completion.get('unique_causal_pairs_seen')==EXPECTED_UNIQUE_CAUSAL_PAIRS
            and completion.get('causal_exposure_count_matches_plan') is True
            and completion.get('all_causal_pairs_covered') is True
        ),
        'formal_tasks_and_C4':(
            evaluation.get('stage')=='layerwise_unified_final_evaluation'
            and evaluation.get('scientific_gate_pass') is True
            and set(evaluation.get('tasks') or [])==EXPECTED_TASKS
            and set(eval_checks)==EXPECTED_EVAL_CHECKS
            and all(eval_checks.values())
            and set(eval_groups)==set(FROZEN_GROUP_SUPPORT)
            and all(int(eval_groups[name].get('n',-1))==support for name,support in FROZEN_GROUP_SUPPORT.items())
            and sum(int(row.get('sampled_rows',0)) for row in task_rows)==FROZEN_TOTAL_ROWS
            and set(performance)==EXPECTED_TASKS-{'diagnose_misconception'}
            and all(row.get('pass') is True for row in performance.values())
            and [row.get('pair_seed') for row in c4_pair_seeds]==[13,42,97]
            and all(int(row.get('all_pairs_n',0))==150 and int(row.get('clean_n',0))>=50 for row in c4_pair_seeds)
            and int(c4.get('passing_seeds',0))>=2
            and c4.get('replication_gate_pass') is True
            and float(c4.get('mean_D_specific',-1))>=0.15
            and float(c4.get('D_specific',-1))==float(c4.get('mean_D_specific',-2))
            and all(float(row.get('noop',1))<=0.05 for row in c4_pair_seeds)
            and sum(float(row.get('same_rule',1))<=0.10 for row in c4_pair_seeds)>=2
        ),
        'T3_semantic':(
            semantic.get('stage')=='final_T3_semantic_judge'
            and semantic.get('pass') is True
            and semantic.get('support')==sum(T3_EXPECTED_SUPPORT.values())
            and math.isfinite(semantic_accuracy)
            and 0.0<=semantic_accuracy<=1.0
            and math.isfinite(semantic_ratio)
            and semantic_ratio>=MIN_RETENTION_RATIO
            and semantic.get('judge_backend') in {'responses','trapi'}
            and len(semantic_rubric)==64
            and all(char in '0123456789abcdef' for char in semantic_rubric.lower())
        ),
        'artifact_manifest':artifacts.get('status')=='verified' and artifacts.get('algorithm')=='sha256',
    }
    adapter=one(root/'final_fit/download','adapter_model.safetensors');state=one(root/'final_fit/download','recurrent_state.pt')
    def digest(path:Path)->str:
        value=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):value.update(chunk)
        return value.hexdigest()
    expected=artifacts.get('artifacts') or {}
    checks['artifacts_downloaded']=(
        adapter.is_file() and state.is_file()
        and int((expected.get('adapter_model.safetensors') or {}).get('bytes',-1))==adapter.stat().st_size
        and int((expected.get('recurrent_state.pt') or {}).get('bytes',-1))==state.stat().st_size
        and (expected.get('adapter_model.safetensors') or {}).get('sha256')==digest(adapter)
        and (expected.get('recurrent_state.pt') or {}).get('sha256')==digest(state)
    )
    completion_sha=completion.get('sha256') or {}
    checks['artifacts_match_training_receipt']=(
        completion_sha.get('adapter_model')==digest(adapter)
        and completion_sha.get('recurrent_state')==digest(state)
    )
    if not all(checks.values()):raise ValueError(f'Final checks failed: {[k for k,v in checks.items() if not v]}')
    return {'version':1,'status':'complete','objective':'full_unified_layerwise_recurrent_student_model','checks':checks,'artifacts':{'adapter_model':str(adapter),'recurrent_state':str(state)},'gates':{'final_fit':str(root/'final_fit'),'formal_and_C4':str(root/'final_eval/summary.json'),'T3_semantic':str(root/'t3_semantic_judge/t3_semantic_gate.json')}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=validate(a.root);a.output.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x,indent=2,sort_keys=True))
if __name__=='__main__':main()
