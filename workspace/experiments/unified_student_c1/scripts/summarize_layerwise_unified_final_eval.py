#!/usr/bin/env python3
"""Validate and summarize all final layerwise task/C4 artifacts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import re
import statistics
from typing import Any
def read(path:Path)->dict[str,Any]:
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise ValueError(path)
    return x

REFERENCE = {
    'predict_correctness': {'metric': 'macro_f1', 'value': 0.6869980398379085, 'min_ratio': 0.90},
    'predict_error_persistence': {'metric': 'macro_f1', 'value': 0.5959866676584252, 'min_ratio': 0.85},
    'predict_response': {'metric': 'math_equivalence', 'value': 0.5182778229082047, 'min_ratio': 0.90},
    'predict_prospective_error': {'metric': 'joint_math_equivalence', 'value': 0.7667391304347826, 'min_ratio': 0.90},
}

FROZEN_GROUP_SUPPORT = {
    'diagnose_misconception/eedi_mining/open_taxonomy_generation': 894,
    'diagnose_misconception/malrule/synthetic_cross_template_open_generation': 300,
    'diagnose_misconception/malrule/synthetic_same_template_open_generation': 300,
    'predict_correctness/dbe_kt22/temporal_warm_item_binary': 225,
    'predict_correctness/eedi_response/profile_heldout_binary': 984,
    'predict_correctness/foundationalassist/temporal_warm_item_binary': 1882,
    'predict_correctness/malrule/synthetic_cross_template_binary': 300,
    'predict_correctness/malrule/synthetic_same_template_binary': 300,
    'predict_correctness/xes3g5m/temporal_warm_item_binary': 7222,
    'predict_error_persistence/dbe_kt22/strict_exact_recurrence_binary': 12,
    'predict_error_persistence/foundationalassist/strict_exact_recurrence_binary': 964,
    'predict_prospective_error/dbe_kt22/delta3_closed_choice_generation/delta=3': 200,
    'predict_prospective_error/foundationalassist/delta_open_generation/delta=1': 1485,
    'predict_prospective_error/foundationalassist/delta_open_generation/delta=3': 1474,
    'predict_prospective_error/foundationalassist/delta_open_generation/delta=5': 1441,
    'predict_response/dbe_kt22/closed_choice_generation': 223,
    'predict_response/eedi_response/closed_choice_generation': 984,
    'predict_response/foundationalassist/structured_math_open_generation': 1886,
    'predict_response/malrule/synthetic_cross_template_open_generation': 300,
    'predict_response/malrule/synthetic_same_template_open_generation': 300,
}
FROZEN_TOTAL_ROWS = 21676

def weighted_metric(groups:dict[str,Any],task:str,metric:str)->dict[str,Any]:
    rows=[value for name,value in groups.items() if name.split('/',1)[0]==task]
    if not rows or any(metric not in value or int(value.get('n',0))<=0 for value in rows):
        raise ValueError(f'Missing {metric} for {task}')
    support=sum(int(value['n']) for value in rows)
    score=sum(float(value[metric])*int(value['n']) for value in rows)/support
    return {'metric':metric,'score':score,'support':support}

def c4_pair_seed(path:Path,payload:dict[str,Any])->int:
    candidates=(path.as_posix(),str(payload.get('pair_manifest') or ''))
    for candidate in candidates:
        match=re.search(r'(?:c4[_-])?seed(13|42|97)(?:\D|$)',candidate)
        if match:return int(match.group(1))
    raise ValueError(f'Cannot recover frozen C4 pair seed: {path}')

def summarize_c4(path:Path)->dict[str,Any]:
    c4=read(path); summary=c4['summary']; clean=summary['clean']
    checks=summary.get('checks') or {}
    return {
        'pair_seed':c4_pair_seed(path,c4),
        'path':str(path),
        'architecture_pass':c4.get('state_architecture')=='layerwise' and c4.get('state_shape')==[64,5120],
        'closed_pass':c4.get('raw_history_visible_to_decoder') is False and c4.get('persistent_history_kv_cache') is False,
        'all_pairs_n':int(summary.get('all_pairs_n',0)),
        'clean_n':int(summary.get('clean_n',0)),
        'swap':float(clean['swap']['flip_to_A_rate']),
        'random':float(clean['random']['flip_to_A_rate']),
        'same_rule':float(clean['samemal']['flip_to_A_rate']),
        'noop':float(clean['noop']['flip_to_A_rate']),
        'D_specific':float(summary.get('D_specific',-1)),
        'scientific_gate_pass':bool(summary.get('scientific_gate_pass')),
        'checks':checks,
    }

def summarize(root:Path)->dict[str,Any]:
    metrics=sorted(root.rglob('metrics.json'))
    if len(metrics)!=4: raise ValueError(f'Expected four task metric files: {metrics}')
    rows=[]; groups={}; tasks=set()
    for p in metrics:
        x=read(p)
        if x.get('state_architecture')!='layerwise' or x.get('state_shape')!=[64,5120]: raise ValueError(f'Architecture drift: {p}')
        if x.get('raw_history_visible_to_decoder') is not False or x.get('persistent_history_kv_cache') is not False: raise ValueError(f'History leak: {p}')
        rows.append({'path':str(p),'sampled_rows':int(x['sampled_rows']),'metrics_by_group':x['metrics_by_group']})
        overlap=set(groups).intersection(x['metrics_by_group'])
        if overlap: raise ValueError(f'Duplicate groups: {overlap}')
        groups.update(x['metrics_by_group'])
        tasks.update(str(name).split('/',1)[0] for name in x['metrics_by_group'])
    required={'predict_correctness','predict_response','diagnose_misconception','predict_prospective_error','predict_error_persistence'}
    if tasks!=required: raise ValueError(f'Task groups drift: {tasks}')
    gates=sorted(root.rglob('recurrent_causal_gate.json'))
    if len(gates)!=3: raise ValueError(f'Expected three C4 pair-seed gates: {gates}')
    pair_seeds=sorted((summarize_c4(path) for path in gates),key=lambda row:row['pair_seed'])
    if [row['pair_seed'] for row in pair_seeds]!=[13,42,97]:
        raise ValueError(f'C4 pair-seed support drift: {pair_seeds}')
    passing_seeds=sum(row['scientific_gate_pass'] for row in pair_seeds)
    mean_specificity=statistics.fmean(row['D_specific'] for row in pair_seeds)
    replication_gate=passing_seeds>=2 and mean_specificity>=0.15
    performance={}
    for task,reference in REFERENCE.items():
        row=weighted_metric(groups,task,reference['metric'])
        row.update({'reference':reference['value'],'min_ratio':reference['min_ratio']})
        row['ratio']=row['score']/row['reference']
        row['pass']=row['ratio']>=row['min_ratio']
        performance[task]=row
    checks={
        'C4_architecture':all(row['architecture_pass'] for row in pair_seeds),
        'C4_closed':all(row['closed_pass'] for row in pair_seeds),
        'C4_support':all(row['all_pairs_n']==150 and row['clean_n']>=50 for row in pair_seeds),
        'C4_specificity':mean_specificity>=0.15,
        'C4_noop':all(row['noop']<=0.05 for row in pair_seeds),
        'C4_same_rule':sum(row['same_rule']<=0.10 for row in pair_seeds)>=2,
        'C4_replication':replication_gate,
        'all_tasks_present':tasks==required,
        'all_protocol_groups_present':set(groups)==set(FROZEN_GROUP_SUPPORT),
        'all_protocol_support_exact':all(
            name in groups and int(groups[name].get('n',-1))==support
            for name,support in FROZEN_GROUP_SUPPORT.items()
        ),
        'full_formal_row_support':(
            sum(row['sampled_rows'] for row in rows)==FROZEN_TOTAL_ROWS
            and sum(int(value.get('n',0)) for value in groups.values())==FROZEN_TOTAL_ROWS
        ),
        'task_performance_retained':all(row['pass'] for row in performance.values()),
    }
    seed42=next(row for row in pair_seeds if row['pair_seed']==42)
    c4_summary={
        **{key:seed42[key] for key in ('clean_n','swap','random','same_rule','noop')},
        'D_specific':mean_specificity,
        'seed42_D_specific':seed42['D_specific'],
        'pair_seeds':pair_seeds,
        'passing_seeds':passing_seeds,
        'mean_D_specific':mean_specificity,
        'min_D_specific':min(row['D_specific'] for row in pair_seeds),
        'replication_gate_pass':replication_gate,
    }
    return {'stage':'layerwise_unified_final_evaluation','scientific_gate_pass':all(checks.values()),'checks':checks,'tasks':sorted(tasks),'performance_retention':performance,'metrics_by_group':groups,'task_rows':rows,'C4':c4_summary}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=summarize(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x,indent=2,sort_keys=True));raise SystemExit(0 if x['scientific_gate_pass'] else 2)
if __name__=='__main__':main()
