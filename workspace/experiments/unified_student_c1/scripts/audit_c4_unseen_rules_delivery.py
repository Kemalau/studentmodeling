"""Independent delivery checks and complete report; never alters evaluator data."""
import argparse
from collections import Counter
import json
from pathlib import Path

from c4_freegen_protocol import sha
from c4_unseen_rules import CONDITIONS, PAIRS, RULES, prompt, reference
from collect_c4_unseen_rules import collect

ROOT=Path(__file__).resolve().parents[1]
EXP='c4-unseen-rules-v1-seed42-feedsllm-20260906'
MANIFEST=ROOT/'manifests/c4_unseen_rules_v1_20260906.json'
MODEL=Path('/home/v-yuejiali/studentmodeling/models/Qwen3-32B')


def frozen_code_checks(fingerprint, frozen):
    expected={name.split('/',1)[1]:checksum for name,checksum in frozen.items()
              if name.startswith('scripts/') and name.split('/',1)[1] in fingerprint['code']}
    required={'evaluate_c4_unseen_rules.py','c4_unseen_rules.py','c4_execution_model.py',
              'c4_execution_data.py','c4_rule_followup.py','c4_freegen_protocol.py',
              'train_c1_lora.py','state_swap_misconception_smoke.py'}
    assert set(expected)==required and fingerprint['code']==expected


def inspect(manifest=MANIFEST,downloads=None,tokenizer=None):
    if downloads is None:downloads=ROOT/'results/downloads'/EXP
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    roots={arm:downloads/f'c4-unseen-rules-arm-{arm.lower()}-seed42'/f'unseen_{arm}' for arm in ('A','B')}
    result=collect(manifest,roots,tokenizer)
    m=json.loads(manifest.read_text())
    frozen={line.split('  ',1)[1]:line.split('  ',1)[0] for line in
            (ROOT/'manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256').read_text().splitlines()}
    assert sha(manifest)==frozen['manifests/c4_unseen_rules_v1_20260906.json']
    for source in m['source_lineage']:
        assert sha(ROOT/source['path'])==source['sha256'],source['path']
    details={};runtimes={}
    for arm,root in roots.items():
        fp=result['fingerprints'][arm];frozen_code_checks(fp,frozen)
        for name,checksum in fp['base'].items():assert sha(MODEL/name)==checksum
        runtime=json.loads((root.parent/f'runtime_{arm}.json').read_text());runtimes[arm]=runtime
        assert runtime['fingerprint']==fp and runtime['arm']==arm
        records=json.loads((root/'records.json').read_text())
        idx={(r['group_id'],r['query_idx'],r['condition']):r for r in records}
        groups=[];by_rule={};parity={};prompt_checks=0
        for g in m['groups']:
            for q in g['queries']:
                for role,rule in g['rules'].items():assert reference(rule,q)==q['targets'][role]
                for c in CONDITIONS:
                    r=idx[(g['id'],q['query_idx'],c)]
                    ids=tokenizer(prompt(g,q,c).replace('<STUDENT_STATE>','<|vision_pad|>'),add_special_tokens=False)['input_ids']
                    assert r['prompt_tokens']==len(ids)
                    assert len(ids)+m['max_new_tokens']<=m['max_length'];prompt_checks+=1
            row=dict(group_id=g['id'],pair_id=g['pair_id'],category=g['category'],
                     donor_rule=g['rules']['donor'],recipient_rule=g['rules']['recipient'],queries=[])
            for q in g['queries']:
                rs={c:idx[(g['id'],q['query_idx'],c)] for c in CONDITIONS}
                row['queries'].append(dict(problem=q['problem_text'],parity=q['operands'][1]%2,
                    expected_donor=q['targets']['donor'],expected_recipient=q['targets']['recipient'],
                    outputs={c:r['parsed'] for c,r in rs.items()},
                    texts={c:r['text'] for c,r in rs.items()},
                    stop={c:r['stop'] for c,r in rs.items()}))
                rule=g['rules']['donor'];pr='odd' if q['operands'][1]%2 else 'even'
                for scope,key in ((by_rule,rule),(parity,g['category']+':'+pr)):
                    if key not in scope:scope[key]={c:dict(n=0,hits=0,invalid=0) for c in CONDITIONS}
                    for c,r in rs.items():
                        count=scope[key][c];count['n']+=1
                        count['hits']+=int(r['parsed']==q['targets']['donor']);count['invalid']+=int(r['parsed'] is None)
            groups.append(row)
        transitions={}
        for category in PAIRS:
            ctr=Counter()
            for g in m['groups']:
                if g['category']!=category:continue
                for q in g['queries']:
                    key=tuple(int(idx[(g['id'],q['query_idx'],c)]['parsed']==q['targets']['donor']) for c in ('swap','random'))
                    ctr[key]+=1
            transitions[category]={f'swap_{a}_random_{b}':ctr[(a,b)] for a in (0,1) for b in (0,1)}
        details[arm]=dict(prompt_length_checks=prompt_checks,by_rule=by_rule,by_parity=parity,
            paired_query_transitions=transitions,groups=groups,
            output_formats=dict(Counter(r['parse_type'] for r in records)),
            stop_reasons=dict(Counter(r['stop'] for r in records)))
    assert runtimes['A']['versions']==runtimes['B']['versions']
    requirements=dict(line.split('==') for line in (ROOT/'requirements-feeds.txt').read_text().splitlines() if line)
    for name,version in requirements.items():assert runtimes['A']['versions'][name]==version
    assert result['n_records']==1728
    return dict(status='complete',aggregate=result,details=details,runtimes=runtimes,
        audit='Exact record coverage, frozen code/manifest/adapter lineage, old-source hashes, tokenizer/base metadata hashes, prompt token lengths, target reference formulas, all output texts/tokens, runtime package equality.',
        limitation='Base metadata hashes are verified; this is not a full hash audit of every 32B base-weight shard.')


def interval(x):
    lo,hi=x['pair_cluster_95ci']
    return f"{x['estimate']:.3f} [{lo:.3f}, {hi:.3f}]"


def measured_conclusions(result):
    """Describe the frozen endpoints, including negative and artifact evidence."""
    lines=[]
    for arm,cats in result['arms'].items():
        supported=[cat for cat in ('offset','composition','conditional')
                   if cats[cat]['specificity']['pair_cluster_95ci'][0]>0]
        joint=sum(cats[cat]['joint3']['swap']['all3_donor_hits']
                  for cat in ('offset','composition','conditional'))
        groups=sum(cats[cat]['joint3']['swap']['n_groups']
                   for cat in ('offset','composition','conditional'))
        seen=cats['seen_control'];sc=seen['counts']
        lines.append(f"- {arm}: constructed categories with a positive lower specificity CI: "
                     f"{', '.join(supported) if supported else 'none'}. "
                     f"Complete three-query donor agreement across constructed-rule groups: {joint}/{groups}. "
                     f"Seen control: swap {sc['swap']['donor_hits']}/{sc['swap']['n']}, "
                     f"wrong-rule {sc['random']['donor_hits']}/{sc['random']['n']}.")
        for cat in ('offset','composition','conditional'):
            v=cats[cat];c=v['counts'];n=c['swap']['n']
            lines.append(f"- {arm}/{cat}: explicit-rule oracle {c['oracle']['donor_hits']}/{n}; "
                         f"open history {c['open_history']['donor_hits']}/{n}; "
                         f"donor native {c['donor_native']['donor_hits']}/{n}; "
                         f"swap {c['swap']['donor_hits']}/{n}; wrong-rule {c['random']['donor_hits']}/{n}. "
                         f"Specificity {interval(v['specificity'])}.")
        noop=sum(len(v['noop_mismatches']) for v in cats.values())
        denom=sum(v['counts']['native']['n'] for v in cats.values())
        lines.append(f"- {arm}: own-state no-op output mismatches {noop}/{denom}; retained, not excluded.")
    contrasts=result['B_minus_A_by_category']
    if all(v['estimate']==0 for v in contrasts.values()):
        lines.append('A and B have identical measured primary specificity in every category. '
                     'This is no evidence of an exchange-training advantage here; it is not a population equivalence test.')
    if all(v['specificity']['pair_cluster_95ci'][0]<=0
           for cats in result['arms'].values()
           for cat,v in cats.items() if cat!='seen_control'):
        lines.append('This panel does not support reliable unseen-rule state transport. '
                     'Known-rule transport and unseen-rule induction are different claims. '
                     'A strong oracle with weak open-history performance points toward history induction/identification '
                     'as a limitation; a weak oracle leaves execution or instruction-following unresolved. '
                     'Open-history is itself an evaluation-path change, so these contrasts do not uniquely locate a cause.')
    return lines


def render(a):
    r=a['aggregate'];lines=['# C4 unseen-rule evaluation: complete audited results','',
    '## 1. Question and non-goals','',
    'Frozen final A (matched candidate-free SFT) and B (state-exchange training) are tested for induction and transport of constructed rules from three observations. Both are final step 240, seed 42. No training, checkpoint selection, recurrent-state update, spontaneous rule invention or pretraining-novelty claim.','',
    '## 2. Sample construction','',
    'Four categories × four unordered history pairs × two swap directions × three queries = 96 directional queries per condition per checkpoint (48 unique questions). Nine conditions × two checkpoints = 1,728 generations. Six constructed rules and two seen-rule controls are separate. Generator seed 2026090603; denominator range 4–40, three templates, both parities in each history and query triplet. Query combinations are excluded from audited CF1/MalRule sources and previous panels. Current targets are not in the history, current correct answer or any menu. Old histories contain their own correct answers.','',
    'Concrete constructed offset-3 history: 36/40 × 39/40 → 1407; 16/37 × 12/37 → 195; 6/16 × 15/16 → 93. Queries and actual outputs appear below. New means absent as a labelled mechanism from audited fine-tuning sources, not absent from pretraining. Histories identify rules only within the eight declared rules.','',
    '## 3. Coordinate and exact objective','',
    'All 64 decoder-block output residuals at the vision-pad state token. `D = mean_i(1[swap_i = donor_target_i] − 1[random_i = donor_target_i])`; primary comparison is `D_B − D_A`, separately by category. This evaluates donor-rule answer agreement, not human-student fidelity.','',
    '## 4. Arithmetic sign example','',
    'Illustration only: 12/24 swap hits and 3/24 control hits yield D = +0.375. Positive means more donor-consistent answers with the donor trajectory than the wrong-rule real-state control.','',
    '## 5. Fixed checkpoints, not ranked candidates','']
    for arm,fp in r['fingerprints'].items():lines.append(f"- {arm}: `{fp['adapter_sha256']}`; fixed before this test, no result-based selection.")
    lines+=['','## 6. Selection funnel','',
    'Six constructed rules + two known controls → four pairs/category → outcome-free number, answer-collision and finite-rule-identifiability filters → all 32 directional groups evaluated. Original manifest records 331 seen-number rejections, three copy/repeated-answer rejections and 1,920 independent answer checks. No post-generation exclusions.','',
    '## 7. Held-out controls','',
    '`native`: recipient history; `donor_native`: donor history; `noop`: independently captured own state; `swap`: donor state; `random`: independently sampled recipient-rule state (not Gaussian noise); `same_donor`: another donor-rule history; `oracle`: explicit donor rule without history; `open_history`: donor history with ordinary causal attention; `empty_history`: no history/rule.','',
    '## 8. Intervention and dose','',
    '`h_recipient^l ← s_donor^l` after blocks l=0…63 at the state-token address, dose 1, prefill only. One captured history trajectory is reused on three queries; each generation uses fresh KV. Corrected late-fusion mask and answer anchor are used except for open-history. This test is not a history-free bridge or a last-layer ablation.','',
    '## 9. Decoding and scoring','',
    'NF4/bfloat16 with kbit preparation; eager attention; rank-8 frozen adapter, eval mode, dropout off. Greedy decoding, 48-token cap, 768-token input+output bound, no truncation. Single tagged or bare numeric output, exact rational equality. Invalid, multiple-answer and token-limit outputs count as misses. No LLM judge.','',
    '## 10. Aggregate and paired results','',
    'Each row has n=24 directional queries and eight three-query groups; only four independent unordered pairs. CIs: 10,000 pair-cluster bootstraps, seed 2026090604, unadjusted, exploratory. Zero-width CIs on uniform outcomes do not prove zero population uncertainty.','',
    '| Arm | Category | Swap hits | Wrong-rule hits | D [95% CI] | Swap all-three | Random all-three | No-op mismatches |',
    '|---|---|---:|---:|---|---:|---:|---:|']
    for arm,cats in r['arms'].items():
        for cat,v in cats.items():
            c=v['counts'];j=v['joint3']
            lines.append(f"| {arm} | {cat} | {c['swap']['donor_hits']}/24 | {c['random']['donor_hits']}/24 | {interval(v['specificity'])} | {j['swap']['all3_donor_hits']}/8 | {j['random']['all3_donor_hits']}/8 | {len(v['noop_mismatches'])}/24 |")
    lines+=['','| Category | Paired B − A specificity [95% CI] |','|---|---|']
    for cat,v in r['B_minus_A_by_category'].items():lines.append(f'| {cat} | {interval(v)} |')
    lines+=['','### All conditions (donor-target hits / 24; invalid count in parentheses)','',
            '| Arm | Category | '+' | '.join(CONDITIONS)+' |',
            '|---|---|'+ '|'.join(['---:']*len(CONDITIONS))+'|']
    for arm,cats in r['arms'].items():
        for cat,v in cats.items():
            cells=[f"{v['counts'][c]['donor_hits']}/24 ({v['counts'][c]['invalid']})" for c in CONDITIONS]
            lines.append(f'| {arm} | {cat} | '+' | '.join(cells)+' |')
    lines+=['','### Per-rule diagnostic (donor-target hits / 12)','',
        '| Arm | Donor rule | Donor native | Swap | Random | Oracle | Open history |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm,detail in a['details'].items():
        for rule,counts in detail['by_rule'].items():
            cells=[f"{counts[c]['hits']}/{counts[c]['n']}" for c in ('donor_native','swap','random','oracle','open_history')]
            lines.append(f'| {arm} | {rule} | '+' | '.join(cells)+' |')
    lines+=['','## 11. Concrete successes, failures and output corruption','',
            'Examples are the first complete swap success and first non-complete-swap group in manifest order per category/arm; missing kinds are explicitly noted. They are descriptive, not selected evaluation data. All groups, raw text, parity breakdowns and paired query transitions are retained in the audit JSON.','']
    for arm,detail in a['details'].items():
        for cat in PAIRS:
            gs=[g for g in detail['groups'] if g['category']==cat]
            for success in (True,False):
                selected=next((g for g in gs if all(q['outputs']['swap']==q['expected_donor'] for q in g['queries'])==success),None)
                kind='all-three success' if success else 'failure/partial'
                if selected is None:lines.append(f'- {arm}, {cat}: no {kind} group.');continue
                lines+=['',f"### {arm} / {selected['group_id']} ({kind})",'',f"Donor: `{selected['donor_rule']}`; recipient: `{selected['recipient_rule']}`.",'',
                    '| Query | Donor target | Swap | Random | Oracle | Open history |',
                    '|---|---|---|---|---|---|']
                for q in selected['queries']:
                    values=[q['outputs'][c] if q['outputs'][c] is not None else 'INVALID' for c in ('swap','random','oracle','open_history')]
                    lines.append(f"| {q['problem']} | {q['expected_donor']} | "+' | '.join(values)+' |')
        lines+=['',f"{arm} output formats: `{json.dumps(detail['output_formats'])}`; stop reasons: `{json.dumps(detail['stop_reasons'])}`.",'']
        bad=next((dict(group=g['group_id'],query=q['problem'],condition=c,text=q['texts'][c],stop=q['stop'][c])
                  for g in detail['groups'] for q in g['queries'] for c in CONDITIONS if q['outputs'][c] is None),None)
        lines.append('First invalid output: '+('none.' if bad is None else '`'+json.dumps(bad,ensure_ascii=False)+'`'))
    lines+=['','## 12. Mechanism conclusion and limitations','']
    lines+=measured_conclusions(r)
    lines+=['',
    'Interpret each new-rule category separately. Partial conditional-branch agreement is not a transported conditional rule: all-three success across both parities is the relevant stronger diagnostic. Oracle failure implicates execution/instruction following; oracle success with weak open-history implicates induction; stronger open-history than donor-native points to a bottleneck limitation; donor-native versus swap diagnoses transport. These are patterns, not uniquely identified causal explanations.','',
    'Only four independent history pairs per category and one training seed; this is exploratory, with no preregistered replication gate or multiplicity correction. A/B alone does not identify improvement over the old checkpoint on these new rules. The narrow positive claim, if supported, is rule-specific cross-query state transport in this fraction task—not arbitrary program execution, spontaneous invention, recurrent self-updating state or human fidelity. Negative categories remain in the report.','',
    '## 13. Reproduction and completion audit','',a['audit'],a['limitation'],'',
    f'Experiment: `{EXP}`. Both arms require complete group files and merged records; all 1,728 records passed the delivery audit. See `results/c4_unseen_rules_v1_feedsllm_20260906.submission.json` for job IDs, target, quota and submission command.','',
    'Protocol: `reports/c4_unseen_rules_protocol_20260906.md`; manifest: `manifests/c4_unseen_rules_v1_20260906.json`; frozen source list: `manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256`; evaluator and collector are in `scripts/`.','']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    out=inspect()
    args.audit.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    args.report.write_text(render(out))
    print(json.dumps(dict(status=out['status'],n_records=out['aggregate']['n_records'],report=str(args.report))))
