"""Render a complete audited C4 pilot; never infer success from partial output."""
import argparse
import json
from pathlib import Path


def render(audit, manifest):
    assert audit['status']=='complete'
    core=audit['core']; assert core['status']=='complete'
    assert set(audit['models'])=={'baseline','A','B'}
    for model in audit['models'].values():
        assert model['integrity']['records']==960
        assert model['integrity']['groups']==32
        assert model['integrity']['token_text_checked']
    for arm in ('A','B'):assert audit['training'][arm]['unique_steps']==240
    def effect(value):
        lo,hi=value['pair_cluster_95ci']
        return f"{value['estimate']:.4f} [{lo:.4f}, {hi:.4f}]"
    def ratio(hits,n):return f'{hits}/{n} ({hits/n:.1%})'
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    counts={label:core['arms'][label]['counts'] for label in ('baseline','A','B')}
    lines=['# C4 candidate-free execution training: audited pilot results',
        '\n## 1. Question and non-goals',
        'Does candidate-free SFT improve one-shot student simulation, and does differentiable '
        'state-exchange supervision improve donor-specific free generation beyond matched SFT? '
        'One exploratory training seed (42), six known arithmetic rules, final checkpoint only. '
        'This is an explicitly trained causal interface, not evidence that the old model naturally '
        'contained a general executable rule engine. No recurrent or human-student claim.',
        '\n## 2. Sample construction and concrete sample',
        'Programmatically generated subtraction and common-denominator multiplication. '
        'Train: 160 independent unordered history pairs, two directions, three queries, two '
        'instruction modes = 1,920 rows. Validation: 8 pairs, 96 rows. Test: 16 pairs, 32 '
        'directions, three queries = 96 directional queries; ten conditions per query = '
        '960 generations per model, 2,880 across baseline/A/B. Each history has three '
        'observations. Test/validation history rules are identifiable within the six-rule panel. '
        'Two auxiliary training histories have documented ambiguity and are not donor/native sources.',
        'Generator seed 2026090601. Number-multiset/operand signatures are excluded against '
        'old CF1 6,000/600, MalRule 9,000/900/900/900, the prior 81-query numerical panel, '
        'and across new train/valid/test. Each target differs from the other rule and correct '
        'answer, visible-history numeric tokens, and the other two query answers. '
        '5,520 operand-based answer checks passed. This does not prove pretraining novelty '
        'or unseen-rule/template generalization.',
        'Illustrative construction (not a claimed measured success): a donor following '
        'ignore-denominators should answer `2/7 × 3/7` with `6`, not `6/49`. '
        'The source contains its history and state token only, without the current query or target.',
        '\n## 3. Coordinate, loss and exact formula',
        'At the `vision_pad` address after each of 64 decoder blocks, B replaces recipient '
        'residual `h_recipient^l ← s_donor^l`. The donor and recipient are rows in one '
        'differentiable forward pass. Non-reentrant gradient checkpointing preserves source '
        'gradients. A uses donor history with ordinary answer-token SFT; both arms execute '
        'two-row forwards. Only target-row answer/EOS tokens contribute to '
        '`L = −(1/|T|) Σ[t∈T] log pθ(y_t | prefix, y_<t)`. Donor-source/prompt labels are −100. '
        'No probe or neuron ranking is optimized.',
        '\n## 4. Illustrative loss sign',
        'Raising a target-token probability from 0.2 to 0.5 lowers its negative log '
        'likelihood from 1.609 to 0.693. These numbers illustrate the objective, not observed '
        'per-neuron attribution or a measured intervention effect.',
        '\n## 5. Fixed candidate and lineage',
        'There is no neural-coordinate candidate search or ranking. The sole endpoint is '
        'the preselected final step 240 for each arm, warm-started from the historical CF1 '
        '`vision_pad` rank-8 adapter. Initial adapter SHA256: `'+manifest['init_adapter_sha256']+'`.',
        table(['Model','Evaluated adapter SHA256','Prediction SHA256'],
            [(label,core['arms'][label]['fingerprint']['adapter_sha256'],core['arms'][label]['sha256'])
             for label in ('baseline','A','B')]),
        '\n## 6. Selection funnel and training budget',
        'Frozen generated manifest → all rows retained → identical supervised-target order '
        'and 26,524 target/EOS tokens per arm → 240 updates → final checkpoint only. No '
        'test-based seed, prompt, layer, checkpoint, or example selection. First 120 updates '
        'use explicit rules; last 120 use history, with differentiable exchange in B. '
        'QLoRA NF4/bfloat16, rank 8, dropout zero, AdamW lr 2e-5, 12-step warmup then '
        'linear decay, accumulation 8, grad clip 1, weight decay .01, max sequence 768 '
        'with no truncation. Save every 40 updates. The original adapter is unchanged.',
        '\n## 7. Held-out causal validation',
        '`D_model = mean_i[1(output_swap,i = target_donor,i) − '
        '1(output_random,i = target_donor,i)]`. Primary contrast: `D_B − D_A`. '
        'The random donor is a distinct history with recipient rule, applicable to the '
        'query and with a target distinct from the donor. It is not an arbitrary random '
        'vector distribution. Same-donor-rule and own-state no-op controls are separate.',
        'Bootstrap: 10,000 resamples of the 16 unordered history-pair IDs, keeping both '
        'directions and all three queries together. Intervals are exploratory 95% '
        'percentile intervals, unadjusted; no retrospective replication gate.',
        '\n## 8. Generation-time intervention',
        'Full residual replacement at all 64 state-token addresses, dose 1, on the '
        'prefill pass only. A donor trajectory is captured once per history and reused '
        'for different queries. Fresh KV per generation. Post-state positions cannot '
        'read history; query tokens cannot read state; answer anchor/generated tokens '
        'can read state. Baseline is freshly evaluated with this same corrected mask. '
        'No history-free or final-layer-versus-trajectory comparison is part of this pilot.',
        '\n## 9. Evaluation protocol',
        'Greedy decoding, 48-token limit, stop at EOS or closing predicted-response tag. '
        'Strict single-number/tag parsing, exact rational equality. Invalid, multiple-answer '
        'or truncated outputs count as misses. No LLM judge. Primary open conditions '
        'provide no current correct answer, menu or rule description. The explicit-rule '
        'control and candidate-visible controls are separately labelled; menu controls '
        'must not be interpreted as unsupported free generation.',
        '\n## 10. Aggregate and paired results',
        table(['Model','Directed donor hit /96','Wrong-rule donor hit /96','D [95% pair-cluster CI]'],
            [(label,ratio(counts[label]['swap']['donor_hits'],96),ratio(counts[label]['random']['donor_hits'],96),
              effect(core['arms'][label]['contrasts']['individual'])) for label in ('baseline','A','B')]),
        '\nPrimary **B − A specificity**: '+effect(core['B_minus_A_specificity'])+'.',
        '\nSecondary paired specificity changes against the freshly evaluated baseline: '+
        '; '.join(label+' = '+effect(core[key]) for label,key in
            (('A − baseline','A_minus_baseline_specificity'),
             ('B − baseline','B_minus_baseline_specificity')))+'. '
        'These are unadjusted exploratory secondary intervals, not replication gates.',
        '\nAll conditions follow (native accuracy means recipient hits, all other accuracy columns '
        'below are explicitly donor or recipient hits as labelled).',
        table(['Model','Condition','Donor hits','Recipient hits','Invalid'],
            [(label,c,ratio(v['donor_hits'],v['n']),ratio(v['recipient_hits'],v['n']),v['invalid'])
             for label in ('baseline','A','B') for c,v in counts[label].items()]),
        '\nThree-query agreement (32 directions, but only 16 independent pairs):',
        table(['Model','Condition','All-three donor matches /32'],
            [(label,c,ratio(v['all3_donor_hits'],v['n_groups']))
             for label in ('baseline','A','B') for c,v in audit['models'][label]['all_three_by_condition'].items()]),
        '\nValidation losses (teacher-forced; not generation accuracy):',
        table(['Arm','Checkpoint','Explicit-rule','History'],
            [(arm,step,f"{v['oracle']['loss']:.6f}",f"{v['history']['loss']:.6f}")
             for arm in ('A','B') for step,v in audit['training'][arm]['validation'].items()]),
        '\n## 11. Failures, corruption and representative examples',
        table(['Model','No-op token/stop mismatches /96'],
            [(label,len(audit['models'][label]['noop_token_mismatches'])) for label in ('baseline','A','B')]),
        'No-op mismatches, invalid generations and per-rule failures remain in the data. '
        'These examples are deterministic first qualifying groups in manifest order, not '
        'an estimate of prevalence or an example-filtered metric. An empty success list '
        'means no group had all three directed hits and zero random hits.',
        '```json\n'+json.dumps(audit['examples'],indent=2,ensure_ascii=False)+'\n```',
        '\nPer-donor-rule results:',
        table(['Model','Donor rule','Condition','Donor hits','Invalid'],
            [(label,rule,c,ratio(v['donor_hits'],v['n']),v['invalid'])
             for label in ('baseline','A','B') for rule,cs in audit['models'][label]['per_donor_rule'].items()
             for c,v in cs.items() if c in ('swap','random','donor_native','oracle')]),
        '\nTraining integrity and replay counts:',
        table(['Arm','Unique steps','Logged updates','Replayed steps'],
            [(arm,t['unique_steps'],t['logged_updates'],json.dumps(t['replayed_steps']))
             for arm,t in audit['training'].items()]),
        '\n## 12. Interpretation and limits']
    delta=core['B_minus_A_specificity'];lo,hi=delta['pair_cluster_95ci']
    mismatch_total=sum(len(m['noop_token_mismatches']) for m in audit['models'].values())
    if mismatch_total:
        lines.append('**Causal-control caveat:** own-state replacement differs from native '
            f'output on {mismatch_total} model-query comparisons in total. The pipeline '
            'is not exactly no-op invariant. Do not claim zero intervention artifact; '
            'quantization/numerical sensitivity versus implementation mismatch remains '
            'to be distinguished. All such cases are retained in primary metrics.')
    if lo>0:
        lines.append('On this frozen six-rule pilot, B has a positive paired advantage over '
            'matched A with an exploratory interval above zero. This supports an incremental '
            'benefit of this specific exchange-training recipe, not a general rule engine.')
    elif hi<0:
        lines.append('On this frozen pilot, the paired contrast favors A over B. The exchange '
            'recipe does not improve specificity over matched candidate-free SFT here.')
    else:
        lines.append('The exploratory interval for B − A includes zero. This pilot does not '
            'establish an incremental specificity benefit of exchange supervision over matched SFT.')
    lines += ['Even a positive D is limited to the known rule panel, fresh numbers and '
        'histories, this checkpoint and decoder. Native/oracle/joint-three performance '
        'and no-op validity must be considered alongside it. This does not establish '
        'unseen-rule compositional execution, recurrent self-updating, or human fidelity. '
        'A positive result is repair-training evidence, not retroactive proof about the old model.',
        '\n## 13. Reproduction and provenance',
        'Experiment `c4-execution-training-v1-seed42-recovery1-20260906`; A job '
        '`stoic_queen_l2zrtkcpcq`, B `tidy_card_gdqp59xzfl`; Feeds-LLM Standard, '
        '1×80G1-A100 each, UAI rankfun_aml, original primary AML workspace. '
        'Requested image alias and actual resolved image differ; see recovery receipt '
        'for both identities. Original startup failure and A platform cancellation/automatic '
        'retry are retained in `reports/c4_execution_training_recovery1_20260906.md`.',
        'Inputs/protocol: `manifests/c4_execution_training_v1_20260906.json`, '
        '`reports/c4_execution_training_protocol_20260906.md`; frozen code hashes: '
        '`manifests/c4_execution_training_v1_recovery1_20260906.sha256`. '
        'Collection: `scripts/collect_c4_execution_training.py`; full audit: '
        '`scripts/audit_c4_execution_delivery.py --tokenizer /home/v-yuejiali/studentmodeling/models/Qwen3-32B`; '
        'report: `scripts/render_c4_execution_results.py`. Full audit requires all 2,880 '
        'unique records and raw token/text equality. Checkpoints remain in remote storage.']
    return '\n\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('audit','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.write_text(render(json.loads(a.audit.read_text()),json.loads(a.manifest.read_text())))
