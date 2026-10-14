# C4 rule follow-ups: complete audited results

## 1. Question and non-goals

Frozen seed-42 CF1 vision-pad checkpoint; no training or recurrent update. A tests explicit-rule elicitation without candidates. B tests whether one transported student state answers two questions differently with an identical candidate menu. This is an exploratory extension, not independent replication.

Primary fixed-menu joint directed-minus-random: **+0.0870, 95% cluster CI [-0.0217, +0.2174]**.

## 2. Sample construction

Parent panel: 27 donors, 81 queries, three per donor. Source row counts: CF1 train/valid 6,000/600; original MalRule train/valid/same-test/cross-test 9,000/900/900/900 (not disjoint unique counts). Numeric combinations excluded from these source histories/queries; pretraining novelty unknown. No output-selected removal.

A uses all 81 legacy queries and separately the 69 semantically valid arithmetic queries. Four groups (12 queries) fail semantic applicability/label checks before new inference. B uses first two queries of 23 valid arithmetic donors. A(q0) and A(q1) differ. Both prompts show the union of A/B/correct answers from both questions, in fixed shuffled and reversed order. All 23 donors pass exclusion of A answers from all five histories, including added controls.

Concrete constructed example: 11/20 × 4/20 has A=44, B=11/5, correct=11/100; 15/19 × 8/19 has A=120, B=120/19, correct=120/361. Both see the same six numbers.

## 3. Coordinate and objective

No coordinate search or gradient optimization. B transports 64 decoder-block OUTPUT residuals at the single vision-pad address (64 × 5120). A uses no patch.

`J(g,o,c) = I[y(g,q0,o,c)=A(q0) and y(g,q1,o,c)=A(q1)]`

`D_joint = mean_(g,o)[J(g,o,swap) - J(g,o,random_real)]`

## 4. Arithmetic sign example

A swap-only joint hit contributes +1/46; a random-only joint hit -1/46. Always choosing one fixed A value yields 1/2 individual accuracy but zero joint accuracy.

## 5. Real measured examples

First directed-only joint success in BOTH orders:

Group 1, donor `subtraction.stops_borrow_at_zero`. Outputs listed for question 0/question 1; no examples used for filtering.

| Question | A target |
|---|---:|
| Calculate: 59009 - 47173 | 11976 |
| A stadium has 3001 seats. 1650 tickets were sold. How many seats are still available? | 1451 |

| Condition | Shuffled menu outputs | Reversed menu outputs |
|---|---|---|
| none | 11976 / 12176 | 11976 / 11976 |
| noop | 11976 / 12176 | 11976 / 11976 |
| swap | 11976 / 1451 | 11976 / 1451 |
| same_donor_rule | 11976 / 1451 | 1451 / 1451 |
| same_recipient_rule | 11976 / 12176 | 11976 / 11836 |
| random_real | 1351 / 1351 | 1351 / 1451 |

First directed joint failure in BOTH orders:

Group 0, donor `fractions.ignore_denominators`. Outputs listed for question 0/question 1; no examples used for filtering.

| Question | A target |
|---|---:|
| Multiply: 11/20 × 4/20 | 44 |
| Multiply: 15/19 × 8/19 | 120 |

| Condition | Shuffled menu outputs | Reversed menu outputs |
|---|---|---|
| none | 11/5 / 120/19 | 11/5 / 120/19 |
| noop | 11/5 / 120/19 | 11/5 / 120/19 |
| swap | 44 / 44 | 120 / 120 |
| same_donor_rule | 44 / 44 | 120 / 120 |
| same_recipient_rule | 11/5 / 120/19 | 11/5 / 120/19 |
| random_real | 11/5 / 120/19 | 11/5 / 120/19 |

## 6. Selection funnel

A: 27 donors × 3 queries × 3 descriptions = 243 generations. B: 23 donors × (2 questions + blank problem) × 2 orders × 6 conditions = 828 generations. Total 1,071. Every invalid output remains a failure.

## 7. Causal controls

Native, own-state no-op, directed A, distinct history with A's rule, distinct history with B's rule, semantically operation-applicable real random. Added histories are response-task-only with equal instructions and distinct literal histories. Random rules differ from A but may equal B. Some random answers are not on the union menu: the supported subset is reported below. Blank-problem joint success is structurally zero, not an independent discovery.

## 8. Steering definition

`h_B[layer,state] <- h_source[layer,state]` at prefill, full replacement; no dose tuning. Capture prefix through state once; no future question or menu enters capture. Reuse fixed tensors across questions, orders and blanks. Fresh KV per generation. Original late-fusion mask retains physically present but masked histories; this is not the strict history-free bridge.

## 9. Evaluation and parsing

Qwen3-32B + original seed-42 CF1 adapter, NF4/bfloat16/eager. Greedy, 48 tokens, no answer prefill or constrained decoder, EOS/closing-tag stopping. No truncation. Exact rational scoring under the original strict parser. B has visible candidates and no current correct-answer line. A supplies a general textual algorithm after state with empty history, no worked example. No LLM judge. The descriptive condition is not an oracle upper bound.

## 10. Aggregate and paired results

A, semantically valid arithmetic subset:

| Description | Donor-rule A matches | Recipient-rule B matches | Invalid | Token limit |
|---|---:|---:|---:|---:|
| donor_rule | 11/69 | 2/69 | 2 | 0 |
| recipient_rule | 4/69 | 7/69 | 0 | 0 |
| no_rule | 1/69 | 3/69 | 10 | 0 |

A donor-description minus no-rule A agreement: +0.1449, 95% cluster CI [+0.0580, +0.2464].

B, all 23 donors (92 question/order outputs per condition; 46 pair/orders):

| State | Individual A | Joint A | B answers | Invalid | Off menu |
|---|---:|---:|---:|---:|---:|
| none | 11/92 | 0/46 | 37 | 0 | 0 |
| noop | 12/92 | 1/46 | 37 | 0 | 0 |
| swap | 34/92 | 5/46 | 10 | 0 | 0 |
| same_donor_rule | 40/92 | 5/46 | 9 | 0 | 0 |
| same_recipient_rule | 13/92 | 2/46 | 36 | 0 | 0 |
| random_real | 14/92 | 1/46 | 25 | 0 | 0 |

Intervals: 10,000 paired resamples over donors, preserving both orders and questions. Exploratory marginal CIs, not multiplicity-adjusted; no post-hoc pass gate. Orders are not independent donor replications.

- joint_swap_minus_random_real: +0.0870, 95% cluster CI [-0.0217, +0.2174].
- joint_swap_minus_noop: +0.0870, 95% cluster CI [-0.0217, +0.2174].
- joint_swap_minus_same_recipient_rule: +0.0652, 95% cluster CI [-0.0435, +0.1739].

Additional sensitivities (do not replace primary):

- random_rule_both_targets_on_menu_subset: 11 donors; +0.1818, 95% cluster CI [+0.0000, +0.4091].
- native_joint_B_subset: 1 donors; +0.0000, 95% cluster CI [+0.0000, +0.0000].
- joint_swap_minus_random_real:donor: +0.0870, 95% cluster CI [-0.0217, +0.2174].
- joint_swap_minus_random_real:rule: +0.0870, 95% cluster CI [-0.0238, +0.2381].
- joint_swap_minus_same_donor_rule:donor: +0.0000, 95% cluster CI [-0.0870, +0.0870].
- joint_swap_minus_same_donor_rule:rule: +0.0000, 95% cluster CI [-0.0909, +0.1087].

Per-donor-rule fixed-menu detail:

| Donor rule | Directed joint | Random joint | Same-A-rule joint |
|---|---:|---:|---:|
| fractions.ignore_denominators | 0/4 | 0/4 | 0/4 |
| fractions.keep_common_denominator_for_multiplication | 1/2 | 0/2 | 1/2 |
| fractions.multiply_across_for_division | 0/4 | 0/4 | 0/4 |
| subtraction.always_borrow_left | 0/6 | 0/6 | 1/6 |
| subtraction.borrow_from_bottom | 1/6 | 0/6 | 1/6 |
| subtraction.borrow_no_decrement | 0/6 | 1/6 | 1/6 |
| subtraction.diff_0_n_equals_n | 0/6 | 0/6 | 0/6 |
| subtraction.smaller_from_larger | 0/6 | 0/6 | 0/6 |
| subtraction.stops_borrow_at_zero | 3/6 | 0/6 | 1/6 |

## 11. Failures, ordering, and numerical consistency

No-op and reproducibility diagnostics: `{"donor_same_rule_output": 20, "noop_native_A": 1, "noop_native_text": 3, "swap_order_output": 23}`.

Cross-role/source-ID state hash disagreements: `{}`.

Same-rule outputs can differ across histories; agreement is not assumed. Prefix/full-prompt quantized computation can make own-state no-op differ from native (see separate format audit). Report such differences rather than importing older zero no-op claims. All raw successes, failures, blank outputs and both menu orders are in the extended audit.

Explicitly post-hoc mixed-number/kg sensitivity changed 12 outputs; its complete separate counts are retained in the audit and never replace strict scoring.

## 12. Mechanism conclusion and limitations

The fixed-menu primary contrast does not provide a clearly positive interval on this panel. Do not promote it to evidence beyond constant-choice alternatives using selected successful examples or an alternative endpoint.

Explicit-description performance concerns instruction elicitation, not what is stored in the transported state. In the separate unsupported generation test, directed=random=3/81 (semantic-valid arithmetic: 1/69 each). That negative result remains. Only one checkpoint is tested, with reused development histories/questions and shared rule categories/control histories. Donor-cluster intervals do not prove independence across rules. Fresh histories, numbers and independent checkpoints would be needed for confirmation.

Also, final-block-output-only nulls have no downstream path to answer tokens and cannot establish single-vector insufficiency. See the computational-path audit.

## 13. Reproduction and execution

Experiment `c4-rule-followup-seed42-v2-20260905`, Feeds, 3 × Standard 80G1-A100, UAI `rankfun_aml`, approved image `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`. Jobs: `cyan_bell_k8778jy8h3`, `elated_shark_cd34fgg71s`, `dynamic_tomato_5kpsglrkjy`. Remote outputs: per-job `AMLT_OUTPUT_DIR/rule_followup`. Base `/mnt/amulet/studentmodeling/models/Qwen3-32B`; adapter `/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter`.

The v1 control-pool implementation was canceled before inspecting outputs, with all three jobs verified killed. It contributes no evidence. Frozen v2 manifest/amendment and 15-file SHA manifest remain unchanged.

Audits verify 1,071 unique expected keys, shard/group agreement, manifest/adapter/code/tokenizer lineage, prompt hashes, raw token-to-text decoding and parsing. Files and raw SHA256 checksums are in the extended JSON. Output completeness and backend terminal status are tracked separately in monitor receipts.

```bash
cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
/home/v-yuejiali/DeepGraph/.venv/bin/python scripts/audit_c4_rule_followup_extended.py \
  --download results/downloads/c4-rule-followup-seed42-v2-20260905 \
  --output results/c4_rule_followup_v2_seed42_20260905.extended_audit.json
/home/v-yuejiali/DeepGraph/.venv/bin/python scripts/render_c4_rule_followup_report.py \
  --audit results/c4_rule_followup_v2_seed42_20260905.extended_audit.json \
  --output reports/c4_rule_followup_results_20260905.md
```
