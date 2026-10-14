# C4 rule follow-ups: frozen pre-inference protocol

## 1. Question and non-goals

Test (A) whether explicit descriptions can elicit the donor's wrong algorithm
without answer candidates, and (B) whether a fixed donor state produces different
rule-consistent answers to different questions when the visible menu is held
identical. Reuse the historical CF1 vision-pad seed-42 checkpoint. No training,
recurrent update, new model selection, or claim of independent replication.
This is an exploratory follow-up motivated by the already-observed format null.

## 2. Sample construction

Source is the previously frozen 27-donor / 81-query / 11-donor-rule numerical
panel. Its source-file row counts are CF1 train 6,000 and valid 600; original
MalRule train 9,000, valid 900, same-template test 900, cross-template test 900.
They are not disjoint-file counts. Operand/text combinations were excluded from
the union of those histories and queries; pretraining novelty is unknown. No
new query numbers are generated and no model-output-based filtering occurs.

A: all 81 queries x donor-rule description / recipient-rule description /
no-rule = 243 generations. History is empty in all three conditions. The
description is a general algorithm, with no worked numeric example or current
answer. Preserve the output-tag instruction, state address, raw prompt format
and original attention-mask implementation. Descriptions are visible after
the state position. This tests instruction elicitation, not state transport.

Independent algorithm and semantic-applicability checks occur BEFORE inference.
All 81 remain in the legacy table; 69 arithmetic queries form the valid primary
descriptive panel. Six queries apply a numerator-comparison rule to subtraction
(groups 22,26). Six decimal queries' recipient labels disagree with the
independent textual comparison algorithm (groups 21,25); these are reported
separately. Do not silently modify old labels or treat old software executability
as sufficient proof of semantic rule applicability.

B: take the first two queries of every valid arithmetic donor, 23 donors / 46
queries. The two donor answers differ. The menu is the UNION of donor, recipient,
and mathematical answers across the two queries, with duplicates removed.
It is exactly identical for the two questions. Fixed shuffled and reversed
orders are both evaluated, retaining the same order across the question pair.
Both targets thus occur in both prompts; a constant answer or position cannot
answer both questions correctly. This controls fixed answer/position preference,
not every question-dependent heuristic. These are reused development questions,
not an independent confirmation set or the original three-answer menu.

Choose a distinct same-donor-rule history as the first eligible history in
sorted example-ID order. Choose a real-random history, RNG 2026090517+group_idx,
from sorted same-family histories whose rule is semantically applicable on BOTH
questions, independently agrees with the library target on both, and yields
different answers from A on both. Exclude every original role identity plus
the same-donor-rule identity. Random rule differs from A but MAY equal B;
requiring a third rule would leave some operations without a valid control.
Report identities, rules and pool sizes. A prespecified subset excludes donor
target numbers in any history, including newly added controls.

Concrete parent-panel example: `2 5/8 / 1 1/2` under multiply-without-inverting
has A=63/16, B=7, correct=7/4. It is a constructed target, not a success.

## 3. Coordinate, objectives and formulas

A uses no patches. Report exact rational donor and recipient agreement, invalid
outputs, token-limit stops, and paired donor-agreement contrasts between the
donor description and the other descriptions. A positive contrast plus good
absolute adherence supports elicitable execution for that description/panel,
not a theoretical upper bound or proof that an internal state contains it.

B patches decoder-block OUTPUT residuals at the single vision-pad position,
all 64 blocks, 64 x 5120 numbers. No layer search, ranking or gradients.

For donor g, candidate order o and condition c:

`J(g,o,c) = I[y(g,q0,o,c)=A(q0) AND y(g,q1,o,c)=A(q1)]`.

Primary contrast: `D_joint = mean_(g,o)[J(g,o,swap)-J(g,o,random_real)]`.
Also report each-question A agreement, same-donor-rule reproducibility,
recipient-rule preservation, no-op discrepancies, order sensitivity and the
history-answer-exclusion subset. Each donor, not each query/order, is a bootstrap
unit: 10,000 paired donor-cluster resamples, retaining both orders and questions.
Secondary contrasts have unadjusted exploratory marginal intervals; no gate is
retrofitted from outcomes. A positive primary interval is evidence on this panel,
not proof of universal rule abstraction or an independent replication.

## 4. Arithmetic sign example

If directed state answers both questions as A while random gets one wrong,
that donor/order contributes +1/(23*2) to D_joint. Always choosing one of the
two distinct A values gives J=0, even if its per-question hit rate is 1/2.

## 5. Prior measured observation

On the parent panel, candidates-only free generation gave donor agreement
61/81 directed versus 29/81 real random, whereas neither candidates nor correct
answer gave 3/81 for both. These motivate the follow-up; they are not new results.

## 6. Selection funnel

A: 27 donors -> 81 original queries -> 3 conditions -> 243 outputs, with 69
semantically valid queries separately specified. B: exclude four non-arithmetic
or rule-inapplicable donor groups -> 23 donors -> first two questions plus one
query-withheld prompt -> 2 menu orders -> 6 state conditions -> 828 outputs.
Total 1,071 raw generations. Shard by original group_idx modulo 3; no overlap.

## 7. Causal controls

B conditions: native; own-state no-op; directed donor; another donor with A's
rule; another student with B's rule; operation-applicable real-random donor.
Each condition also generates once per order with `[Problem withheld.]` and
the identical menu. Score that single blank output against each of the two
different targets; its joint success is necessarily zero. Do not treat this
structural zero as an independent statistical discovery. Report any questionless
answer preference instead of averaging it away.

## 8. Intervention and dose

Capture every history only through the state position, without future queries
or candidate lists. Reuse tensors across questions, orders and blank prompts.
`h_B[layer,state] <- h_source[layer,state]`, full replacement, no dose tuning.
Histories remain physically present but masked under the existing late-fusion
mask. This is not the strict history-free bridge. Fresh KV cache per generation.
Record state hashes; all shards use identical code, model and manifest hashes.

## 9. Evaluation rubric

Qwen3-32B, original seed-42 CF1 adapter; NF4 4-bit/bfloat16, eager attention.
Greedy generation, maximum 48 tokens, no answer prefill or constrained decoder,
stop at EOS/closing tag, reject truncation. Strict original parser: one tagged
numeric response or bare scalar, exact rational equality. Invalid, nonnumeric,
multi-answer and unfinished outputs remain failures. Candidate menus are visible
in B, so B is not unsupported open execution. No judge model or teacher forcing.

## 10. Aggregate and paired results

Pending at freeze. Audit all 1,071 unique keys, prompt hashes, raw parsing,
group/shard membership, per-group/merged agreement and checkpoint/code lineage.
Report full tables including negative results, rule/family detail, counts and
cluster uncertainty. Do not compare these different denominators to old 0.411.

## 11. Successes, failures and corruptions

Pending. Include representative matched question-pair successes and failures,
same-rule discrepancies, order reversals and malformed outputs. Do not select
examples without reporting aggregate denominators.

## 12. Scope and falsifiers

Explicit-rule failure leaves instruction/format/arithmetic as alternatives to
state insufficiency. Explicit-rule success with state failure localizes a gap
between available execution capability and this state interface, not necessarily
absence of information. Positive fixed-menu D_joint rules out constant-answer
preference but remains compatible with a learned rule-conditioned verifier.
It does not show a program lives inside the trajectory or establish human
student causality. Confirmatory generalization needs fresh histories/numbers and
independent checkpoints after these exploratory designs are fixed.

## 13. Reproduction

Builder/scorer: `scripts/c4_rule_followup.py`.
Evaluator: `scripts/evaluate_c4_rule_followup.py`.
Manifest: `manifests/c4_rule_followup_seed42_20260905.json`.
Config: derived from previously accepted format diagnostic config via
`scripts/prepare_c4_rule_followup.py`; Feeds, 3 x Standard 80G1-A100,
approved image `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`,
UAI `rankfun_aml`. Shared remote base `/mnt/amulet/studentmodeling/models/Qwen3-32B`,
adapter `/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter`.
Per-job output `AMLT_OUTPUT_DIR/rule_followup`, atomic donor checkpoints.
Experiment intended name `c4-rule-followup-seed42-v1-20260905`.
Submission and completion receipts are separate from this pre-inference protocol.
