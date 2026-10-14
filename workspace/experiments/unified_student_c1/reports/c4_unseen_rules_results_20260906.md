# C4 unseen-rule evaluation: complete audited results

## 1. Question and non-goals

Frozen final A (matched candidate-free SFT) and B (state-exchange training) are tested for induction and transport of constructed rules from three observations. Both are final step 240, seed 42. No training, checkpoint selection, recurrent-state update, spontaneous rule invention or pretraining-novelty claim.

## 2. Sample construction

Four categories × four unordered history pairs × two swap directions × three queries = 96 directional queries per condition per checkpoint (48 unique questions). Nine conditions × two checkpoints = 1,728 generations. Six constructed rules and two seen-rule controls are separate. Generator seed 2026090603; denominator range 4–40, three templates, both parities in each history and query triplet. Query combinations are excluded from audited CF1/MalRule sources and previous panels. Current targets are not in the history, current correct answer or any menu. Old histories contain their own correct answers.

Concrete constructed offset-3 history: 36/40 × 39/40 → 1407; 16/37 × 12/37 → 195; 6/16 × 15/16 → 93. Queries and actual outputs appear below. New means absent as a labelled mechanism from audited fine-tuning sources, not absent from pretraining. Histories identify rules only within the eight declared rules.

## 3. Coordinate and exact objective

All 64 decoder-block output residuals at the vision-pad state token. `D = mean_i(1[swap_i = donor_target_i] − 1[random_i = donor_target_i])`; primary comparison is `D_B − D_A`, separately by category. This evaluates donor-rule answer agreement, not human-student fidelity.

## 4. Arithmetic sign example

Illustration only: 12/24 swap hits and 3/24 control hits yield D = +0.375. Positive means more donor-consistent answers with the donor trajectory than the wrong-rule real-state control.

## 5. Fixed checkpoints, not ranked candidates

- A: `7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6`; fixed before this test, no result-based selection.
- B: `87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32`; fixed before this test, no result-based selection.

## 6. Selection funnel

Six constructed rules + two known controls → four pairs/category → outcome-free number, answer-collision and finite-rule-identifiability filters → all 32 directional groups evaluated. Original manifest records 331 seen-number rejections, three copy/repeated-answer rejections and 1,920 independent answer checks. No post-generation exclusions.

## 7. Held-out controls

`native`: recipient history; `donor_native`: donor history; `noop`: independently captured own state; `swap`: donor state; `random`: independently sampled recipient-rule state (not Gaussian noise); `same_donor`: another donor-rule history; `oracle`: explicit donor rule without history; `open_history`: donor history with ordinary causal attention; `empty_history`: no history/rule.

## 8. Intervention and dose

`h_recipient^l ← s_donor^l` after blocks l=0…63 at the state-token address, dose 1, prefill only. One captured history trajectory is reused on three queries; each generation uses fresh KV. Corrected late-fusion mask and answer anchor are used except for open-history. This test is not a history-free bridge or a last-layer ablation.

## 9. Decoding and scoring

NF4/bfloat16 with kbit preparation; eager attention; rank-8 frozen adapter, eval mode, dropout off. Greedy decoding, 48-token cap, 768-token input+output bound, no truncation. Single tagged or bare numeric output, exact rational equality. Invalid, multiple-answer and token-limit outputs count as misses. No LLM judge.

## 10. Aggregate and paired results

Each row has n=24 directional queries and eight three-query groups; only four independent unordered pairs. CIs: 10,000 pair-cluster bootstraps, seed 2026090604, unadjusted, exploratory. Zero-width CIs on uniform outcomes do not prove zero population uncertainty.

| Arm | Category | Swap hits | Wrong-rule hits | D [95% CI] | Swap all-three | Random all-three | No-op mismatches |
|---|---|---:|---:|---|---:|---:|---:|
| A | offset | 0/24 | 0/24 | 0.000 [0.000, 0.000] | 0/8 | 0/8 | 0/24 |
| A | composition | 1/24 | 0/24 | 0.042 [0.000, 0.125] | 0/8 | 0/8 | 0/24 |
| A | conditional | 14/24 | 13/24 | 0.042 [-0.083, 0.167] | 0/8 | 1/8 | 0/24 |
| A | seen_control | 24/24 | 0/24 | 1.000 [1.000, 1.000] | 8/8 | 0/8 | 0/24 |
| B | offset | 0/24 | 0/24 | 0.000 [0.000, 0.000] | 0/8 | 0/8 | 1/24 |
| B | composition | 1/24 | 0/24 | 0.042 [0.000, 0.125] | 0/8 | 0/8 | 0/24 |
| B | conditional | 14/24 | 13/24 | 0.042 [-0.083, 0.167] | 0/8 | 1/8 | 0/24 |
| B | seen_control | 24/24 | 0/24 | 1.000 [1.000, 1.000] | 8/8 | 0/8 | 0/24 |

| Category | Paired B − A specificity [95% CI] |
|---|---|
| offset | 0.000 [0.000, 0.000] |
| composition | 0.000 [0.000, 0.000] |
| conditional | 0.000 [0.000, 0.000] |
| seen_control | 0.000 [0.000, 0.000] |

### All conditions (donor-target hits / 24; invalid count in parentheses)

| Arm | Category | native | donor_native | noop | swap | random | same_donor | oracle | open_history | empty_history |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | offset | 0/24 (0) | 0/24 (0) | 0/24 (0) | 0/24 (0) | 0/24 (0) | 0/24 (0) | 19/24 (0) | 0/24 (0) | 0/24 (0) |
| A | composition | 0/24 (0) | 1/24 (0) | 0/24 (0) | 1/24 (0) | 0/24 (0) | 0/24 (0) | 1/24 (0) | 0/24 (0) | 0/24 (0) |
| A | conditional | 10/24 (0) | 14/24 (0) | 10/24 (0) | 14/24 (0) | 13/24 (0) | 11/24 (0) | 12/24 (0) | 14/24 (0) | 8/24 (0) |
| A | seen_control | 0/24 (0) | 24/24 (0) | 0/24 (0) | 24/24 (0) | 0/24 (0) | 24/24 (0) | 20/24 (0) | 23/24 (0) | 3/24 (0) |
| B | offset | 0/24 (0) | 0/24 (0) | 0/24 (0) | 0/24 (0) | 0/24 (0) | 0/24 (0) | 19/24 (0) | 0/24 (0) | 0/24 (0) |
| B | composition | 0/24 (0) | 1/24 (0) | 0/24 (0) | 1/24 (0) | 0/24 (0) | 0/24 (0) | 1/24 (0) | 0/24 (0) | 0/24 (0) |
| B | conditional | 10/24 (0) | 14/24 (0) | 10/24 (0) | 14/24 (0) | 13/24 (0) | 11/24 (0) | 11/24 (0) | 14/24 (0) | 7/24 (0) |
| B | seen_control | 0/24 (0) | 24/24 (0) | 0/24 (0) | 24/24 (0) | 0/24 (0) | 24/24 (0) | 20/24 (0) | 23/24 (0) | 3/24 (0) |

### Per-rule diagnostic (donor-target hits / 12)

| Arm | Donor rule | Donor native | Swap | Random | Oracle | Open history |
|---|---|---:|---:|---:|---:|---:|
| A | offset_3 | 0/12 | 0/12 | 0/12 | 9/12 | 0/12 |
| A | offset_7 | 0/12 | 0/12 | 0/12 | 10/12 | 0/12 |
| A | shift_1_keep | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| A | shift_2_keep | 1/12 | 1/12 | 0/12 | 1/12 | 0/12 |
| A | even_ignore_odd_keep | 7/12 | 7/12 | 7/12 | 5/12 | 8/12 |
| A | even_keep_odd_ignore | 7/12 | 7/12 | 6/12 | 7/12 | 6/12 |
| A | seen_ignore | 12/12 | 12/12 | 0/12 | 12/12 | 12/12 |
| A | seen_keep | 12/12 | 12/12 | 0/12 | 8/12 | 11/12 |
| B | offset_3 | 0/12 | 0/12 | 0/12 | 9/12 | 0/12 |
| B | offset_7 | 0/12 | 0/12 | 0/12 | 10/12 | 0/12 |
| B | shift_1_keep | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| B | shift_2_keep | 1/12 | 1/12 | 0/12 | 1/12 | 0/12 |
| B | even_ignore_odd_keep | 7/12 | 7/12 | 7/12 | 4/12 | 8/12 |
| B | even_keep_odd_ignore | 7/12 | 7/12 | 6/12 | 7/12 | 6/12 |
| B | seen_ignore | 12/12 | 12/12 | 0/12 | 12/12 | 12/12 |
| B | seen_keep | 12/12 | 12/12 | 0/12 | 8/12 | 11/12 |

## 11. Concrete successes, failures and output corruption

Examples are the first complete swap success and first non-complete-swap group in manifest order per category/arm; missing kinds are explicitly noted. They are descriptive, not selected evaluation data. All groups, raw text, parity breakdowns and paired query transitions are retained in the audit JSON.

- A, offset: no all-three success group.

### A / offset:000:0 (failure/partial)

Donor: `offset_3`; recipient: `offset_7`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| Multiply: 16/20 × 13/20 | 211 | 208 | 208 | 211 | 212 |
| What is 27/39 times 19/39? | 516 | 513 | 513 | 540 | 513 |
| What is 5/16 times 3/16? | 18 | 15 | 15 | 18 | 15 |
- A, composition: no all-three success group.

### A / composition:000:0 (failure/partial)

Donor: `shift_1_keep`; recipient: `shift_2_keep`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| What is 18/26 times 5/26? | 57/13 | 45/13 | 45 | 50/13 | 9 |
| Multiply: 21/23 × 2/23 | 66/23 | 42/23 | 42 | 45/23 | 42/23 |
| Calculate (26/32) * (28/32). | 783/32 | 91/8 | 728 | 819/8 | 273/4 |
- A, conditional: no all-three success group.

### A / conditional:000:0 (failure/partial)

Donor: `even_ignore_odd_keep`; recipient: `even_keep_odd_ignore`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| Calculate (17/24) * (7/24). | 119 | 119 | 119 | 119 | 119 |
| What is 8/11 times 8/11? | 64/11 | 64 | 64/11 | 64/11 | 64 |
| Calculate (29/40) * (9/40). | 261 | 261 | 261 | 261/40 | 261 |

### A / seen_control:000:0 (all-three success)

Donor: `seen_ignore`; recipient: `seen_keep`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| Calculate (17/20) * (4/20). | 68 | 68 | 17/5 | 68 | 68 |
| What is 36/37 times 32/37? | 1152 | 1152 | 1152/37 | 1152 | 1152 |
| Multiply: 10/32 × 21/32 | 210 | 210 | 105/16 | 210 | 210 |
- A, seen_control: no failure/partial group.

A output formats: `{"tagged": 864}`; stop reasons: `{"closing_tag": 864}`.

First invalid output: none.
- B, offset: no all-three success group.

### B / offset:000:0 (failure/partial)

Donor: `offset_3`; recipient: `offset_7`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| Multiply: 16/20 × 13/20 | 211 | 208 | 208 | 211 | 212 |
| What is 27/39 times 19/39? | 516 | 513 | 513 | 540 | 513 |
| What is 5/16 times 3/16? | 18 | 15 | 15 | 18 | 15 |
- B, composition: no all-three success group.

### B / composition:000:0 (failure/partial)

Donor: `shift_1_keep`; recipient: `shift_2_keep`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| What is 18/26 times 5/26? | 57/13 | 45/13 | 45 | 50/13 | 9 |
| Multiply: 21/23 × 2/23 | 66/23 | 42/23 | 42 | 45/23 | 42/23 |
| Calculate (26/32) * (28/32). | 783/32 | 91/8 | 728 | 819/8 | 91/4 |
- B, conditional: no all-three success group.

### B / conditional:000:0 (failure/partial)

Donor: `even_ignore_odd_keep`; recipient: `even_keep_odd_ignore`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| Calculate (17/24) * (7/24). | 119 | 119 | 119 | 119/12 | 119 |
| What is 8/11 times 8/11? | 64/11 | 64 | 64/11 | 64/11 | 64 |
| Calculate (29/40) * (9/40). | 261 | 261 | 261 | 261/40 | 261 |

### B / seen_control:000:0 (all-three success)

Donor: `seen_ignore`; recipient: `seen_keep`.

| Query | Donor target | Swap | Random | Oracle | Open history |
|---|---|---|---|---|---|
| Calculate (17/20) * (4/20). | 68 | 68 | 17/5 | 68 | 68 |
| What is 36/37 times 32/37? | 1152 | 1152 | 1152/37 | 1152 | 1152 |
| Multiply: 10/32 × 21/32 | 210 | 210 | 105/16 | 210 | 210 |
- B, seen_control: no failure/partial group.

B output formats: `{"tagged": 864}`; stop reasons: `{"closing_tag": 864}`.

First invalid output: none.

## 12. Mechanism conclusion and limitations

- A: constructed categories with a positive lower specificity CI: none. Complete three-query donor agreement across constructed-rule groups: 0/24. Seen control: swap 24/24, wrong-rule 0/24.
- A/offset: explicit-rule oracle 19/24; open history 0/24; donor native 0/24; swap 0/24; wrong-rule 0/24. Specificity 0.000 [0.000, 0.000].
- A/composition: explicit-rule oracle 1/24; open history 0/24; donor native 1/24; swap 1/24; wrong-rule 0/24. Specificity 0.042 [0.000, 0.125].
- A/conditional: explicit-rule oracle 12/24; open history 14/24; donor native 14/24; swap 14/24; wrong-rule 13/24. Specificity 0.042 [-0.083, 0.167].
- A: own-state no-op output mismatches 0/96; retained, not excluded.
- B: constructed categories with a positive lower specificity CI: none. Complete three-query donor agreement across constructed-rule groups: 0/24. Seen control: swap 24/24, wrong-rule 0/24.
- B/offset: explicit-rule oracle 19/24; open history 0/24; donor native 0/24; swap 0/24; wrong-rule 0/24. Specificity 0.000 [0.000, 0.000].
- B/composition: explicit-rule oracle 1/24; open history 0/24; donor native 1/24; swap 1/24; wrong-rule 0/24. Specificity 0.042 [0.000, 0.125].
- B/conditional: explicit-rule oracle 11/24; open history 14/24; donor native 14/24; swap 14/24; wrong-rule 13/24. Specificity 0.042 [-0.083, 0.167].
- B: own-state no-op output mismatches 1/96; retained, not excluded.
A and B have identical measured primary specificity in every category. This is no evidence of an exchange-training advantage here; it is not a population equivalence test.
This panel does not support reliable unseen-rule state transport. Known-rule transport and unseen-rule induction are different claims. A strong oracle with weak open-history performance points toward history induction/identification as a limitation; a weak oracle leaves execution or instruction-following unresolved. Open-history is itself an evaluation-path change, so these contrasts do not uniquely locate a cause.

Interpret each new-rule category separately. Partial conditional-branch agreement is not a transported conditional rule: all-three success across both parities is the relevant stronger diagnostic. Oracle failure implicates execution/instruction following; oracle success with weak open-history implicates induction; stronger open-history than donor-native points to a bottleneck limitation; donor-native versus swap diagnoses transport. These are patterns, not uniquely identified causal explanations.

Only four independent history pairs per category and one training seed; this is exploratory, with no preregistered replication gate or multiplicity correction. A/B alone does not identify improvement over the old checkpoint on these new rules. The narrow positive claim, if supported, is rule-specific cross-query state transport in this fraction task—not arbitrary program execution, spontaneous invention, recurrent self-updating state or human fidelity. Negative categories remain in the report.

## 13. Reproduction and completion audit

Exact record coverage, frozen code/manifest/adapter lineage, old-source hashes, tokenizer/base metadata hashes, prompt token lengths, target reference formulas, all output texts/tokens, runtime package equality.
Base metadata hashes are verified; this is not a full hash audit of every 32B base-weight shard.

Experiment: `c4-unseen-rules-v1-seed42-feedsllm-20260906`. Both arms require complete group files and merged records; all 1,728 records passed the delivery audit. See `results/c4_unseen_rules_v1_feedsllm_20260906.submission.json` for job IDs, target, quota and submission command.

Protocol: `reports/c4_unseen_rules_protocol_20260906.md`; manifest: `manifests/c4_unseen_rules_v1_20260906.json`; frozen source list: `manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256`; evaluator and collector are in `scripts/`.
