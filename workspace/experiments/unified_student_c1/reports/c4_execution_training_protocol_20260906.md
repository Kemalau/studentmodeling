# C4 execution-training pilot: frozen pre-training design

## 1. Question and non-goals

Does candidate-free execution training improve open student simulation, and does
explicit differentiable state-exchange training improve it beyond matched SFT?
One exploratory seed (42), same historical CF1 vision-pad adapter, rank 8.
No recurrent training, no claim of naturally emerging ability in the old model.
No checkpoint, prompt, seed or endpoint selection from final-test results.

## 2. Sample construction

Fresh programmatic subtraction and common-denominator multiplication; six rules:
smaller-from-larger, borrow-no-decrement, diff-0-n-equals-n, stops-borrow-at-zero,
ignore-denominators, keep-common-denominator-for-multiplication. This is a narrow
repair panel, not all MalRule families, unseen rules, new template families, or
human students. Half the pairs concern fractions, half subtraction; per-rule
counts are retained in the manifest (not claimed perfectly balanced).
An outcome-free identifiability check finds all test and validation histories
unique among the six-rule panel. Two underlying auxiliary training histories
(four directed-role appearances) are compatible with two subtraction rules;
they are not used as native/donor training sources, and remain documented in
the manifest rather than silently claiming universal history identifiability.

Train: 160 unordered history pairs, two directions per pair, three new questions
per direction, each in explicit-rule and history-only form = 1,920 rows.
Validation: eight unordered pairs = 96 rows. Test: 16 unordered pairs, two
directions, three questions = 96 directional queries (48 unique question texts).
Test has ten conditions per directional query = 960 generations per model.
Old checkpoint + final A + final B = 2,880 primary-generation records.

Every history contains three diagnostic observations; four distinct histories
per pair support native/own-state, directed donor, same-donor-rule, and random
wrong-rule controls. Random rule equals recipient rule with a distinct history;
it is applicable and yields a different answer from donor. No cross-rule unseen
generalization or fully arbitrary random-state distribution is claimed.

Operand and rendered-number multiset signatures are excluded against old CF1
6,000/600 and MalRule 9,000/900/900/900 source rows and the old 81-query numerical
panel. New train/valid/test number signatures are mutually disjoint. Two
directions intentionally share questions and histories within an unordered pair.
Current targets must differ from each other and the correct answer, not occur
among visible-history numeric tokens, and differ across a pair's three questions.
All new histories are generated, not reused from old evaluation. Pretraining
novelty and unique unseen answer strings are not claimed.

Generator RNG 2026090601. Four subtraction algorithms and ignore-denominators
are checked against operand-based existing library implementations. Common-
denominator multiplication is independently checked by direct rational arithmetic;
the library's target-metadata shortcut is not misrepresented as an independent
oracle. 5,520 answer checks passed before model inference. All filters/counts,
source and library hashes are in the manifest.

Illustrative sample (not an observed success): history indicates ignore-denominators;
`2/7 × 3/7` is supervised as 6 rather than the correct 6/49. Other questions for
the same history must have different targets. The actual sampled examples are
in `manifests/c4_execution_training_v1_20260906.json`.

## 3. Coordinate, loss and formulas

Let s_A^l be donor residual at the vision-pad address after decoder block l.
Arm A uses ordinary answer-token cross-entropy with donor history.
Arm B replaces recipient state h_B^l <- s_A^l at all 64 blocks during its
history-only phase, supervising the same donor answer as arm A.

`L = -(1/|T|) sum_(t in target and EOS) log p_theta(y_t | prefix,y_<t)`.

Prompt and donor-source tokens have labels -100; only target-row answer and EOS
are supervised. B's source row has instruction/history/state ONLY, no current
question, candidates, correct answer or labels. Source and recipient are rows
of a single forward pass. Pure block-output hooks remain installed through
backward; cloning/assignment is differentiable. Non-reentrant checkpointing
recomputes the same operation. Source history receives gradients in CPU tests.

## 4. Arithmetic sign example

Increasing p(correct donor target) from 0.2 to 0.5 changes its token loss from
1.609 to 0.693, a reduction of 0.916. This illustration is not a measured result.
State probe accuracy is not optimized or used for model selection.

## 5. Previous measured motivation

Old unsupported generation was 3/81 directed and 3/81 real random; explicit
rule-description follow-up achieved only 11/69 on semantic-valid arithmetic.
These motivate execution training, not a positive-control ceiling.

## 6. Selection funnel and training budget

No layer search, neural-coordinate ranking or hyperparameter sweep. All manifest
rows retained; no model-output filtering. Both arms: 240 optimizer steps,
accumulation 8, identical target sequence and supervised-token exposure,
AdamW lr 2e-5, 12 warmup steps, linear decay, weight decay .01, grad clip 1.
First 120 steps explicit rule with empty history; last 120 history-only.
Same warm start, rank 8, NF4/bfloat16/eager, dropout disabled in both arms.
Both arms execute two-row forwards, including the unused source in A, for
approximately matched compute; exact elapsed time and sequence lengths reported.
Max sequence 768. No truncation permitted. Checkpoint every 40 steps, final
step 240 is the only selected endpoint. Loss checks at steps 120/240 descriptive.
Original adapter immutable. Resume restores optimizer/scheduler/RNG and exact
example index; only checkpoints with atomic completion receipts are loaded.

## 7. Causal validation and controls

Test conditions: native recipient, native donor, own-state no-op, directed swap,
real wrong-rule donor, same-donor-rule history, explicit-rule/no-history oracle,
and candidate-visible native/swap/random retention conditions. All conditions
use the same questions per pair. The candidate-retention set is new and small,
not a claim of retaining T1-T5 or the entire historical C4 evaluation.

Primary endpoint: per-query specificity D = mean(hit_A(swap)-hit_A(random)),
and the matched improvement D_B-D_A. Secondary: D_A versus initial checkpoint,
absolute native/donor/oracle accuracy, same-rule, no-op, all-three-query joint
rule agreement, and candidate-visible retention. Report every endpoint whether
positive or negative. Bootstrap 10,000 times over unordered history-pair IDs,
preserving both directions and all questions; 16 independent clusters, not 96.
Intervals exploratory and unadjusted; no retrospective replication gate.

## 8. Steering and decode consistency

Full replacement h_B^l <- s_A^l, no dose scaling. Prefix state captured once per
history and reused across questions/conditions in evaluation, fresh KV each time.
Late-fusion masks prevent post-state tokens from reading history and prevent
query tokens from reading state; the answer-anchor and generated answers CAN
read state. This matches current trainer's intended mask. Earlier generation
script blocked the anchor as well: therefore the original checkpoint is freshly
evaluated with this exact decoder. Old numerical results are not subtracted as
the baseline. No history-free or vector-versus-trajectory comparison is claimed.

## 9. Evaluation rubric

Greedy, 48 token limit, EOS or closing response-tag stopping. No current correct
answer, rule description, candidate list or worked example in primary open
conditions. Explicit-rule and menu controls are separately labelled. Strict
single number/tag parser, exact rational equality. Truncation/invalid/multiple
answers count as failures. No LLM judge. Hash prompts and source states; retain
raw text/tokens/stop causes and all negative outcomes.

## 10. Aggregate and paired reporting

Pending. Require 960 unique expected records per model, exact matched target
exposure across arms, final-adapter and manifest lineage, group files versus
merged records, regenerated prompts and strict parsing. The collector computes
pair-cluster CIs and B-minus-A on the frozen manifest without selecting examples.

## 11. Failures and corruption

Pending. Preserve malformed outputs, nonfinite losses/gradients, no-op mismatches,
rule-level failures, candidate retention losses and all-three failures. Failed
runs remain engineering failures until diagnosed, not evidence of scientific null.

## 12. Scope and falsifiers

This intervention explicitly TRAINS the causal interface. Even if it succeeds,
do not retroactively claim the old representation was an executable rule.
If oracle improves but native does not, history identification/interface remains
weak. If native improves but swap does not, transport remains weak. If B does
not outperform matched A, do not claim swap supervision is necessary. Small
test and single seed only justify pilot evidence. Neither arm guarantees general
rule execution, arithmetic compositionality, recurrent updating, or human fidelity.

## 13. Reproduction and execution

Builder/model/runner: `scripts/c4_execution_data.py`, `c4_execution_model.py`,
`run_c4_execution_training.py`; tests `tests/test_c4_execution_training.py`.
Collector: `scripts/collect_c4_execution_training.py`.
Derived AMLT config: `amlt/feeds_c4_execution_training_v1_20260906.yaml`.
Two Standard 80G1-A100 jobs, approved image
`amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`,
explicit UAI `rankfun_aml`, primary target Feeds. Alternate target requires user
approval and separate receipt. Each job bounded at 8 hours, maximum two attempts.
Inputs: `/mnt/amulet/studentmodeling/models/Qwen3-32B` and
`/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter`.
Small manifest uploaded with source package; output/checkpoints written directly
to `AMLT_OUTPUT_DIR/execution_A` or `execution_B` on project Blob storage.
Local one-backward 32B diagnostic performs NO optimizer update and contributes
no scientific result. Submission and completion receipts maintained separately.
