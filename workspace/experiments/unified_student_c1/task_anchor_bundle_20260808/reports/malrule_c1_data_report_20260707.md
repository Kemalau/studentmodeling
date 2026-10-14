# MalruleLib C1 Clean Data Report

Date: 2026-07-07

## Why This Was Needed

The first C1 pilot used Eedi interaction logs. That was useful for testing the unified prompt/evaluation pipeline, but it cannot prove rule-level mathematical understanding because the current local Eedi canonical file mostly exposes student/item IDs, option letters, correctness, and skill metadata. It does not give clean executable misconception labels.

MalruleLib is the right source for the "understanding" side: every misconception is an executable rule, so the student answer, correct answer, and misconception identity can be mechanically verified.

## Implemented

New script:

`/data1/liyuejia/experiments/unified_student_c1/scripts/prepare_malrule_c1.py`

Evaluator patch:

`/data1/liyuejia/experiments/unified_student_c1/scripts/evaluate_c1.py`

The evaluator now supports per-row dynamic candidate sets, which is required for Malrule response prediction because each query problem has a different set of possible executable answers.

## Cleaning Filters

Each generated malrule problem must pass:

- Manifestation: malrule answer differs from the correct answer.
- Metadata consistency: if `expected_answer` or `correct_answer` exists in problem metadata, it must match the surface module's correct algorithm.
- Domain consistency: when applying malrule M to another generator's problem, M's own correct algorithm must agree with the surface generator's correct algorithm.
- Query informativeness: every query must have at least two distinct candidate answers.

## Output

Directory:

`/data1/liyuejia/experiments/unified_student_c1/data/processed/malrule_c1_clean/`

Raw student-query files:

- `raw_train.jsonl`
- `raw_val.jsonl`
- `raw_test_same_template.jsonl`
- `raw_test_cross_template.jsonl`

C1 task files:

- `train.jsonl`
- `val.jsonl`
- `test_same_template.jsonl`
- `test_cross_template.jsonl`

Audit/stat files:

- `stats.json`
- `audit_50_samples.json`

## Dataset Size

Each raw example is one synthetic student plus one query. Each raw example emits three C1 tasks:

- `malrule_next_response`
- `malrule_identity`
- `malrule_next_correctness`

| Split | Raw Examples | Task Rows |
|---|---:|---:|
| train | 3,000 | 9,000 |
| val | 300 | 900 |
| test_same_template | 300 | 900 |
| test_cross_template | 300 | 900 |

## Pool Stats

This C1 construction uses the three families `subtraction`, `decimals`, and `fractions`.

| Quantity | Count |
|---|---:|
| discovered malrules | 21 |
| pooled malrules | 21 |
| cross-capable malrules | 21 |
| surface A | 88 |
| surface B | 44 |

The pool is broader than the stricter Phase 0 C4 handoff target because this C1 data builder treats metadata `template`/`operation` as useful surface axes, not only `problem_type`. For C4 swap/probe, we should later tighten this to the exact Phase 0 contract.

## Split Quality

| Split | NONE Ratio | Avg Distinct Query Answers | Min Distinct Query Answers |
|---|---:|---:|---:|
| train | 0.187 | 3.033 | 2 |
| val | 0.207 | 3.013 | 2 |
| test_same_template | 0.213 | 3.047 | 2 |
| test_cross_template | 0.187 | 2.953 | 2 |

Fixed-seed reproducibility was checked by rerunning the full command and comparing SHA-256 hashes of `train.jsonl` and `test_cross_template.jsonl`; there was no diff.

## Simple Baselines

Same-template:

| Task | Best Simple Baseline | Accuracy |
|---|---|---:|
| malrule_identity | question_majority | 0.4900 |
| malrule_next_response | skill_majority | 0.4267 |
| malrule_next_correctness | history_state | 1.0000 |

Cross-template:

| Task | Best Simple Baseline | Accuracy |
|---|---|---:|
| malrule_identity | history_state/global_majority | 0.1867 |
| malrule_next_response | history_state | 0.4833 |
| malrule_next_correctness | history_state | 1.0000 |

`malrule_next_correctness` is intentionally not a headline task here: malrule students are constructed to be incorrect and NONE students are correct, so history alone solves it. The meaningful C1 Malrule tasks are response prediction and misconception identity.

## Qwen3-32B Results

Model:

`/data1/liyuejia/models/Qwen3-32B`

Unified QLoRA adapter:

`/data1/liyuejia/experiments/unified_student_c1/results/qwen3_32b_malrule_unified_qlora_s250/final_adapter`

Training used all 9,000 Malrule task rows for 250 steps with 4-bit QLoRA. Validation loss fell to `0.01656`, so this is a clean smoke-to-signal C1 run rather than just a data-pipeline check.

Same-template:

| Task | Best Simple Baseline Acc | 32B Base Acc | 32B Base Macro-F1 | 32B QLoRA Acc | 32B QLoRA Macro-F1 |
|---|---:|---:|---:|---:|---:|
| malrule_identity | 0.4900 | 0.5600 | 0.3706 | 0.7300 | 0.6235 |
| malrule_next_response | 0.4267 | 0.6567 | 0.4976 | 0.8600 | 0.7488 |
| malrule_next_correctness | 1.0000 | 0.8267 | 0.7937 | 1.0000 | 1.0000 |

Cross-template:

| Task | Best Simple Baseline Acc | 32B Base Acc | 32B Base Macro-F1 | 32B QLoRA Acc | 32B QLoRA Macro-F1 |
|---|---:|---:|---:|---:|---:|
| malrule_identity | 0.1867 | 0.5367 | 0.3658 | 0.7333 | 0.6559 |
| malrule_next_response | 0.4833 | 0.6233 | 0.4616 | 0.8600 | 0.7440 |
| malrule_next_correctness | 1.0000 | 0.8100 | 0.7652 | 1.0000 | 1.0000 |

Result files:

- `/data1/liyuejia/experiments/unified_student_c1/results/malrule_c1_qwen3_32b_base_same.json`
- `/data1/liyuejia/experiments/unified_student_c1/results/malrule_c1_qwen3_32b_base_cross.json`
- `/data1/liyuejia/experiments/unified_student_c1/results/malrule_c1_qwen3_32b_qlora_s250_same.json`
- `/data1/liyuejia/experiments/unified_student_c1/results/malrule_c1_qwen3_32b_qlora_s250_cross.json`

Interpretation: this is the first meaningful C1 signal on clean executable misconception data. The base 32B model already has nontrivial cross-template rule understanding, and unified QLoRA improves both headline Malrule tasks substantially. The correctness task is solved by construction and should remain a sanity check, not a claim.

## Remaining C1 Work

This run does not finish the full C1 paper claim yet. The remaining pieces are:

1. real-data C1 tasks beyond MalruleLib, especially correctness/knowledge tracing on student logs;
2. strong KT/CDM baselines for those real-data tasks;
3. single-task SFT controls versus unified multitask SFT, to test whether unification gives positive transfer rather than just coexistence;
4. larger held-out reporting with confidence intervals.
