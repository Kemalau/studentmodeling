# Mixed C1 Late-Fusion Smoke - 2026-07-12

## Setup

- Base model: `/data1/liyuejia/models/Qwen3-32B`
- Train data:
  `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck/train.jsonl`
- Test data:
  `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck/test.jsonl`
- Adapter:
  `experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_latefusion_qlora_s120/final_adapter`
- Eval:
  `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_latefusion_qwen3_32b_qlora_s120_all_tasks_n300.json`
- Training budget: 120 steps, batch size 1, gradient accumulation 4
- Max length: 1536
- Mask: `--bottleneck-mask --bottleneck-style late_fusion`
- Scoring: closed-set candidate target logprob reranking

The original mixed OCR dataset had some OCR-history prompts above 1536 tokens.
For bottleneck training, those cannot be left-truncated freely because the mask
needs intact `<student_history>` and `<STUDENT_STATE>` positions. A filtered
derived dataset was made:

| split | kept | dropped |
|---|---:|---:|
| train | 76,914 | 3,623 |
| test | 5,835 | 232 |

Only Eedi OCR rows were dropped. Malrule and Eedi-Mining rows were unchanged.

## Results

Comparison is against the previous mixed OCR s120 smoke report. This is not a
perfectly controlled comparison because the new run uses the lte1536 filtered
dataset, but Malrule rows are unchanged, and the direction is clear.

| task | n | late acc | old acc | delta | late macro-F1 | old macro-F1 | delta F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| future_mastery | 300 | 0.403 | 0.463 | -0.060 | 0.233 | 0.380 | -0.147 | 1.122 | 0.039 |
| next_correctness | 300 | 0.380 | 0.573 | -0.193 | 0.365 | 0.525 | -0.160 | 0.734 | 0.174 |
| next_response | 300 | 0.530 | 0.653 | -0.123 | 0.427 | 0.636 | -0.209 | 1.221 | 0.221 |
| diagnose_misconception | 300 | 0.483 | 0.493 | -0.010 | 0.308 | 0.320 | -0.012 | 6.246 | 0.404 |
| malrule_identity | 300 | 0.267 | 0.587 | -0.320 | 0.216 | 0.453 | -0.237 | 1.904 | 0.065 |
| malrule_next_correctness | 300 | 0.787 | 0.980 | -0.193 | 0.495 | 0.968 | -0.473 | 0.593 | 0.191 |
| malrule_next_response | 300 | 0.457 | 0.793 | -0.336 | 0.315 | 0.657 | -0.342 | 0.990 | 0.091 |

## Base-Model Comparison

I also evaluated the untrained Qwen3-32B base model on the same lte1536 filtered
test set, with the same candidate-reranking protocol, but without the
bottleneck mask. This is the cleanest available "before training" comparison.
The base model has not learned the added state token, so applying late-fusion
masking to base would conflate "no finetuning" with "untrained bottleneck
interface".

Base eval file:
`experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_qwen3_32b_base_all_tasks_n300.json`

| task | base acc | old mixed acc | late-fusion acc | late-base | late-old | base F1 | late F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| future_mastery | 0.413 | 0.463 | 0.403 | -0.010 | -0.060 | 0.410 | 0.233 |
| next_correctness | 0.673 | 0.573 | 0.380 | -0.293 | -0.193 | 0.439 | 0.365 |
| next_response | 0.643 | 0.653 | 0.530 | -0.113 | -0.123 | 0.624 | 0.427 |
| diagnose_misconception | 0.460 | 0.493 | 0.483 | +0.023 | -0.010 | 0.282 | 0.308 |
| malrule_identity | 0.527 | 0.587 | 0.267 | -0.260 | -0.320 | 0.358 | 0.216 |
| malrule_next_correctness | 0.813 | 0.980 | 0.787 | -0.027 | -0.193 | 0.769 | 0.495 |
| malrule_next_response | 0.627 | 0.793 | 0.457 | -0.170 | -0.336 | 0.466 | 0.315 |

Read: the late-fusion mixed s120 model is not merely worse than the old mixed
s120 model; it is also worse than the untrained base model on six of seven
accuracy columns. The only small exception is Eedi-Mining
`diagnose_misconception`, where late-fusion beats base by +0.023 accuracy and
+0.026 macro-F1.

## Read

This s120 unified late-fusion smoke does **not** improve C1. It is worse on all
seven evaluated tasks.

The result does not contradict the positive C4 single-task late-fusion result:

- The C4 positive run trained only `malrule_next_response` for 800 steps.
- This C1 run mixes seven tasks and trains for only 120 steps.
- Late-fusion bottleneck makes the problem harder: answer tokens cannot use
  direct history or question-state shortcuts.
- The new state token and bottleneck interface likely need more task-specific
  exposure before the mixed model can recover.

Most important failure:

- Malrule tasks fall sharply even though their rows were not filtered out. This
  strongly suggests undertraining / task dilution rather than only a filtered
  data artifact.

## Immediate Next Options

1. Run a longer mixed late-fusion model, at least 800 steps, matching the
   successful C4 exposure scale.
2. Use a staged curriculum:
   - stage 1: Malrule late-fusion response/identity;
   - stage 2: mixed C1 late-fusion.
3. Tune mixture ratios so Malrule state tasks are not diluted by Eedi OCR KT
   tasks.
4. Add task-specific evaluation jobs instead of one all-task eval, because
   32B + 4D mask + candidate reranking is very slow.

Current bottom line:

> Late fusion is promising for C4 controllability, but the quick unified C1
> s120 smoke does not transfer that gain to mixed multitask performance.
