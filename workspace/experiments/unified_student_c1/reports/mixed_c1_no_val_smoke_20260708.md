# Mixed C1 No-Val Smoke Run 2026-07-08

## Setup

- Train data: `experiments/unified_student_c1/data/processed/mixed_c1_no_val/train.jsonl`
- Test data: `experiments/unified_student_c1/data/processed/mixed_c1_no_val/test.jsonl`
- Model: `/data1/liyuejia/models/Qwen3-32B`
- Adapter: `experiments/unified_student_c1/results/qwen3_32b_mixed_c1_no_val_qlora_s120_fast/final_adapter`
- Training: QLoRA, 120 steps, no validation split, max length 1024, gradient accumulation 4.
- Evaluation: candidate log-likelihood scoring, test subset with max 50 examples per task.

This is a fast smoke run, not a final C1 number. The earlier n=300 evaluation was interrupted because the current evaluator is too slow for multi-task 32B candidate scoring.

## Data Mix

Train rows: 80,537

| Source | Rows |
|---|---:|
| Eedi response | 68,061 |
| MalruleLib | 9,000 |
| Eedi-Mining IR | 3,476 |

Test rows: 6,067

| Source | Rows |
|---|---:|
| Eedi response | 4,273 |
| MalruleLib cross-template | 900 |
| Eedi-Mining IR fold 4 | 894 |

## Test Results, n=50 Per Task

| Task | Model Acc | Model Macro-F1 | Model NLL | Best Baseline Acc | Best Baseline |
|---|---:|---:|---:|---:|---|
| diagnose_misconception | 0.500 | 0.328 | 6.796 | 0.060 | global_majority |
| future_mastery | 0.440 | 0.411 | 1.114 | 0.520 | question_majority |
| malrule_identity | 0.660 | 0.383 | 1.071 | 0.220 | global_majority |
| malrule_next_correctness | 1.000 | 1.000 | 0.088 | 1.000 | history_state |
| malrule_next_response | 0.840 | 0.724 | 0.556 | 0.560 | history_state |
| next_correctness | 0.640 | 0.639 | 0.653 | 0.620 | global_majority |
| next_response | 0.520 | 0.514 | 1.272 | 0.560 | question_majority |

For `next_correctness`, model AUC on the n=50 subset is 0.671.

## Readout

- The mixed model preserves strong Malrule behavior even with Eedi response dominating the mixture.
- Eedi-Mining misconception diagnosis shows a clear signal in the 32-way closed candidate setting.
- Eedi response `next_response` and `future_mastery` do not beat the strongest simple baselines in this smoke run.
- Eedi response KT is only slightly above the best accuracy baseline on n=50 and should not be treated as a stable win.

## Caveats

- `diagnose_misconception` here uses gold + 31 deterministic negatives, not official MAP@25 over all 2,587 misconceptions.
- The Eedi response dataset still lacks full option text, so its C1 value is limited compared with FoundationalASSIST.
- This run did not use validation or model selection.
- Full-test evaluation needs a faster evaluator with task-level progress and preferably prefix caching.
