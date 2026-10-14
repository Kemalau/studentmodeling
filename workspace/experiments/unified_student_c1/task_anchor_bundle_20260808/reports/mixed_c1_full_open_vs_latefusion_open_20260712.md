# Mixed C1 Fully-Open Training vs Late-Fusion Training, 2026-07-12

## Purpose

Compare two unified C1 adapters on the same filtered test set:

- **fully-open train**: trained without bottleneck/late-fusion mask, evaluated open.
- **late-fusion train, open test**: trained with late-fusion bottleneck, evaluated open.

This is the fair comparison for C1 task performance, since both use the same `lte1536` test split and standard open attention at evaluation time.

## Files

- Base eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_qwen3_32b_base_all_tasks_n300.json`
- Fully-open train eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_FULLYOPEN_train_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`
- Late-fusion open eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_latefusion_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`
- Late-fusion masked eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_latefusion_qwen3_32b_qlora_s120_all_tasks_n300.json`

## Results

| task | base acc/F1 | fully-open train acc/F1 | late-fusion train, open test acc/F1 | LF open - full open acc | LF open - base acc |
|---|---:|---:|---:|---:|---:|
| diagnose_misconception | 0.460/0.282 | 0.500/0.326 | 0.487/0.310 | -0.013 | +0.027 |
| future_mastery | 0.413/0.410 | 0.467/0.379 | 0.410/0.357 | -0.057 | -0.003 |
| malrule_identity | 0.527/0.358 | 0.593/0.463 | 0.573/0.427 | -0.020 | +0.047 |
| malrule_next_correctness | 0.813/0.769 | 0.977/0.963 | 0.927/0.894 | -0.050 | +0.113 |
| malrule_next_response | 0.627/0.466 | 0.780/0.643 | 0.697/0.521 | -0.083 | +0.070 |
| next_correctness | 0.673/0.439 | 0.563/0.512 | 0.490/0.487 | -0.073 | -0.183 |
| next_response | 0.643/0.624 | 0.647/0.631 | 0.533/0.446 | -0.113 | -0.110 |

## Read

Fully-open training is currently stronger for C1 pure task performance. It beats late-fusion-open on all seven tasks under the same test split.

Late-fusion-open is still useful because it preserves part of the C4 state-intervention story and beats base on the Malrule tasks:

- `malrule_identity`: +0.047 over base
- `malrule_next_correctness`: +0.113 over base
- `malrule_next_response`: +0.070 over base

But on real-log tasks, late-fusion-open is still below base:

- `next_correctness`: -0.183 below base
- `next_response`: -0.110 below base

Working interpretation:

- For **C1 headline performance**, fully-open training is the current better recipe.
- For **C4 causal state**, late-fusion/bottleneck training is the better recipe.
- If we want one model to serve both, we probably need a mixed-mask curriculum or auxiliary consistency recipe rather than pure late-fusion throughout.
