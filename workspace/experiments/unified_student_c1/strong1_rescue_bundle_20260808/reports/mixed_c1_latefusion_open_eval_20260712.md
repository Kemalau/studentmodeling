# Mixed C1 Late-Fusion Open-Attention Eval, 2026-07-12

## Setup

Same unified C1 late-fusion adapter, evaluated two ways:

- **masked**: test with late-fusion bottleneck mask.
- **open**: test with standard attention, no bottleneck mask.

This isolates whether the poor mixed-C1 late-fusion masked numbers came from test-time history blocking.

## Files

- Adapter: `experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_latefusion_qlora_s120/final_adapter`
- Test data: `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck/test.jsonl`
- Base eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_qwen3_32b_base_all_tasks_n300.json`
- Late-fusion masked eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_latefusion_qwen3_32b_qlora_s120_all_tasks_n300.json`
- Late-fusion open eval: `experiments/unified_student_c1/results/mixed_c1_ocr_lte1536_latefusion_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`

## Result

| task | base acc/F1 | late-fusion masked acc/F1 | late-fusion open acc/F1 | open - masked acc | open - base acc |
|---|---:|---:|---:|---:|---:|
| diagnose_misconception | 0.460/0.282 | 0.483/0.308 | 0.487/0.310 | +0.003 | +0.027 |
| future_mastery | 0.413/0.410 | 0.403/0.233 | 0.410/0.357 | +0.007 | -0.003 |
| malrule_identity | 0.527/0.358 | 0.267/0.216 | 0.573/0.427 | +0.307 | +0.047 |
| malrule_next_correctness | 0.813/0.769 | 0.787/0.495 | 0.927/0.894 | +0.140 | +0.113 |
| malrule_next_response | 0.627/0.466 | 0.457/0.315 | 0.697/0.521 | +0.240 | +0.070 |
| next_correctness | 0.673/0.439 | 0.380/0.365 | 0.490/0.487 | +0.110 | -0.183 |
| next_response | 0.643/0.624 | 0.530/0.427 | 0.533/0.446 | +0.003 | -0.110 |

## Read

Open attention strongly helps the Malrule tasks:

- `malrule_identity`: 0.267 -> 0.573
- `malrule_next_correctness`: 0.787 -> 0.927
- `malrule_next_response`: 0.457 -> 0.697

It does not fully recover real-log KT/response tasks:

- `next_correctness`: 0.380 -> 0.490, still below base 0.673
- `next_response`: 0.530 -> 0.533, still below base 0.643

Interpretation: the late-fusion bottleneck itself is not intrinsically bad. For Malrule, training with state bottleneck and testing open improves both task performance and preserves the C4 story. For mixed C1 real-log tasks, the short S120 unified adapter is undertrained or negatively interfered; simply opening history is not enough to match the base model.
