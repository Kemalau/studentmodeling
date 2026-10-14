# C1 OCR Mixed Smoke Report — 2026-07-09

## Setup

- Base model: `/data1/liyuejia/models/Qwen3-32B`
- Adapter: `experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter`
- Train data: `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val/train.jsonl`
- Eval data: no-val setting, evaluated directly on held-out test JSONL files.
- Training: 4-bit QLoRA, max length 1536, 120 steps, batch size 1, grad accumulation 4, lr 1e-4.
- LoRA: r=8, alpha=16, dropout=0.05; target modules include attention projections and MLP projections.

This is a smoke run, not the final C1 run. It is useful for checking whether OCR text helps and whether the mixed-task format is learnable.

## Result Files

- Eedi OCR K12 tasks: `experiments/unified_student_c1/results/mixed_c1_ocr_no_val_qwen3_32b_qlora_s120_len1536_eedi_tasks_n300.json`
- Malrule tasks: `experiments/unified_student_c1/results/mixed_c1_ocr_no_val_qwen3_32b_qlora_s120_len1536_malrule_tasks.json`
- Eedi-Mining diagnose task: `experiments/unified_student_c1/results/mixed_c1_ocr_no_val_qwen3_32b_qlora_s120_len1536_eedi_mining_n300.json`

## Headline Metrics

| Data/task | n | Model acc | Model macro-F1 | Best simple baseline | Verdict |
|---|---:|---:|---:|---|---|
| Eedi OCR `future_mastery` | 300 | 0.463 | 0.380 | question majority 0.470 | roughly baseline |
| Eedi OCR `next_correctness` | 300 | 0.573 | 0.525 | global majority 0.703; question majority 0.633 | improved, not competitive yet |
| Eedi OCR `next_response` | 300 | 0.653 | 0.636 | question majority 0.650 | tiny win |
| Eedi-Mining `diagnose_misconception` | 300 | 0.493 | 0.320 | simple majority 0.083 | strong closed-set signal |
| Malrule `malrule_identity` | 300 | 0.587 | 0.453 | majority/history 0.187 | strong |
| Malrule `malrule_next_correctness` | 300 | 0.980 | 0.968 | history-state 1.000; majority 0.813 | saturated, near oracle baseline |
| Malrule `malrule_next_response` | 300 | 0.793 | 0.657 | history-state 0.483 | strong |

## Comparison To Previous Thin-Text Smoke

| Task | Previous thin-text acc | OCR-text acc | Change |
|---|---:|---:|---:|
| Eedi `future_mastery` | 0.473 | 0.463 | -0.010 |
| Eedi `next_correctness` | 0.463 | 0.573 | +0.110 |
| Eedi `next_response` | 0.620 | 0.653 | +0.033 |
| Malrule `malrule_identity` | 0.447 | 0.587 | +0.140 |
| Malrule `malrule_next_correctness` | 1.000 | 0.980 | -0.020 |
| Malrule `malrule_next_response` | 0.793 | 0.793 | +0.000 |

OCR mainly helps the Eedi response setting and makes the answer-choice distribution more sensible. It does not solve KT correctness yet.

## Interpretation

The mixed generative format is working mechanically: the same 32B LoRA can score Eedi response prediction, Eedi-Mining misconception diagnosis, and Malrule synthetic tasks with the same target-tag interface.

The strongest current C1 evidence is not KT yet. It is:

- Malrule exact response prediction: 0.793 vs 0.483 history-state baseline.
- Malrule identity diagnosis: 0.587 vs 0.187 majority/history baseline.
- Eedi-Mining closed-set misconception diagnosis: 0.493 vs 0.083 simple baselines.

The weak point is Eedi next correctness. Accuracy improves substantially with OCR text, but still loses to simple correctness priors. The model predicts both classes now, but the task is imbalanced and the question-level prior is strong.

## Caveats

- Evaluation is only `n=300` per task for this smoke.
- No validation split was used in this run.
- Eedi-Mining here is a closed-set candidate scoring smoke, not the official MAP@25 retrieval protocol.
- Eedi OCR is imperfect: the OCR pass marked 782/948 question images as ok and 166/948 as weak.
- The QLoRA run is only 120 steps, far from a full convergence run.

## Next Fixes

1. Add class-balanced or calibrated scoring for Eedi `next_correctness`.
2. Run a larger evaluation, at least full test for Eedi response and Malrule.
3. Separate task-mixture ablations: Eedi-only vs Eedi+Malrule vs all mixed.
4. For C1 tables, add proper task baselines beyond simple priors where available.
