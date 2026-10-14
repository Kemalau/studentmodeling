# Eedi KT Comparison

Date: 2026-07-07

## Setup

Dataset:

`/data1/liyuejia/experiments/unified_student_c1/data/processed/eedi_c1_k12/`

Task:

`next_correctness`, i.e. predict whether a held-out student answers the next math item correctly.

Split:

- train: 23,604 rows
- valid: 1,473 rows
- test: 1,479 rows
- student-held-out split
- 53 skills, 948 questions
- test correct rate: 996 / 1479 = 0.6734

Result files:

- `/data1/liyuejia/experiments/unified_student_c1/results/eedi_kt_baselines_full_test.json`
- `/data1/liyuejia/experiments/unified_student_c1/results/eedi_c1_qwen3_32b_unified_qlora_s100_next_correctness_full.json`
- `/data1/liyuejia/experiments/unified_student_c1/results/eedi_c1_qwen3_32b_kt_only_qlora_s150_next_correctness_full.json`
- `/data1/liyuejia/experiments/unified_student_c1/results/eedi_c1_qwen3_32b_kt_only_qlora_s300_next_correctness_full.json`

## Models Compared

- `global_rate`: always predicts the train-set correct rate.
- `skill_rate`: Laplace-smoothed correctness rate by skill.
- `question_rate`: Laplace-smoothed correctness rate by item/question.
- `history_rate`: Laplace-smoothed correctness rate from the student's prompt history.
- `logistic_features`: logistic model over item/skill rates and short history features.
- `dkt_skill_lstm`: DKT-style LSTM over historical skill/correctness events.
- `dkt_item_lstm_mlp`: item-aware LSTM with query skill/question embeddings and history features.
- `qwen3_32b_unified_qlora_s100`: the existing unified 32B QLoRA adapter, scored by candidate target likelihood.
- `qwen3_32b_kt_only_qlora_s150/s300`: Qwen3-32B QLoRA trained only on `next_correctness`, scored by candidate target likelihood.

## Test Results

| Model | Accuracy | Macro-F1 | AUC(correct) | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| global_rate | 0.6734 | 0.4024 | 0.5000 | 0.6881 | 0.2475 | 0.1661 |
| skill_rate | 0.5828 | 0.5431 | 0.5712 | 0.6782 | 0.2427 | 0.0292 |
| question_rate | 0.6295 | 0.6102 | 0.6724 | 0.6421 | 0.2261 | 0.0274 |
| history_rate | 0.5720 | 0.5239 | 0.5445 | 0.7332 | 0.2639 | 0.1361 |
| logistic_features | 0.6734 | 0.4024 | 0.5644 | 0.6319 | 0.2200 | 0.0469 |
| dkt_skill_lstm | 0.6011 | 0.5795 | 0.6284 | 0.6571 | 0.2329 | 0.0092 |
| dkt_item_lstm_mlp | 0.6288 | 0.6029 | 0.6474 | 0.6474 | 0.2281 | 0.0181 |
| qwen3_32b_unified_qlora_s100 | 0.6660 | 0.4254 | 0.5661 | 0.7464 | 0.2442 | 0.1446 |
| qwen3_32b_kt_only_qlora_s150 | 0.6024 | 0.5213 | 0.5642 | 0.6455 | 0.2270 | 0.0394 |
| qwen3_32b_kt_only_qlora_s300 | 0.6018 | 0.5107 | 0.5638 | 0.6435 | 0.2263 | 0.0295 |

## Readout

The unified 32B QLoRA does not currently win the KT cell. Its accuracy is close to the global majority baseline because it predicts `correct` for 1440 / 1479 test rows. On the more meaningful KT metrics, it loses to the item/question baseline and the item-aware DKT-style model:

- AUC: 32B QLoRA `0.5661` vs question_rate `0.6724` vs item-aware DKT `0.6474`.
- Macro-F1: 32B QLoRA `0.4254` vs question_rate `0.6102` vs item-aware DKT `0.6029`.
- NLL: 32B QLoRA `0.7464` vs question_rate `0.6421` vs item-aware DKT `0.6474`.

The KT-only run improves calibration and NLL relative to the multitask 32B adapter, but it does not improve ranking. Its AUC stays around `0.564`, essentially unchanged from the unified adapter and far below `question_rate` / item-aware DKT. Longer KT-only training from 150 to 300 steps does not help.

This is a real C1 weakness, not just multitask undertraining. The Malrule result is strong, but the real-log KT result is not yet competitive against simple item difficulty / KT baselines.

## Next Actions

1. Rebuild the Eedi prompt without relying on thin pseudo-item text; add richer item text/options if available, or use FoundationalASSIST if it exposes cleaner content.
2. Try a discriminative scoring head or calibrated probe over LLM hidden states for KT, because candidate likelihood over XML tags is currently not producing useful ranking.
3. Add a SAKT/AKT-style baseline if this KT cell becomes paper-critical.
4. For C1 framing, treat KT as the hardest cell: the current claim should be "Malrule misconception/response is strong; KT on real logs still needs work."
