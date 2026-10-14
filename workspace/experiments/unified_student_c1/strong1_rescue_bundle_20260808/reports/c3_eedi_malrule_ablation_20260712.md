# C3 Eedi-Malrule Transfer Ablation

Date: 2026-07-12

## Question

Does synthetic Malrule supervision improve real Eedi-Mining misconception diagnosis?

This is a local C3 transfer smoke test. The target task is always real `diagnose_misconception` on Eedi-Mining. We compare an Eedi-only model against variants that add different Malrule auxiliary tasks during training.

## Setup

- Base model: `/data1/liyuejia/models/Qwen3-32B`
- Training: QLoRA, r=8, alpha=16, lr=1e-4, max length 1536, max steps 120, response-token-only loss.
- Eval: Eedi-Mining `diagnose_misconception`, `n=300`, candidate-target logprob scoring.
- Split/source: `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val/test.jsonl`
- No bottleneck/state mask in this C3 run. This is a task-transfer test, not a C4 controllability test.

## Training Sets

| condition | training tasks | train rows |
|---|---|---:|
| Eedi-only | `diagnose_misconception` | 3,476 |
| Eedi+Malrule identity | `diagnose_misconception` + `malrule_identity` | 6,476 |
| Eedi+Malrule response | `diagnose_misconception` + `malrule_next_response` | 6,476 |
| Eedi+Malrule identity+response | `diagnose_misconception` + `malrule_identity` + `malrule_next_response` | 9,476 |
| All mixed C1 | all available C1 smoke tasks | mixed |

New ablation files created:

- `experiments/unified_student_c1/data/processed/transfer_eedi_malrule_ocr_no_val/train_eedi_mining_plus_malrule_identity.jsonl`
- `experiments/unified_student_c1/data/processed/transfer_eedi_malrule_ocr_no_val/train_eedi_mining_plus_malrule_response.jsonl`

## Results

| condition | n | acc | delta acc | macro-F1 | delta F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eedi-only | 300 | 0.437 | +0.000 | 0.269 | +0.000 | 6.551 | 0.418 |
| Eedi+Malrule identity | 300 | 0.470 | +0.033 | 0.314 | +0.045 | 6.286 | 0.393 |
| Eedi+Malrule response | 300 | 0.423 | -0.013 | 0.271 | +0.003 | 6.474 | 0.427 |
| Eedi+Malrule identity+response | 300 | 0.487 | +0.050 | 0.314 | +0.045 | 5.961 | 0.386 |
| All mixed C1 | 300 | 0.493 | +0.057 | 0.320 | +0.051 | 6.374 | 0.379 |

## Read

There is a positive C3 transfer signal, but it is specific:

- Adding Malrule identity supervision improves real Eedi diagnosis: `+3.3` acc points and `+4.5` macro-F1 points.
- Adding Malrule response-only does not help: acc drops by `-1.3` points, while macro-F1 is essentially flat.
- Combining identity and response is better than identity alone on accuracy and NLL, but macro-F1 is unchanged.
- The all-mixed model is still the best on accuracy and macro-F1 in this n=300 smoke setting.

Interpretation: the useful transfer is not simply "more synthetic Malrule text." The useful auxiliary signal appears to be the explicit rule/misconception identity, which aligns with Eedi's diagnosis target. Response prediction alone teaches executable answer style, but it does not reliably transfer into Eedi misconception classification under this short training regime.

## Caveats

- This is still smoke scale: one seed, `n=300`, and 120 training steps.
- It is not exposure matched: adding Malrule increases total examples and task diversity.
- Eedi-Mining evaluation here is closed candidate logprob accuracy/F1, not the official MAP@25 retrieval protocol.
- This is C3 task-transfer evidence only; it does not test state readability or state controllability.

## Main Artifacts

- Eedi-only eval: `experiments/unified_student_c1/results/single_diagnose_misconception_ocr_no_val_qwen3_32b_qlora_s120_len1536_eval_n300.json`
- Eedi+Malrule identity eval: `experiments/unified_student_c1/results/eedi_mining_plus_malrule_identity_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`
- Eedi+Malrule response eval: `experiments/unified_student_c1/results/eedi_mining_plus_malrule_response_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`
- Eedi+Malrule identity+response eval: `experiments/unified_student_c1/results/eedi_mining_plus_malrule_ir_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`
- All mixed eval: `experiments/unified_student_c1/results/mixed_c1_ocr_no_val_qwen3_32b_qlora_s120_len1536_eedi_mining_n300.json`

