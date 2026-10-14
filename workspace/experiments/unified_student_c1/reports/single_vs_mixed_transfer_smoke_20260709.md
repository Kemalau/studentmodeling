# Single vs Mixed Transfer Smoke — 2026-07-09

## Setup

Goal: check whether the current mixed C1 QLoRA model shows positive transfer over task-specific single-task QLoRA models.

All single-task models used the same quick smoke configuration as the mixed OCR run:

- Base model: `/data1/liyuejia/models/Qwen3-32B`
- 4-bit QLoRA
- max length 1536
- max steps 120
- batch size 1, gradient accumulation 4
- lr 1e-4
- seed 42
- evaluation: first 300 test rows for the target task, candidate-target logprob scoring

Important caveat: this is a smoke comparison, not the final C3 protocol. The mixed model was trained on the first 20k mixed examples due to `train_c1_lora.py`'s default `--max-train-samples=20000`; single-task models also used the same script and the same 120-step budget, but every single-task gradient update is task-specific. This means the comparison is useful as a transfer signal, but not yet a publication-grade exposure-matched ablation.

## Model Paths

Mixed model:

- `experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter`

Single-task models:

- `experiments/unified_student_c1/results/qwen3_32b_single_next_correctness_ocr_no_val_qlora_s120_len1536/final_adapter`
- `experiments/unified_student_c1/results/qwen3_32b_single_next_response_ocr_no_val_qlora_s120_len1536/final_adapter`
- `experiments/unified_student_c1/results/qwen3_32b_single_diagnose_misconception_ocr_no_val_qlora_s120_len1536/final_adapter`
- `experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_ocr_no_val_qlora_s120_len1536/final_adapter`

## Result Files

- `experiments/unified_student_c1/results/single_next_correctness_ocr_no_val_qwen3_32b_qlora_s120_len1536_eval_n300.json`
- `experiments/unified_student_c1/results/single_next_response_ocr_no_val_qwen3_32b_qlora_s120_len1536_eval_n300.json`
- `experiments/unified_student_c1/results/single_diagnose_misconception_ocr_no_val_qwen3_32b_qlora_s120_len1536_eval_n300.json`
- `experiments/unified_student_c1/results/single_malrule_next_response_ocr_no_val_qwen3_32b_qlora_s120_len1536_eval_n300.json`

## Transfer Table

`delta = mixed - single`.

| Task | Single acc | Mixed acc | Delta | Single macro-F1 | Mixed macro-F1 | Single NLL | Mixed NLL | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Eedi `next_correctness` | 0.577 | 0.573 | -0.003 | 0.553 | 0.525 | 0.658 | 0.677 | no positive transfer |
| Eedi `next_response` | 0.650 | 0.653 | +0.003 | 0.634 | 0.636 | 1.132 | 1.068 | tiny positive signal |
| Eedi-Mining `diagnose_misconception` | 0.437 | 0.493 | +0.057 | 0.269 | 0.320 | 6.551 | 6.374 | positive transfer candidate |
| Malrule `malrule_next_response` | 0.870 | 0.793 | -0.077 | 0.760 | 0.657 | 0.348 | 0.629 | negative transfer / dilution |

## Interpretation

The current evidence is mixed.

The clearest positive-transfer signal is Eedi-Mining misconception diagnosis: the mixed model beats the single-task diagnosis model by +5.7 accuracy points and +5.1 macro-F1 points. This is exactly the kind of result C3 wants: response/correctness/Malrule-style supervision may be helping the model organize misconception semantics.

Eedi next-response shows only a tiny gain: +0.3 accuracy points, with better NLL. This is directionally positive, but too small to claim yet.

Eedi next-correctness does not show transfer. Single-task is slightly better, and both remain below the global-majority baseline in raw accuracy. This task needs class balancing/calibration or a stronger KT-specific setup before it can support the C1/C3 story.

Malrule next-response shows negative transfer in this smoke: the single-task model is much stronger than the mixed model. The likely explanation is task dilution: this synthetic exact-answer task is highly learnable, and the single-task model spends all 120 steps on it, while the mixed model sees relatively few Malrule response examples among many heterogeneous tasks.

## Bottom Line

This smoke does not justify the broad claim "multi-task training improves all tasks." It does justify a narrower next hypothesis:

> Mixed training may help semantic misconception diagnosis, while exact synthetic response prediction may prefer task-specific exposure unless the mixture ratio is tuned.

For final C3, we need an exposure-matched ablation:

1. same base model;
2. same total compute;
3. controlled per-task sample exposure;
4. multiple seeds;
5. single-task, mixed-all, and leave-one-family/task-out variants;
6. full test evaluation, not only n=300 smoke.

The next practical move is to tune mixture ratios instead of treating all tasks uniformly. A plausible next run is `Eedi-Mining + Malrule identity/response` vs `Eedi-Mining only`, because the strongest positive transfer appeared on misconception diagnosis.
