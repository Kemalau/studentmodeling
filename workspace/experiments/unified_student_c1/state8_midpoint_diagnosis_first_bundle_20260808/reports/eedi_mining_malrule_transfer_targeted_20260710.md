# Targeted Malrule Transfer Smoke — 2026-07-10

## Question

Does Malrule supervision help real Eedi-Mining misconception diagnosis?

This run narrows the previous broad multi-task transfer question to a cleaner local test:

- Baseline: Eedi-Mining diagnosis only.
- Treatment: Eedi-Mining diagnosis + Malrule identity + Malrule next-response.
- Reference: previous all-mixed C1 OCR model.

## Setup

All models use Qwen3-32B with the same 4-bit QLoRA smoke configuration:

- max length 1536
- max steps 120
- batch size 1
- gradient accumulation 4
- learning rate 1e-4
- seed 42
- eval: first 300 Eedi-Mining `diagnose_misconception` test rows
- scoring: candidate-target logprob over 32 misconception candidates

New treatment train data:

`experiments/unified_student_c1/data/processed/transfer_eedi_malrule_ocr_no_val/train_eedi_mining_plus_malrule_identity_response.jsonl`

| Task | Train rows |
|---|---:|
| `diagnose_misconception` | 3476 |
| `malrule_identity` | 3000 |
| `malrule_next_response` | 3000 |
| total | 9476 |

Treatment adapter:

`experiments/unified_student_c1/results/qwen3_32b_eedi_mining_plus_malrule_ir_ocr_no_val_qlora_s120_len1536/final_adapter`

Treatment eval:

`experiments/unified_student_c1/results/eedi_mining_plus_malrule_ir_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`

## Results

| Run | Acc | Macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| Eedi-only | 0.437 | 0.269 | 6.551 | 0.418 |
| Eedi + Malrule identity/response | 0.487 | 0.314 | 5.961 | 0.386 |
| All-mixed | 0.493 | 0.320 | 6.374 | 0.379 |

## Read

This is a positive local transfer signal.

Adding Malrule identity and Malrule next-response improves Eedi-Mining misconception diagnosis by:

- +5.0 accuracy points over Eedi-only;
- +4.5 macro-F1 points over Eedi-only;
- substantially better NLL than both Eedi-only and all-mixed.

It nearly matches the previous all-mixed accuracy while using a much narrower auxiliary mixture. This suggests the previous all-mixed gain on Eedi-Mining is plausibly coming from Malrule-style misconception supervision, not just generic multi-task regularization.

## Caveats

This is still a smoke result:

- only n=300 eval;
- only one seed;
- not the official Eedi MAP@25 retrieval protocol;
- not exposure-matched by number of Eedi diagnosis gradient updates.

Still, this is the cleanest C3-shaped evidence so far: executable misconception rules appear to transfer into real misconception diagnosis.

## Next

Run the decisive ablation:

1. Eedi-only.
2. Eedi + Malrule identity only.
3. Eedi + Malrule response only.
4. Eedi + Malrule identity + response.
5. All-mixed.

Then evaluate on full Eedi-Mining test and repeat with at least 3 seeds.
