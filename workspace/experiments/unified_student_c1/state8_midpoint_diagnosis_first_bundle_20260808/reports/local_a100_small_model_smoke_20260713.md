# Local A100 Small-Model Smoke

Date: 2026-07-13

## Environment

- GPU: NVIDIA A100 80GB PCIe
- Python environment: `/home/v-yuejiali/.venvs/studentmodeling`
- Model: `/home/v-yuejiali/studentmodeling/models/Qwen3-0.6B`
- Dataset source: MalruleLib commit `8ddedfb`
- PyTorch: `2.11.0+cu128`
- Transformers: `4.57.6`
- PEFT: `0.19.1`
- bitsandbytes: `0.49.2`

## Data

Generated `data/processed/malrule_c1_tiny` with:

- 240 raw training students, expanded to 720 task rows;
- 48 raw validation students, expanded to 144 task rows;
- 60 raw same-template and 60 raw cross-template test students;
- 21 discovered executable malrules across subtraction, decimals, and fractions.

## GPU Preflight

Qwen3-0.6B loaded successfully in NF4 4-bit mode and completed a CUDA forward pass. The check used approximately 0.51 GiB allocated GPU memory.

## Training Smoke

Task: `malrule_next_response`

- train rows: 240
- optimizer steps: 20
- effective batch size: 8
- max length: 512
- LoRA trainable parameters: 5,046,272
- runtime: 17.83 seconds
- loss at steps 1-10 aggregate: 0.6365
- loss at steps 11-20 aggregate: 0.0848

## Same-Template Evaluation

Closed-set candidate log-probability evaluation on the first 30 examples:

| model | accuracy | macro-F1 | NLL | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B base | 0.0667 | 0.0370 | 3.5580 | 0.7590 | 0.6722 |
| Qwen3-0.6B LoRA, 20 steps | 0.3667 | 0.2319 | 1.1715 | 0.2113 | 0.2882 |

This is a pipeline smoke rather than a paper-grade result. It confirms that dataset generation, 4-bit loading, LoRA training, adapter saving, and closed-set evaluation all work on the local server.

## Reproduction

Run:

```bash
bash scripts/run_local_qwen3_0p6b_malrule_smoke.sh
```
