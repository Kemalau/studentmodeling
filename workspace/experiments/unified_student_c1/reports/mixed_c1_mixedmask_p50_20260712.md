# Mixed-Mask p=0.5 Unified C1/C4 Smoke, 2026-07-12

## Setup

Implemented a mixed-mask training option in `train_c1_lora.py`:

- `--bottleneck-mask-prob 0.5`
- each training example gets late-fusion bottleneck mask with probability 0.5;
- otherwise it gets standard causal attention via a 4D causal mask.

Run:

- Base: `/data1/liyuejia/models/Qwen3-32B`
- Adapter: `experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_mixedmask_p50_qlora_s120/final_adapter`
- Train data: `experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck/train.jsonl`
- Steps: 120
- Train loss: 0.2947

## C1 Open-Attention Eval

Same filtered test split and open-attention eval as previous comparisons.

| task | base | full-open train | latefusion-open | mixed-mask p50 open | p50-full | p50-lf |
|---|---:|---:|---:|---:|---:|---:|
| diagnose_misconception | 0.460/0.282 | 0.500/0.326 | 0.487/0.310 | 0.440/0.283 | -0.060 | -0.047 |
| future_mastery | 0.413/0.410 | 0.467/0.379 | 0.410/0.357 | 0.443/0.291 | -0.023 | +0.033 |
| malrule_identity | 0.527/0.358 | 0.593/0.463 | 0.573/0.427 | 0.563/0.402 | -0.030 | -0.010 |
| malrule_next_correctness | 0.813/0.769 | 0.977/0.963 | 0.927/0.894 | 0.993/0.989 | +0.017 | +0.067 |
| malrule_next_response | 0.627/0.466 | 0.780/0.643 | 0.697/0.521 | 0.807/0.664 | +0.027 | +0.110 |
| next_correctness | 0.673/0.439 | 0.563/0.512 | 0.490/0.487 | 0.500/0.492 | -0.063 | +0.010 |
| next_response | 0.643/0.624 | 0.647/0.631 | 0.533/0.446 | 0.600/0.569 | -0.047 | +0.067 |

## C4 Masked Swap

All-layer state replacement on `malrule_next_response`, `n=150` pairs, masked late-fusion eval.

All pairs:

| condition | flip_to_A | stay_B | correct | other | mean p_A | mean p_B |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.1467 | 0.2667 | 0.3800 | 0.2067 | 0.1849 | 0.2240 |
| noop | 0.1467 | 0.2400 | 0.4133 | 0.2000 | 0.1846 | 0.2260 |
| samemal | 0.1600 | 0.2667 | 0.3800 | 0.1933 | 0.1861 | 0.2248 |
| random | 0.1467 | 0.2667 | 0.3933 | 0.1933 | 0.1855 | 0.2237 |
| swap | 0.1467 | 0.2667 | 0.3800 | 0.2067 | 0.1854 | 0.2248 |

Clean subset where `none=stay_B`: `n=40`.

| condition | flip_to_A | stay_B | correct | other | changed top-1 | mean delta p_A | mean delta p_B |
|---|---:|---:|---:|---:|---:|---:|---:|
| noop | 0.0000 | 0.8750 | 0.1000 | 0.0250 | 0.1250 | -0.0030 | +0.0072 |
| samemal | 0.0000 | 0.9750 | 0.0250 | 0.0000 | 0.0250 | -0.0014 | +0.0025 |
| random | 0.0000 | 0.9500 | 0.0500 | 0.0000 | 0.0500 | +0.0002 | -0.0005 |
| swap | 0.0250 | 0.9500 | 0.0250 | 0.0000 | 0.0500 | -0.0005 | +0.0005 |

## Read

The p=0.5 mixed-mask model partially improves the C1/C4 tradeoff on task performance:

- It beats latefusion-open on `malrule_next_correctness`, `malrule_next_response`, `future_mastery`, `next_correctness`, and `next_response`.
- It even beats full-open on the two Malrule next-step tasks.
- It remains below full-open on real-log `next_correctness` and `next_response`.

But it does **not** preserve C4 state controllability:

- Swap is indistinguishable from none/random/samemal.
- Clean subset has only `1/40 = 0.025` flip_to_A under swap.
- `noop` is not perfectly stable, likely because many examples are near decision boundaries; in any case, no directed state effect is visible.

Conclusion: random 50/50 mask mixing is not enough for a single model that is both high-performing and causally state-controllable. It teaches task behavior, but does not force answer generation to rely on `<STUDENT_STATE>` strongly enough.

## Next Candidate

A better "one model" recipe should be staged rather than purely random:

1. train open for C1 competence;
2. continue with late-fusion bottleneck on Malrule/state-heavy tasks;
3. optionally keep a small open fraction so C1 does not collapse.

This is more likely to preserve the state circuit than p=0.5 simultaneous mixing.
