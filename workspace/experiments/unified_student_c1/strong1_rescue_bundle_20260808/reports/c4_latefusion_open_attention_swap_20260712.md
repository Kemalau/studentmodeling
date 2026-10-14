# C4 Late-Fusion Open-Attention Check, 2026-07-12

## Question

We trained the single-task Malrule next-response model with the late-fusion bottleneck mask, then evaluated the same adapter in two test-time modes:

- **masked**: answer tokens can attend to `<STUDENT_STATE>` and question, but not history.
- **open**: standard attention at test time; answer tokens can attend to history directly as well as state.

This tests whether the learned state remains useful when direct history is available, and whether state-swap still has causal effect under conflicting evidence.

## Model

- Base: `/data1/liyuejia/models/Qwen3-32B`
- Adapter: `experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter`
- Task: `malrule_next_response`
- Eval data: `experiments/unified_student_c1/data/processed/malrule_c1_clean/test_same_template.jsonl`

## Closed-Set Prediction

Same protocol in both rows: candidate target log-probability reranking over legal answers.

| test attention | n | accuracy | macro F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| masked late-fusion | 300 | 0.8800 | 0.7849 | 0.3002 | 0.0385 |
| open standard | 300 | 0.8667 | 0.7442 | 0.2843 | 0.0324 |

Read: opening history at test time does not collapse task performance. Accuracy drops only 1.3 points, while NLL/ECE slightly improve.

## State Swap, Clean Subset

Clean subset = pairs where `none` predicts receiver B's malrule answer. Main intervention is all-layer state replacement.

| test attention | clean n | condition | flip_to_A | stay_B | other | changed top-1 | mean delta p_A | mean delta p_B |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| masked | 89 | noop | 0.0000 | 1.0000 | 0.0000 | 0.0000 | -0.0003 | -0.0005 |
| masked | 89 | samemal | 0.0449 | 0.8652 | 0.0899 | 0.1348 | +0.0175 | -0.0304 |
| masked | 89 | random | 0.2921 | 0.5056 | 0.2022 | 0.4944 | +0.1413 | -0.2912 |
| masked | 89 | swap | 0.4944 | 0.2247 | 0.2809 | 0.7753 | +0.2881 | -0.4279 |
| open | 93 | noop | 0.0108 | 0.9892 | 0.0000 | 0.0108 | +0.0001 | +0.0005 |
| open | 93 | samemal | 0.0645 | 0.8710 | 0.0645 | 0.1290 | +0.0109 | -0.0237 |
| open | 93 | random | 0.1398 | 0.6989 | 0.1613 | 0.3011 | +0.0748 | -0.1740 |
| open | 93 | swap | 0.3011 | 0.4731 | 0.2258 | 0.5269 | +0.1877 | -0.2937 |

Read: open-attention swap remains directionally effective, but weaker than masked swap.

## Interpretation

The apparent contradiction is real: in open attention, the receiver history says "use B's error rule" while the patched state says "use A's error rule." The model partly follows direct history, so the intervention weakens.

However, the effect does not disappear. In the clean subset, open swap flips to A in 30.1% of pairs, compared with 14.0% for random and 6.5% for same-malrule control. Noop is nearly identical to none, so the hook pipeline is not the source of the effect.

This supports a nuanced C4 claim:

- late-fusion bottleneck training creates a state vector that is both readable and causally usable;
- direct history is still a competing pathway when test-time attention is opened;
- training with bottleneck and testing open can preserve task performance while retaining partial controllability.

## Files

- Open eval: `experiments/unified_student_c1/results/malrule_next_response_latefusion_s800_eval_same_OPEN_full300.json`
- Open swap: `experiments/unified_student_c1/results/c4_response_latefusion_s800_swap_response_OPEN_all_layers_n150_controls.json`
- Masked eval: `experiments/unified_student_c1/results/malrule_next_response_latefusion_s800_eval_same_masked_full300.json`
- Masked swap: `experiments/unified_student_c1/results/c4_response_latefusion_s800_swap_response_all_layers_n150_controls.json`
