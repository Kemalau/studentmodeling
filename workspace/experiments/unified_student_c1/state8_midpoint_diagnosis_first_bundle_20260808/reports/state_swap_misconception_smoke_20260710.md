# Misconception State-Swap Smoke - 2026-07-10

## Question

If we swap another student's `<STUDENT_STATE>` activation into the current student's forward pass, does the model's misconception diagnosis change toward the donor student's misconception?

## Important Scope

This is a mechanics smoke test on a diagnosis model directly trained to output Malrule identity:

`<misconception>{malrule_id}</misconception>`

So this is not the final C4 "emergent state from response-only training" proof. It answers a narrower question: does a marker-only diagnosis model causally use the STATE activation for its diagnosis output?

## Model

Trained a strict single-token STATE Malrule identity adapter:

`experiments/unified_student_c1/results/qwen3_32b_single_malrule_identity_state_token_qlora_s60_len1536/final_adapter`

Training setup:

- Base: `/data1/liyuejia/models/Qwen3-32B`
- Data: `malrule_c1_clean/train.jsonl`
- Task filter: `malrule_identity`
- Steps: 60
- QLoRA 4-bit
- `<STUDENT_STATE>` registered as one special token
- Tokenizer check: `<STUDENT_STATE>` -> `[151669]`

The adapter is large because `embed_tokens` is saved after resizing.

## Diagnosis Accuracy Sanity

Evaluation file:

`experiments/unified_student_c1/results/malrule_identity_state_token_s60_eval_same_n120.json`

Same-template `malrule_identity`, first 120 examples:

- Model accuracy: 0.700
- Model macro-F1: 0.617
- Question-majority baseline accuracy: 0.458
- Random accuracy: 0.045

So the model did learn the diagnosis task moderately well.

## Swap Method

Script:

`experiments/unified_student_c1/scripts/state_swap_misconception_smoke.py`

For each ordered pair:

- Donor A and recipient B are same-family, different-malrule, non-NONE students.
- Capture A and B's STATE activations at all 64 decoder block outputs.
- Score B's candidate misconception labels by log-likelihood under three conditions:
  - `none`: normal B forward pass
  - `noop`: patch B's own STATE activations back into B
  - `swap`: patch A's STATE activations into B

The scoring is closed-set over candidate targets, not free-form generation.

## Main All-Layer Swap Result

Result file:

`experiments/unified_student_c1/results/state_swap_misconception_strict_identity_all_layers_n20.json`

Pairs: 20

| Condition | stay_B | flip_to_A | other | mean p(A) | mean p(B) | mean A-B margin |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.55 | 0.10 | 0.35 | 0.0906 | 0.4884 | -3.1238 |
| noop | 0.55 | 0.10 | 0.35 | 0.0906 | 0.4899 | -3.1547 |
| swap | 0.55 | 0.10 | 0.35 | 0.0889 | 0.4916 | -3.1457 |

Additional check:

- Number of pairs whose top-1 prediction changed under swap: 0 / 20
- Mean delta p(A), swap - none: -0.0017
- Mean delta p(B), swap - none: +0.0032

## Tiny Hook Sanity

Tiny two-layer sanity result:

`experiments/unified_student_c1/results/state_swap_misconception_strict_identity_tiny_layers56_63.json`

`noop` matched `none`, which supports that the patching pipeline itself is not scrambling the forward pass.

## Interpretation

The result is negative for causal control in the current marker-only diagnosis model.

Swapping the donor STATE vector into the recipient did not move the misconception diagnosis toward the donor. The exact top-1 predictions were unchanged for all 20 all-layer swap pairs.

This is consistent with the marker-only concern: because question/target tokens can still attend directly to the full history, the model can diagnose from the raw history path and ignore the marker as a causal bottleneck.

In short:

- Readability exists from previous probes.
- Diagnosis ability exists in this identity model.
- But marker-only STATE swapping does not causally steer diagnosis.

## Next Step

The next experiment should be the bottleneck variant:

- train with `<STUDENT_STATE>` as a single token,
- prevent question/target tokens from attending to history,
- force information through STATE,
- rerun this same swap script.

That is the setup where a positive swap effect should be expected if STATE is truly being used.

