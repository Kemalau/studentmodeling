# Bottleneck Mask Swap Smoke - 2026-07-10

## Goal

Test whether a bottleneck attention mask makes `<STUDENT_STATE>` causally control Malrule misconception diagnosis.

The mask blocks tokens after `<STUDENT_STATE>` from attending to the student history. The STATE token itself can still attend to history.

## Implementation

Updated:

- `experiments/unified_student_c1/scripts/train_c1_lora.py`
  - Added `--bottleneck-mask`.
  - Builds a 4D additive causal mask.
  - Masks history columns for all positions after STATE.
  - Leaves the STATE row's access to history unmasked.
  - Forces `attn_implementation="eager"` when the bottleneck mask is enabled.
  - Includes a startup assertion on the first training example.

- `experiments/unified_student_c1/scripts/state_swap_misconception_smoke.py`
  - Added `--bottleneck-mask`.
  - Uses the same 4D mask for STATE capture and candidate likelihood scoring.

0.6B sanity training passed before running 32B.

## Bottleneck Model

Adapter:

`experiments/unified_student_c1/results/qwen3_32b_single_malrule_identity_state_token_bottleneck_qlora_s60_len1536/final_adapter`

Setup:

- Base: `/data1/liyuejia/models/Qwen3-32B`
- Task: `malrule_identity`
- Data: `experiments/unified_student_c1/data/processed/malrule_c1_clean/train.jsonl`
- Steps: 60
- `<STUDENT_STATE>` is a single token: `[151669]`
- Startup mask assertion passed.

Training loss:

- step 10: 0.6754
- step 20: 0.1372
- step 30: 0.1208
- step 40: 0.0942
- step 50: 0.1078
- step 60: 0.1191

## Swap Result

Result file:

`experiments/unified_student_c1/results/state_swap_misconception_bottleneck_identity_all_layers_n20.json`

Configuration:

- test split: `test_same_template`
- pairs: 20
- all 64 layers patched
- closed-set likelihood scoring over misconception labels
- bottleneck mask enabled during scoring

| Condition | stay_B | flip_to_A | other | mean p(A) | mean p(B) | mean A-B margin |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.35 | 0.00 | 0.65 | 0.0800 | 0.2662 | -1.8106 |
| noop | 0.40 | 0.00 | 0.60 | 0.0789 | 0.2730 | -1.8511 |
| swap | 0.35 | 0.00 | 0.65 | 0.0818 | 0.2680 | -1.7901 |

Extra checks:

- swap changed top-1 prediction: 0 / 20
- noop changed top-1 prediction: 3 / 20
- swap-induced mean delta p(A): +0.0018
- swap-induced mean delta p(B): +0.0018
- swap-induced mean delta A-B margin: +0.0205

## Comparison To Marker-Only Swap

Marker-only all-layer swap:

`experiments/unified_student_c1/results/state_swap_misconception_strict_identity_all_layers_n20.json`

| Model | none stay_B | swap flip_to_A | swap changed top-1 | noop changed top-1 |
|---|---:|---:|---:|---:|
| marker-only | 0.55 | 0.10 | 0 / 20 | 0 / 20 |
| bottleneck | 0.35 | 0.00 | 0 / 20 | 3 / 20 |

## Interpretation

This bottleneck smoke did not produce causal steering.

Even with the history blocked from query/target tokens, swapping all STATE-layer activations did not flip the diagnosis toward the donor misconception. The top-1 prediction under swap was unchanged for all 20 pairs.

However, this is not yet a decisive negative result:

- The 60-step bottleneck model is weaker at baseline than the marker-only identity model on these swap pairs.
- `none` predicts the recipient misconception only 35% of the time.
- `noop` changes top-1 in 3/20 pairs, so the all-layer patching pipeline under 4D mask is slightly numerically fragile.

The right read is:

1. The bottleneck mask implementation runs and enforces the intended visibility constraint.
2. This short 32B bottleneck diagnosis smoke did not show state-swap control.
3. A stronger bottleneck model or response-prediction swap is needed before making a hard C4 decision.

## Next Step

If continuing C4:

- train the bottleneck variant longer,
- evaluate its same-template identity accuracy under the same mask,
- restrict swap analysis to pairs where `none` correctly predicts B,
- then rerun all-layer swap.

