# C4 Response Bottleneck S800 Check

Date: 2026-07-11

## Setup

- Base: `/data1/liyuejia/models/Qwen3-32B`
- Adapter: `experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_bottleneck_qlora_s800_len1536/final_adapter`
- Task: `malrule_next_response`
- Training data: 3,000 response-only MalruleLib examples
- Training: QLoRA, bottleneck mask, `<STUDENT_STATE>`, max length 1536, max steps 800
- Note: this is still a fast QLoRA pilot, not the final bf16 C4 run.

## Training

S800 completed successfully.

- Runtime: 4,417 s
- Effective epochs: 1.067
- Train loss: 0.09358
- Late losses were mostly around 0.04-0.07.

This addresses the main weakness of the S200 run: S200 had only seen about 0.27 epoch and reached same-template response accuracy 0.5917.

## Response Prediction Gate

File:
`experiments/unified_student_c1/results/malrule_next_response_bottleneck_s800_eval_same_masked_full300.json`

Full same-template response eval, bottleneck mask enabled, candidate target-logprob scoring:

| Model | n | Accuracy | Macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| S200 bottleneck | 120 | 0.5917 | 0.4304 | 0.9577 | 0.1216 |
| S800 bottleneck | 300 | 0.7200 | 0.5687 | 0.6464 | 0.0695 |

Baselines on the same S800 full-300 eval:

| Baseline | Accuracy |
|---|---:|
| random uniform over candidates | 0.3847 |
| global majority | 0.4167 |

Conclusion: the response model now passes the planned ~0.7 behavior gate. The earlier 0.59 was largely undertraining, not an intrinsic ceiling.

## 4a Probe

File:
`experiments/unified_student_c1/results/c4_response_bottleneck_s800_probe_malrule_layers_smoke.json`

Probe target: malrule identity, 22 classes, random level 1/22 = 0.0455.

Rows:

- train: 1,075
- same-template: 239
- cross-template: 253

Best STATE probe:

| Layer | Same Acc | Cross Acc | Same Macro-F1 | Cross Macro-F1 |
|---:|---:|---:|---:|---:|
| 48 | 0.8870 | 0.8696 | 0.8888 | 0.8501 |
| 56 | 0.8787 | 0.8696 | 0.8845 | 0.8628 |
| 63 | 0.9038 | 0.8656 | 0.9099 | 0.8602 |

Best history-last position control:

| Layer | Same Acc | Cross Acc | Same Macro-F1 | Cross Macro-F1 |
|---:|---:|---:|---:|---:|
| 40 | 0.8243 | 0.8379 | 0.8313 | 0.8193 |

Conclusion: 4a is stronger after S800. Malrule identity is highly linearly readable across templates from the STATE activation. However, history-last is also strong, so current evidence supports "readable distributed history-derived representation" more than "STATE is uniquely special."

## 4b State Swap

File:
`experiments/unified_student_c1/results/c4_response_bottleneck_s800_swap_response_all_layers_n150_controls.json`

Setup:

- 150 legal ordered response-level donor/recipient pairs
- all 64 layers patched at `<STUDENT_STATE>`
- five conditions: none, noop, random, samemal, swap
- classification by executable answer match

All-pair distribution:

| Condition | flip_to_A | stay_B | other | mean p(A) | mean p(B) |
|---|---:|---:|---:|---:|---:|
| none | 40/150 | 54/150 | 56/150 | 0.2660 | 0.2903 |
| noop | 39/150 | 51/150 | 60/150 | 0.2663 | 0.2908 |
| samemal | 41/150 | 54/150 | 55/150 | 0.2659 | 0.2908 |
| random | 45/150 | 59/150 | 46/150 | 0.2396 | 0.2669 |
| swap | 40/150 | 52/150 | 58/150 | 0.2672 | 0.2906 |

Clean subset where `none == stay_B`:

| Condition | n | flip_to_A | stay_B | other | changed top-1 | mean delta p(A) | mean delta p(B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| noop | 54 | 3 | 49 | 2 | 5 | +0.0028 | -0.0023 |
| samemal | 54 | 2 | 51 | 1 | 3 | +0.0009 | -0.0010 |
| random | 54 | 4 | 47 | 3 | 7 | -0.0116 | -0.0504 |
| swap | 54 | 2 | 51 | 1 | 3 | +0.0034 | -0.0028 |

Conclusion: with the behavior gate now passed, all-layer STATE swap still does not show a directional causal flip. In the clean subset, swap matches samemal exactly on flip_to_A (2/54) and changed top-1 (3/54), while random causes more disruption but not a targeted A-style move.

## Current Interpretation

The current result is:

- 4a: strong positive. The model has a linearly readable, cross-template malrule representation.
- 4b: negative under this all-layer STATE swap setup. The readable representation is not yet shown to be a causal control knob for response style.

This is the "readable but not controllable" pattern unless a later intervention design changes the result. The remaining caveat is that the clean subset is 54, not the desired >=100, because only 54/150 sampled pairs had `none == stay_B`. A larger pair run or recipient-filtered sampler would make the negative 4b conclusion tighter.
