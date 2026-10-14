# C4 Late-Fusion Bottleneck Smoke Report - 2026-07-11

## Setup

- Base model: `/data1/liyuejia/models/Qwen3-32B`
- Adapter:
  `experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter`
- Task: `malrule_next_response`
- Data: `experiments/unified_student_c1/data/processed/malrule_c1_clean/`
- Max length: 1536
- Training: 800 steps, 4-bit QLoRA, state token enabled
- Mask style: `late_fusion`

Late-fusion mask semantics:

- `<STUDENT_STATE>` can attend instruction and history.
- Question tokens after `<STUDENT_STATE>` cannot attend history.
- Question tokens also cannot attend `<STUDENT_STATE>`.
- Target/answer tokens can attend `<STUDENT_STATE>`.

The goal is to stop the question segment from absorbing recipient state before
the intervention, while still forcing the answer segment to use the state token.

## Prediction Gate

Evaluation on `test_same_template`, 300 examples.

Important scoring caveat: this is closed-set candidate scoring, not free-form
generation. The prompt lists valid outputs, and the evaluator scores each
candidate target by log probability, then chooses the highest-scoring
candidate. The target does not appear after `Answer:`, so this is not direct
label leakage, but the 0.88 number should be read as candidate reranking
accuracy.

| model variant | response accuracy | macro F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| standard bottleneck S800 | 0.7200 | 0.5687 | 0.6464 | 0.0695 |
| late-fusion bottleneck S800 | 0.8800 | 0.7849 | 0.3002 | 0.0385 |

The late-fusion version clears the behavior gate comfortably and is much
stronger than the previous standard bottleneck run.

## Linear Probe

Probe target: malrule identity, 22 labels.
Random level: `1 / 22 = 0.0455`.

Best cross-template linear probe results:

| position | best layer | same accuracy | cross accuracy | cross macro F1 |
|---|---:|---:|---:|---:|
| `<STUDENT_STATE>` | 48 | 0.8828 | 0.9051 | 0.8946 |
| history last token | 48 | 0.8410 | 0.8617 | 0.8482 |

Layer sweep, cross accuracy:

| position | L0 | L8 | L16 | L24 | L32 | L40 | L48 | L56 | L63 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state | 0.6877 | 0.7312 | 0.7668 | 0.7510 | 0.7549 | 0.7787 | 0.9051 | 0.8854 | 0.8656 |
| history_last | 0.7194 | 0.7668 | 0.7628 | 0.7905 | 0.7787 | 0.8221 | 0.8617 | 0.8063 | 0.7984 |

Read: the malrule representation remains highly linearly readable after
late-fusion training. The state position is better than the history-last
control at the best layer, but the control is also strong, so a paper-grade
run should include stricter position and masking controls.

## State Swap

Configuration:

- 150 ordered donor-recipient pairs from `test_same_template`
- Same family, different malrule
- Full-layer replacement, layers 0-63
- Conditions: `none`, `noop`, `samemal`, `random`, `swap`
- Output file:
  `experiments/unified_student_c1/results/c4_response_latefusion_s800_swap_response_all_layers_n150_controls.json`

All pairs:

| condition | flip_to_A | stay_B | other | mean p_A | mean p_B | mean margin A-B |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.1133 | 0.5933 | 0.2933 | 0.1711 | 0.4925 | -1.4587 |
| noop | 0.1133 | 0.6267 | 0.2600 | 0.1710 | 0.4940 | -1.4677 |
| samemal | 0.1533 | 0.5933 | 0.2533 | 0.1896 | 0.4733 | -1.2632 |
| random | 0.2933 | 0.3667 | 0.3400 | 0.2607 | 0.3170 | -0.2536 |
| swap | 0.4733 | 0.2267 | 0.3000 | 0.4053 | 0.2222 | 0.9017 |

Clean subset: only pairs where the no-intervention output stayed with the
recipient malrule (`none = stay_B`), n = 89.

| condition | flip_to_A | stay_B | other | changed top-1 | delta p_A | delta p_B | delta margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| noop | 0.0000 | 1.0000 | 0.0000 | 0.0000 | -0.0003 | -0.0005 | 0.0002 |
| samemal | 0.0449 | 0.8652 | 0.0899 | 0.1348 | 0.0175 | -0.0304 | 0.0479 |
| random | 0.2921 | 0.5056 | 0.2022 | 0.4944 | 0.1413 | -0.2912 | 0.4325 |
| swap | 0.4944 | 0.2247 | 0.2809 | 0.7753 | 0.2881 | -0.4279 | 0.7160 |

Read:

- `noop` is exactly stable on the clean subset, so the hook and mask pipeline
  pass the main self-check.
- `swap` is strongly directional: it flips to the donor answer on 49.4% of
  clean pairs and shifts the A-vs-B probability margin by +0.716.
- `samemal` barely flips, so the effect is specific to donor malrule identity,
  not just any other student vector.
- `random` is disruptive and sometimes lands on A, but it is much weaker than
  directed swap in both top-1 flip rate and probability margin.

## Interpretation

This run is the first clean positive 4b result in the current C4 pilot.

The previous standard bottleneck model had readable states but weak or absent
swap effects. Late fusion changes the setup in the right way: it improves
behavioral prediction and makes the state token causally useful for generation.

Current claim supported by this smoke run:

> In a late-fusion bottleneck model, the `<STUDENT_STATE>` activation contains a
> linearly readable malrule representation and full-layer replacement of that
> activation causally shifts predicted response style toward a donor student's
> malrule.

Remaining before paper-grade status:

- Repeat with the full default train size, not only the S800 smoke setting.
- Run at least two more seeds.
- Add shuffled-label probe control to this exact report.
- Run single-layer and layer-band swap scans.
- Run cross-template swap if the pair construction supports enough legal pairs.
- Compare against the marker and standard bottleneck variants in one table.

## Quick Audit Notes

Checks performed after the run:

- The late-fusion mask passes a toy invariant check:
  - state row can attend history;
  - question rows cannot attend history;
  - question rows cannot attend state;
  - target rows can attend state;
  - target rows still cannot attend history.
- In the evaluation file, `target` never appears after `Answer:` in the prompt.
- The true target does appear in the `Valid outputs` list for every checked
  example, by design. This means the metric is closed-set reranking.
- `noop` has zero top-1 changes on the clean swap subset, which is the strongest
  available check that the hook itself is not perturbing predictions.

Plausible reason late fusion improves both prediction and controllability:

- The standard bottleneck let question tokens attend to the recipient state, so
  recipient information could be copied into question-token activations before
  the state patch. Replacing only the state vector then left a lot of recipient
  information elsewhere in the prompt.
- Late fusion keeps question tokens problem-only and forces the answer segment
  to combine the problem representation with the state vector at prediction
  time. This makes the state vector a cleaner causal handle.
- The same pressure can also make the malrule direction more linearly readable,
  explaining the stronger probe result.
