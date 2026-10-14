# Method-Small Fully-Open Adapter C4 Probe and State-Swap Audit

Date: 2026-07-24

## 1. Question and non-goals

Question: after one epoch of fully-open multitask training, does the hidden state
at the prompt's `<STUDENT_STATE>` position (encoded during training as the
existing single token `<|vision_pad|>`, token ID 151654) contain a readable
student-malrule representation, and can replacing that state causally redirect
the predicted response toward a donor student's malrule?

This experiment does not discover neurons, optimize refusal or safety behavior,
or show that the state token is a sufficient student-state bottleneck. The
adapter was trained with ordinary causal attention and no bottleneck mask, so
answer tokens can read the full student history directly.

## 2. Sample construction

### Model and data

- Base model: Qwen3-32B, loaded in NF4 4-bit mode.
- Adapter: one-epoch `method_small_v1` fully-open QLoRA adapter,
  `projects/v-yuejiali/amlt-results/7215295067.10143-255b28af-1e91/model/final_adapter`.
- Adapter training seed: 42; 12,000 training rows, including 700 Malrule rows
  for each of correctness, response, and diagnosis.
- C4 source: synthetic executable Malrule data with four history trials per
  student, 21 non-NONE malrules plus `NONE`, across subtraction, decimals, and
  fractions.
- Probe task and swap task: `malrule_next_response` only.
- Maximum sequence length: 1536. The swap script rejects over-length prompts;
  it does not silently truncate them.
- No free decoding is used. Responses are scored as teacher-forced legal
  candidates supplied by the synthetic task.

The C4 same- and cross-template raw IDs have zero overlap with the adapter's
`method_small_v1` train split. The adapter train split contains 700 of the 3,000
raw Malrule train students. The probe classifier is fitted only on the Malrule
train split; same- and cross-template rows are held out from that fit.

### Probe sampling

The script stratifies by malrule label after filtering to
`malrule_next_response`:

| Split | Seed | Requested per label | Actual rows | Labels |
|---|---:|---:|---:|---:|
| train | 42 | 50 | 1,075 | 22 |
| same-template | 43 | 15 | 239 | 22 |
| cross-template | 44 | 15 | 253 | 22 |

Short labels account for the difference between requested and actual counts.
The random-class level is `1/22 = 0.04545`.

### Swap-pair sampling

The swap pool contains the 300 same-template `malrule_next_response` rows.
`NONE` students are excluded. Ordered donor/recipient candidates must:

1. be different examples;
2. have the same mathematical family but different malrules;
3. have distinct donor and recipient responses;
4. contain both responses in the recipient's legal candidate set; and
5. have neither response equal the correct mathematical answer.

All valid ordered pairs are enumerated, shuffled with seed 42, and the first
150 are retained. Same-malrule controls use seed `42 + 1009 = 1051`.
All 150 pairs have none, noop, same-malrule, norm-matched random, and directed
swap conditions. The clean subset contains the 94 pairs for which the
no-intervention prediction equals the recipient response.

### Concrete safe sample

One held-out response example shows four fraction-comparison errors consistent
with a denominator-comparison malrule, followed by `<STUDENT_STATE>` and a new
comparison between `1/14` and `1/10`. Legal outputs are `1/10`, `1/14`, and
`equal`; the synthetic student target is `1/14`. The loss/scoring code sees the
rendered history and query, then teacher-forces each legal tagged response.

## 3. Coordinate, objective, and formulas

### Linear readout

A forward hook records the output of each selected decoder block at:

- the `<|vision_pad|>` state-token position; and
- the last token immediately before that position (`history_last`) as a control.

The fixed layer set is `0, 8, 16, 24, 32, 40, 48, 56, 63` out of 64 layers.
For activation vector `h`, train-split mean `mu`, and standard deviation
`sigma`, the standardized feature is

```text
z = (h - mu) / sigma
p(y | h) = softmax(W z + b)
L_probe = -(1/N) sum_i log p(y_i | h_i)
```

Each probe uses AdamW for 250 full-batch epochs, learning rate 0.1, weight decay
`1e-4`, and seed 42. Fitting ran on CUDA; the activation model was removed
before fitting. Accuracy and macro-F1 are evaluated separately on same- and
cross-template splits. This objective measures linear malrule decodability; it
does not optimize response accuracy or causal control.

### Candidate scoring

For candidate response `c` with target tokens `x_t`, the swap evaluator uses

```text
s(c) = sum_{t in target(c)} log p(x_t | prompt, x_<t)
p_c = softmax_c s(c)
margin(A,B) = s(A) - s(B)
```

The top-scoring legal candidate is the prediction.

## 4. Arithmetic example

There is no gradient-attribution sign or neuron-ranking score in this
experiment. An illustrative, not measured, two-class probe example is:

```text
z = [1, -1]
W_A = [1, 0], W_B = [0, 1]
logits = [1, -1]
p(A) = exp(1) / (exp(1) + exp(-1)) = 0.881
```

This only illustrates linear readout. It cannot establish that changing `z`
changes the model's response; that requires the separate swap experiment.

## 5. Real measured coordinate example

The strongest state-position cross-template readout is layer 56:

- same-template accuracy / macro-F1: `0.7280 / 0.7372`;
- cross-template accuracy / macro-F1: `0.7747 / 0.7600`.

Layer 63 ties its cross accuracy (`0.7747`) and has cross macro-F1 `0.7601`.
These are descriptive best layers, not intervention-selected candidates. The
history-last control is stronger: layer 48 reaches cross accuracy `0.8893` and
macro-F1 `0.8818`.

## 6. Selection funnel

There is no neuron selection or post-hoc top-K intervention:

```text
64 decoder layers
-> 9 fixed stride/checkpoint layers for readout
-> report every state and history-last result
-> descriptive best layer only (no selection used by swap)

64 decoder layers
-> all 64 fixed a priori for state replacement
-> 150 fixed ordered pairs
-> 94 clean no-intervention-stay_B pairs
-> compare all five conditions
```

The swap does not use the probe's best layer and does not tune on probe results.

## 7. Held-out causal validation

The same-template swap split overlaps the probe's same-template evaluation
split, but it is not used to select layers, doses, or pairs after observing
probe performance. Cross-template probe rows are not used in swap. This is a
development C4 test, not a final disjoint multi-seed confirmation.

## 8. Intervention definition and dose semantics

Let `t_s` be the `<|vision_pad|>` position and `H_l` the output of decoder
layer `l`. For every layer `l = 0..63`, directed replacement is

```text
H'_{l, t_s, :}(recipient) = H_{l, t_s, :}(donor)
```

Conditions:

- `none`: no hook;
- `noop`: replace with the recipient's own captured vector;
- `samemal`: replace with another student's vector from the recipient's family
  and malrule;
- `random`: Gaussian vector normalized to the recipient vector's L2 norm at
  each layer;
- `swap`: replace with the different-malrule donor vector.

There is no scalar dose. This is full vector replacement at one token position
across all layers under standard open attention.

## 9. Evaluation task and rubric

The task is closed-set exact response prediction. The donor response is A and
the recipient response is B. Outcomes are:

- `flip_to_A`: top-scoring candidate equals A;
- `stay_B`: top-scoring candidate equals B;
- `correct`: top-scoring candidate is the mathematically correct answer; or
- `other`: another legal candidate.

No natural-language judge, free generation, or semantic matching is involved.

## 10. Aggregate and paired results

### Linear probe

| Position | Best cross layer | Same acc. | Same macro-F1 | Cross acc. | Cross macro-F1 |
|---|---:|---:|---:|---:|---:|
| state token | 56 | 0.7280 | 0.7372 | 0.7747 | 0.7600 |
| history-last control | 48 | 0.8912 | 0.8936 | 0.8893 | 0.8818 |

The state position is strongly readable relative to chance, but it is not the
strongest local carrier; history-last is substantially better.

Full accuracy sweep:

| Layer | state same | state cross | history-last same | history-last cross |
|---:|---:|---:|---:|---:|
| 0 | 0.5816 | 0.5968 | 0.7113 | 0.7391 |
| 8 | 0.5439 | 0.6087 | 0.7197 | 0.7431 |
| 16 | 0.5816 | 0.6364 | 0.7280 | 0.7826 |
| 24 | 0.6318 | 0.6522 | 0.7531 | 0.7589 |
| 32 | 0.6276 | 0.6561 | 0.7657 | 0.7668 |
| 40 | 0.6360 | 0.6522 | 0.7992 | 0.7945 |
| 48 | 0.6444 | 0.7273 | 0.8912 | 0.8893 |
| 56 | 0.7280 | 0.7747 | 0.8661 | 0.8775 |
| 63 | 0.7448 | 0.7747 | 0.8033 | 0.8854 |

### All 150 swap pairs

| Condition | flip_to_A | stay_B | other | mean p_A | mean p_B | mean score margin A-B |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.1533 | 0.6267 | 0.2200 | 0.1750 | 0.4869 | -1.6181 |
| noop | 0.1467 | 0.6400 | 0.2133 | 0.1747 | 0.4862 | -1.6125 |
| same-malrule | 0.1533 | 0.6333 | 0.2133 | 0.1738 | 0.4877 | -1.6188 |
| random | 0.1533 | 0.6400 | 0.2067 | 0.1749 | 0.4812 | -1.5736 |
| directed swap | 0.1533 | 0.6333 | 0.2133 | 0.1757 | 0.4863 | -1.6152 |

### Clean subset (`none = stay_B`, n=94)

| Condition | flip_to_A | stay_B | changed top-1 | delta p_A | delta p_B | delta margin A-B |
|---|---:|---:|---:|---:|---:|---:|
| noop | 0.0000 | 1.0000 | 0.0000 | +0.0006 | -0.0021 | +0.0102 |
| same-malrule | 0.0000 | 1.0000 | 0.0000 | -0.0005 | -0.0010 | +0.0049 |
| random | 0.0213 | 0.9787 | 0.0213 | +0.0043 | -0.0139 | +0.0831 |
| directed swap | 0.0106 | 0.9894 | 0.0106 | +0.0021 | -0.0037 | +0.0169 |

Directed swap is weaker than the norm-matched random control and is close to
noop/same-malrule on probability and score shifts. It provides no evidence of
directional causal control.

## 11. Representative outcomes

- The sole clean directed-swap flip is subtraction pair 23, from
  `smaller_from_larger` response `51` toward `borrow_from_bottom` response
  `269`. Its score margin moves from `-0.0942` to `+0.0628`, but the random
  control also flips this exact pair, so it is not selective evidence.
- In subtraction pair 137, random flips from recipient `568` to donor `632`,
  while directed swap stays at `568`.
- In fraction pair 1, all conditions stay at recipient response `7/10`; the
  directed donor response `7` remains near probability `0.0076`.
- No clean pair produces an invalid or corrupted output because scoring is
  restricted to legal candidates.

## 12. Mechanism conclusion, limitations, and falsifiers

The narrow supported conclusion is:

> Fully-open multitask training leaves malrule identity linearly readable at
> the state-marker position, but the representation is more readable at the
> adjacent history control and all-layer marker replacement does not causally
> redirect response behavior.

This is consistent with direct-history pathways carrying student information
without concentrating it into the marker. It explains why the fully-open model
can perform well on C1/T2 while failing the C4 controllability test.

Limitations and alternative explanations:

- one adapter seed and synthetic Malrule data only;
- closed-set candidate scoring rather than free generation;
- same-template causal validation is not disjoint from same-template probe
  evaluation, though no probe layer was selected for swap;
- all-layer replacement may overwrite or wash out a narrower causal band;
- only the marker coordinate was patched, while query-token or residual-path
  representations may be causal;
- the tokenizer vocabulary was resized from padded embedding size 151936 to
  tokenizer size 151669 before adapter loading; the alias ID is within range,
  but this implementation detail should remain fixed in comparisons.

The all-layer negative would be weakened if a preregistered single-layer or
layer-band scan produced a directed swap effect that exceeded noop,
same-malrule, and norm-matched random controls on held-out pairs. A late-fusion
model remains the positive control and should retain substantially larger
directional effects under the same evaluator.

## 13. Reproduction files and commands

Experiment: `method-small-fullopen-c4-20260724`, Feeds Standard SLA,
`80G1-A100`, image
`amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`,
UAI `rankfun_aml`.

- Swap job: `magenta_button_t96mzfn3bk` (Pass, 20 minutes).
- GPU-fit probe job: `tidy_fork_f396v0c6h0` (Pass, 21 minutes).
- Initial CPU-fit probe: `kind_lobster_wwbz8jmwyb` (intentionally killed after
  32 minutes because fitting was projected to waste roughly two A100-hours;
  no result from it is used).
- AMLT config: `amlt/feeds_method_small_fullopen_c4_eval.yaml`.
- Probe script: `scripts/probe_student_state_smoke.py`.
- Swap script: `scripts/state_swap_misconception_smoke.py`.
- Local probe artifact:
  `results/method_small_fullopen_c4_download/method-small-fullopen-c4-20260724/qwen3-32b-method-small-fullopen-c4-probe-gpufit/c4/probe.json`.
- Local swap artifact:
  `results/method_small_fullopen_c4_download/method-small-fullopen-c4-20260724/qwen3-32b-method-small-fullopen-c4-swap/c4/swap.json`.
- Remote probe result prefix:
  `projects/v-yuejiali/amlt-results/7215133986.65189-c2c5cbb9-e40a`.
- Remote swap result prefix:
  `projects/v-yuejiali/amlt-results/7215136183.09675-a99e8e17-0334`.

Input SHA256 values recorded by the jobs:

```text
train.jsonl                6fa5f4160e4a096db6228eb165136d927d0a72bae5a8e5dce9df5bda71a73dae
test_same_template.jsonl   079f7634cc54dfd55dfc234503bb5925ab734b1ca74aa2f2a9a17cc4375a2c2f
test_cross_template.jsonl  405446c371185d23f06b4a0530c8ef4da4d1ac9addb696180f83bdf092402779
```

Submission command:

```bash
amlt run amlt/feeds_method_small_fullopen_c4_eval.yaml \
  method-small-fullopen-c4-20260724 -t Feeds
```
