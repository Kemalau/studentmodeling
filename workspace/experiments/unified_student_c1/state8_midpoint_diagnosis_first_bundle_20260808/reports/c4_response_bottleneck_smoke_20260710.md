# C4 Response-Only Bottleneck Smoke — 2026-07-10

## Status Note

This is a smoke test, not a specification-complete 4b result.

Do not cite the swap result below as evidence that 4b fails. It is underpowered and missing the full control set required by the C4 spec:

- only 30 sampled pairs;
- only 7 pairs where `none = stay_B`;
- no `random` or `samemal` controls;
- response EM is 0.5917, below the intended 0.7 gate;
- pair sampling used at most one donor per recipient, even though the data contains many more legal ordered pairs.

The only robust positive claim from this run is 4a: malrule identity is strongly cross-template decodable from STATE after response-only training.

## 目的

这轮按更接近 C4 科学 claim 的设定跑:

- 训练任务只用 `malrule_next_response`;
- target 只有 `<predicted_response>...`;
- 不直接监督 `malrule_id`;
- 训练、评测、probe、swap 都使用 bottleneck mask;
- 读数分成:
  - 4a: 从 `<STUDENT_STATE>` 线性 probe 读 malrule identity;
  - 4b: 把 donor 的 STATE swap 到 recipient,看预测答案是否变成 donor malrule 在 recipient 查询题上的可执行答案。

## 代码改动

- `experiments/unified_student_c1/scripts/probe_student_state_smoke.py`
  - 增加 `--bottleneck-mask`;
  - activation collection 使用 4D bottleneck attention mask;
  - bottleneck 下使用 eager attention。
- `experiments/unified_student_c1/scripts/state_swap_misconception_smoke.py`
  - 增加 `--swap-target response`;
  - response swap 的 `flip_to_A` 定义为:生成答案等于 donor malrule 在 recipient 查询题上的答案;
  - pair 过滤要求 donor answer、recipient answer、correct answer 三者可区分。

## 模型

- base: `/data1/liyuejia/models/Qwen3-32B`
- adapter: `experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_bottleneck_qlora_s200_len1536/final_adapter`
- task: `malrule_next_response`
- train samples: 3000
- max steps: 200
- LoRA rank/alpha: 8/16
- `<STUDENT_STATE>` special token 加入并训练 embedding
- bottleneck mask: on

训练 loss:

- step 10: 1.271
- step 20: 0.1592
- step 30: 0.06834
- step 100: 0.0834
- step 160: 0.04854
- step 200: 0.06419
- train loss overall: 0.1312

## Response Prediction Eval

文件:

`experiments/unified_student_c1/results/malrule_next_response_bottleneck_s200_eval_same_masked_n120.json`

同模板 masked candidate likelihood,120 条:

| model | accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| response-only bottleneck S200 | 0.5917 | 0.4304 | 0.9577 | 0.1216 |

结论:模型确实学到了答案预测,但不是很强。后面的 swap 读数要记住这一点:baseline 本身只有 7/30 个 pair 预测为 recipient answer。

## 4a Probe: Malrule 可读性

文件:

`experiments/unified_student_c1/results/c4_response_bottleneck_s200_probe_malrule_layers_smoke.json`

设置:

- train rows: 440
- same rows: 166
- cross rows: 171
- labels: 22
- random level: 1/22 = 0.0455
- layers: 0,8,16,24,32,40,48,56,63

STATE probe:

| layer | same acc | cross acc |
|---:|---:|---:|
| 0 | 0.6265 | 0.6257 |
| 8 | 0.6084 | 0.6550 |
| 16 | 0.6566 | 0.6608 |
| 24 | 0.6506 | 0.6667 |
| 32 | 0.6506 | 0.6842 |
| 40 | 0.5843 | 0.6023 |
| 48 | 0.6687 | 0.6959 |
| 56 | 0.6386 | 0.6959 |
| 63 | 0.6747 | 0.7895 |

Best STATE cross:

- layer 63
- cross accuracy: 0.7895
- cross macro-F1: 0.7841
- same accuracy: 0.6747

Position control, history-last:

- best cross accuracy: 0.7661 at layer 40
- best same accuracy: 0.7470 at layer 48

Interpretation:

4a smoke is positive: even though the model was trained only to predict answers, malrule identity is strongly linearly decodable from STATE and transfers across templates. However, history-last is also strong, so this is not yet evidence that STATE is uniquely special; it is evidence that the model/run contains a readable malrule representation.

## 4b Swap: Response-Level Causal Test

### All-layer STATE swap

文件:

`experiments/unified_student_c1/results/c4_response_bottleneck_s200_swap_response_all_layers_n30.json`

30 response-level legal pairs.

| condition | stay_B | flip_to_A | correct | other | mean p(A) | mean p(B) |
|---|---:|---:|---:|---:|---:|---:|
| none | 7 | 7 | 10 | 6 | 0.2325 | 0.2297 |
| noop | 9 | 6 | 9 | 6 | 0.2327 | 0.2293 |
| swap | 7 | 7 | 10 | 6 | 0.2328 | 0.2310 |

Clean subset: only pairs where `none = stay_B`.

| subset | n | swap flip_to_A | swap changed top-1 | delta p(A) | delta margin A-B |
|---|---:|---:|---:|---:|---:|
| none stay_B | 7 | 0 | 0 | +0.000672 | +0.014795 |

Smoke observation:

All-layer response swap did not flip any of the 7 `none = stay_B` pairs. This is not enough to conclude non-controllability. The apparent 7/30 flip_to_A is already present in `none`, so it is not caused by swap in this small run.

### Single-layer STATE swap

文件:

`experiments/unified_student_c1/results/c4_response_bottleneck_s200_swap_response_single_layers_n30.json`

Clean subset again means `none = stay_B`.

| layer | clean n | clean swap flip | clean swap changed | delta p(A) | delta margin A-B |
|---:|---:|---:|---:|---:|---:|
| L0 | 7 | 0 | 0 | +0.002373 | +0.019516 |
| L16 | 7 | 0 | 0 | +0.004161 | +0.022444 |
| L32 | 7 | 0 | 0 | +0.004122 | +0.033547 |
| L48 | 7 | 0 | 0 | -0.000644 | +0.016185 |
| L56 | 7 | 0 | 0 | +0.000723 | +0.019410 |
| L63 | 7 | 0 | 0 | +0.000000 | +0.000000 |

Smoke observation:

No tested single layer caused `stay_B -> flip_to_A` in this small sample. Some layers slightly moved the A-B margin, but the effect was tiny and never changed top-1.

## Current C4 Readout

This smoke supports:

- 4a readable: yes, strongly. Response-only training still yields a cross-template decodable malrule representation in STATE.
- 4b controllable: not adjudicated. This run is underpowered and lacks required controls.

Best current interpretation:

> The model forms a readable student/malrule representation while learning answer prediction. Whether this STATE residual vector is a reliable causal handle on output behavior remains untested at spec.

Important caveats:

- Response model accuracy is only 0.5917 on same-template n=120; stronger training may be needed before swap is clean.
- In the 30 legal pairs, only 7 had `none = stay_B`, so the clean causal subset is small.
- STATE is readable, but history-last is also readable; this needs a stronger position/control analysis.
- We have not yet tested continuous layer blocks or query-side patching for this response-only model.

## Next Step

The next most useful run is not more random pairs immediately. It is:

1. train response-only bottleneck longer or larger effective epochs until same-template response accuracy is closer to 0.7;
2. run swap on a filtered set where `none = stay_B`, aiming for at least 50 clean pairs;
3. scan continuous blocks around late layers, especially L48-L63;
4. if still no flip, patch query-token post-attention/post-MLP states rather than only the STATE token.
