# Bottleneck S200 Masked Swap Smoke — 2026-07-10

## 目的

这轮实验专门回答一个问题:

> probe 已经能从 `<STUDENT_STATE>` 位置读出 malrule 信息,那么如果把另一个学生的 state swap 进去,并且用 bottleneck mask 强制 query 看不到 history,误解诊断会不会被因果地翻成 donor 的误解?

相比上一轮 60-step bottleneck,本轮做了两点增强:

- bottleneck identity adapter 从 60 steps 加到 200 steps;
- swap 分析时额外看 `none` 本来就预测成 recipient malrule 的干净子集,避免把 baseline 自己没学会的样本混进因果读数。

## 模型与文件

- base model: `/data1/liyuejia/models/Qwen3-32B`
- adapter: `experiments/unified_student_c1/results/qwen3_32b_single_malrule_identity_state_token_bottleneck_qlora_s200_len1536/final_adapter`
- eval output: `experiments/unified_student_c1/results/malrule_identity_bottleneck_s200_eval_same_masked_n120.json`
- swap output: `experiments/unified_student_c1/results/state_swap_misconception_bottleneck_identity_s200_all_layers_n50.json`

训练设置要点:

- task: `malrule_identity`
- train samples: 3000
- max steps: 200
- LoRA rank/alpha: 8/16
- max length: 1536
- `<STUDENT_STATE>` 作为 special token 加入并训练 embedding
- bottleneck 4D attention mask 开启

## 模型诊断准确率

同模板 `malrule_identity`,masked candidate likelihood,120 条样本:

| model | accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| bottleneck S200 | 0.6167 | 0.4930 | 0.9408 | 0.0863 |
| question-majority baseline | 0.4583 | 0.4025 | 1.6294 | 0.1255 |
| global majority | 0.2000 | 0.0159 | 2.9129 | 0.0140 |
| random uniform | 0.0455 | 0.0159 | 3.0910 | 0.1545 |

结论:mask 版模型确实学到了一部分 identity signal,明显强于题目多数类 baseline,但还没有到稳定高精度阶段。它比 60-step 更像一个可用模型,但不是一个足够强的 C4 headline 模型。

## All-layer Swap 结果

50 个有序 donor/recipient pair,同 family、不同 malrule,替换所有层的 STATE 位置 block output。

| condition | n | stay_B | flip_to_A | other | mean p(A) | mean p(B) |
|---|---:|---:|---:|---:|---:|---:|
| none | 50 | 24 / 48.0% | 2 / 4.0% | 24 / 48.0% | 0.0658 | 0.4440 |
| noop | 50 | 24 / 48.0% | 2 / 4.0% | 24 / 48.0% | 0.0666 | 0.4461 |
| swap | 50 | 24 / 48.0% | 2 / 4.0% | 24 / 48.0% | 0.0663 | 0.4442 |

整体看,swap 和 none/noop 的分布几乎一样。`flip_to_A` 没有高于 baseline,donor 的 malrule 概率也没有实质提升。

## 干净子集: none 本来预测 B 的 pair

更关键的是只看 `none = stay_B` 的 24 个 pair:

| subset | n | swap stay_B | swap flip_to_A | swap other |
|---|---:|---:|---:|---:|
| none stay_B only | 24 | 23 | 0 | 1 |

额外自检:

- `stay_B -> flip_to_A`: 0 / 24
- `swap` 改变 top-1: 1 / 24
- `noop` 改变 top-1: 2 / 24
- mean delta p(A): +0.00067
- mean delta p(B): -0.00556
- mean delta margin A-B: +0.03117

结论:在最有说服力的干净子集里,swap 没有把 recipient 翻成 donor。p(A) 只涨了约 0.0007,量级可以视为没有定向因果效应。

## 当前解释

这轮结果支持下面这个判断:

> 当前 setup 下,`<STUDENT_STATE>` 位置包含可被读出的 malrule 信息,但模型的 malrule_identity 输出并不主要由这个单点 state 激活因果控制。

换句话说,probe 能读不等于 generation/classification 会用。更强 bottleneck mask 让模型学会了一部分任务,但 all-layer STATE patch 仍然不能定向翻转误解诊断。

需要谨慎的是:noop 仍然会改变少量 top-1,说明当前 hook + 4D mask + candidate likelihood 管线有轻微数值/上下文脆弱性。因此这还不能写成最终定论,但可以作为非常有价值的中间结论:单点 STATE patch 目前不工作,下一步应该换更严格或更贴近输出路径的干预对象。

## 下一步建议

1. 先把 bottleneck identity 模型训到更高同模板准确率,目标至少接近或超过 0.70,否则 swap 读数会被 baseline 不稳污染。
2. 对比 response-prediction 任务的 swap,因为 C4 原 claim 是“只学预测答案后内部涌现误解表示”,不是直接训练 identity 标签后做 identity swap。
3. 做 layer subset scan,尤其是 probe 最强层附近,避免 all-layer replacement 被后续层重写或平均稀释。
4. 若 STATE 单点仍不动,尝试 patch query token 在关键层的 post-attention/post-MLP 表示,因为输出行为可能由 query-side computation 直接承载。
5. 继续保留 noop 作为硬门槛:noop 和 none 的类别分布必须近似一致,否则正式 swap 读数不可信。
