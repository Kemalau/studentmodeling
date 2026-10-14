# Bottleneck Layer Scan and Token Budget Check — 2026-07-10

## 目的

上一轮 all-layer STATE swap 没有把 recipient 的 malrule diagnosis 翻成 donor。这里检查两个可能解释:

1. 不是所有层都该换,也许只有某个中间层有因果效应;
2. `max_length=1536` 可能不够,导致 history 被截断,STATE 没有读到完整学生历史。

## Token Budget

Tokenizer 使用 bottleneck S200 adapter 目录,只统计 `malrule_identity` prompt 长度。

| split | n | min | p50 | p90 | p95 | p99 | max | >1536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 3000 | 573 | 639 | 688 | 710 | 756 | 835 | 0 |
| test_same_template | 300 | 583 | 640 | 686 | 703 | 767 | 810 | 0 |
| test_cross_template | 300 | 580 | 639 | 695 | 711 | 769 | 805 | 0 |

结论:当前 Malrule identity 数据没有 token budget 问题。prompt 最大约 835 tokens,远低于 1536,所以 swap 不动不是因为 history 被截断。

## 代码改动

`experiments/unified_student_c1/scripts/state_swap_misconception_smoke.py` 增加了单层扫描模式:

- `--scan-single-layers`: 对 `--layers` 中每个层单独做 patch;
- `--scan-stride`: 可对层列表按 stride 粗扫;
- 输出 `layer_configs`,并按 layer config 分别写 summary。

这样模型只加载一次,STATE 激活只抓一次,然后对多个单层配置重复打分。

## 粗扫: L0,L8,...,L56

输出文件:

`experiments/unified_student_c1/results/state_swap_misconception_bottleneck_identity_s200_single_layer_stride8_n24.json`

设置:

- model: `/data1/liyuejia/models/Qwen3-32B`
- adapter: `experiments/unified_student_c1/results/qwen3_32b_single_malrule_identity_state_token_bottleneck_qlora_s200_len1536/final_adapter`
- pairs: 24
- clean subset: `none` 本来预测为 recipient malrule 的 10 对
- bottleneck mask: on

| layer | none flip | swap flip | clean n | clean swap flip | clean swap changed | clean noop changed | delta p(A) | delta p(B) | delta margin A-B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 2 | 2 | 10 | 0 | 1 | 0 | -0.001121 | -0.004967 | -0.031299 |
| L8 | 2 | 2 | 10 | 0 | 0 | 0 | -0.001909 | 0.008039 | -0.043781 |
| L16 | 2 | 2 | 10 | 0 | 0 | 1 | -0.000870 | -0.003790 | -0.074989 |
| L24 | 2 | 2 | 10 | 0 | 0 | 0 | -0.000004 | -0.006342 | -0.012496 |
| L32 | 2 | 2 | 10 | 0 | 1 | 0 | 0.002102 | -0.010456 | 0.100019 |
| L40 | 2 | 2 | 10 | 0 | 0 | 0 | -0.001008 | 0.001073 | -0.031231 |
| L48 | 2 | 2 | 10 | 0 | 0 | 0 | -0.000946 | -0.001074 | -0.037514 |
| L56 | 2 | 2 | 10 | 0 | 0 | 0 | 0.000846 | 0.001751 | 0.006225 |

粗扫结论:

- 每个单层的 `swap flip_to_A` 都是 2/24,和 none 一样;
- clean subset 中没有任何 `stay_B -> flip_to_A`;
- L32 的 margin 有小幅正向移动,但 top-1 没有翻转,p(A) 只增加 0.002102。

## 细扫: L28,L30,L32,L34,L36

输出文件:

`experiments/unified_student_c1/results/state_swap_misconception_bottleneck_identity_s200_single_layer_l28_36_n24.json`

| layer | none flip | swap flip | clean n | clean swap flip | clean swap changed | clean noop changed | delta p(A) | delta p(B) | delta margin A-B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L28 | 2 | 2 | 10 | 0 | 0 | 0 | 0.000049 | -0.001091 | 0.031196 |
| L30 | 2 | 2 | 10 | 0 | 0 | 0 | -0.001068 | 0.001754 | -0.000068 |
| L32 | 2 | 2 | 10 | 0 | 1 | 0 | 0.002102 | -0.010456 | 0.100019 |
| L34 | 2 | 2 | 10 | 0 | 0 | 0 | -0.000988 | -0.002324 | -0.006187 |
| L36 | 2 | 2 | 10 | 0 | 0 | 0 | -0.000743 | -0.002424 | 0.018809 |

细扫结论:

- L32 附近没有隐藏的单层 flip;
- `clean swap flip` 仍然是 0/10 for every tested layer;
- L32 的弱 margin 信号复现,但太小,且没有 top-1 行为变化。

## 当前判断

这轮排除了两个简单解释:

- 不是因为 token length 不够;
- 不是因为只看了最后一层而漏掉明显的单层因果层。

当前更合理的解释是:

> malrule 信息可以被 probe 从 STATE 位置读出,但模型的 diagnosis 输出路径没有把单个 STATE residual vector 当作可替换的因果控制变量使用。

这个结论仍然限定在当前 setup:

- 任务是直接训练过的 `malrule_identity`,不是 response-only C4 headline;
- 模型同模板 accuracy 约 0.617,还不是很强;
- 扫的是单层和 all-layer STATE patch,还没扫连续层块或 query-side patch。

## 下一步

如果继续追 C4 causality,优先级应该是:

1. 训更强的 bottleneck response-prediction 模型,回到“只监督答案、不监督 malrule label”的科学设定;
2. 扫连续层块,例如 L24-L36、L28-L40,因为单层 patch 可能被后续层抹掉;
3. patch query token 的 post-attention/post-MLP 表示,检验行为路径是否在 question-side computation;
4. 将 noop 稳定性作为硬门槛,否则不要解释 swap 数字。
