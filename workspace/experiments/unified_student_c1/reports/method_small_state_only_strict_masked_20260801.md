# Method-small 只走 state token 严格复核（2026-08-01）

## 结论

3000-step `standard` hard-bottleneck adapter 已训练完成且远端为 Passed。它确实使用
`<|vision_pad|>` 作为单 token state 地址，但旧的 T1/T2/T3/T5/T6 五个 headline
是 open-attention 推理，不能解释为“预测时只能经过 state token”。本次修正正式生成器，
确保 prompt forward 和每一步 KV-cache 增量解码都持续屏蔽 raw history，并在相同
method-small test 上执行每个 frozen protocol group 25 条的小测。

## Adapter 与训练配置

- experiment: `method-small-state-only-resumable-20260725`
- job: `qwen3-32b-method-small-state-only-resumable`
- state alias: `<|vision_pad|>`
- `bottleneck_mask=true`, `bottleneck_style=standard`, mask probability 1.0
- max steps: 3000
- adapter SHA256: `d94020128ef7cd8ca51a35ea5ee67ac4f449180585634d77a039e0149f4a2d03`

这个旧 adapter 只有 LoRA 权重；它没有当前 layerwise recurrent pipeline 的独立
`64 x 5120` state-transition artifact。后者是另一条正在远端训练的正式模型。

## 任务性能

| 任务 | 旧 open 推理 | 严格 state-only 小测 | 严格支持集 | 指标 |
|---|---:|---:|---:|---|
| T1 | 0.610 | 0.487 | 150 | accuracy；macro-F1 0.487，AUROC 0.545 |
| T2 | 0.457 | 0.376 | 125 | math equivalence |
| T3 | 0.004 | 0.000 | 75 | normalized exact match |
| T5 | 0.717 | 0.730 | 100 | joint math equivalence |
| T6 | 0.840 | 0.622 | 37 | accuracy；macro-F1 0.621，AUROC 0.788 |

旧 open 支持集为 T1/T2/T3/T5/T6 = 1000/600/500/600/200。严格小测按 protocol
group 和 binary label 做确定性分层抽样，因此 T6 的类别分布不同，不能把 0.622 与
旧自然分布 0.840 当作同分布直接差值。T1/T2/T3 的下降仍足以表明 open headline
受益于 raw-history 路径。T5 在严格路径保持，是当前最强的任务性能信号。

## C4 可读性与因果控制

同一 adapter 的 strict masked C4 probe：

- 22 类随机水平：0.0455
- best same-template accuracy：0.715（layer 63）
- best cross-template accuracy：0.735（layer 56）

因此 state 中有强可读的 misconception 表征。但 150-pair all-layer swap 的旧
`flip_to_A=0.287` 是无条件比例，混入了正常推理本来就预测 donor answer 的样本，
不能作为因果 swap rate。

在 `none=stay_B` 的 46 个 clean pairs 上：

| 条件 | flip-to-donor |
|---|---:|
| swap | 2/46 = 0.0435 |
| noop | 3/46 = 0.0652 |
| random | 2/46 = 0.0435 |
| same-malrule control | 2/46 = 0.0435 |

- swap − noop：-0.0217，paired bootstrap 95% interval [-0.0870, 0.0435]
- swap − random：0.0000，paired bootstrap 95% interval [-0.0652, 0.0652]
- swap − noop mean donor-margin change：-0.00987

结论是“可读但不可控”：特殊 token 形成了强 readout 地址，却没有在这个旧模型里形成
能稳定驱动答案的因果 state。

## 实现修正与验证

`evaluate_formal_protocols.py` 新增严格 bottleneck classification/generation：生成阶段
先以完整 4D mask 建 cache，之后每个 token 都用增量 4D mask 继续屏蔽 history KV
columns；闭合标签早停只减少无关尾部解码。相关单测 14/14 通过。

结果文件：

- `results/method_small_state_only_strict_masked_smoke_20260801/formal_n25_shard_a/metrics.json`
- `results/method_small_state_only_strict_masked_smoke_20260801/formal_n25_shard_b/metrics.json`
- `results/method_small_state_only_strict_masked_smoke_20260801/c4_causal_effect_summary.json`

## 对下一步训练的含义

不应只换 token 后直接宣称解决。需要让训练目标直接约束“state 改变应导致答案按 donor
方向改变”，并持续检查任务保留：

1. 以当前 layerwise recurrent state 为主线，decoder 永久不见 raw history。
2. 保留 specificity-triplet / paired donor-directional loss，而不是仅靠 response SFT。
3. 课程顺序采用任务保持 warm start → 少量 causal rehearsal → full mixed training；每个
   checkpoint 同时测 clean paired swap 与 T1/T2/T3/T5/T6。
4. token 类型（vision/image/video/new random）只作为 factorial control；核心 claim 依赖
   swap−noop/random 的配对增量，而非 token 的表面 flip rate。
5. 当前远端 layerwise full run 已含独立 student state artifact、causal schedule 和正式
   checkpoint-1000 C4/T1–T6 screen，完成后才是主结论。

## 2026-08-02 全量严格复核

同一 adapter 已在完整冻结 method-small test 上完成 strict state-only 推理，进程
exit code 为 0，共覆盖 2900 条样本。按 protocol support 加权的 headline 为：

| 任务 | open attention | strict state-only | strict support | 差值 |
|---|---:|---:|---:|---:|
| T1 | 0.610 | 0.594 | 1000 | -0.016 |
| T2 | 0.457 | 0.400 | 600 | -0.057 |
| T3 | 0.004 | 0.006 | 500 | +0.002 |
| T5 | 0.717 | 0.708 | 600 | -0.008 |
| T6 | 0.840 | 0.865 | 200 | +0.025 |

这否定了“严格只走 state token 会导致整体任务性能崩溃”，但 T2 有中等幅度下降，
T3 在两种可见性下都接近零。全量指标见
`results/method_small_state_only_strict_masked_full_20260802/metrics.json` 和
`results/method_small_state_only_strict_masked_full_20260802/headline.json`。

另外使用独立 seed 20260802 冻结 300 对 donor/recipient/same-malrule 样本，复测
all-layer state swap；1500 条条件记录完整（每对 none/noop/random/samemal/swap）。
在无干预预测为 recipient 答案的 96 个 clean pairs 上：

| 条件 | flip-to-donor |
|---|---:|
| swap | 8/96 = 0.0833 |
| noop | 8/96 = 0.0833 |
| random | 7/96 = 0.0729 |
| same-malrule | 9/96 = 0.0938 |

- swap - noop：0.0000，paired bootstrap 95% interval [-0.0417, 0.0417]
- swap - random：0.0104，paired bootstrap 95% interval [-0.0417, 0.0625]
- swap - same-malrule：-0.0104，paired bootstrap 95% interval [-0.0521, 0.0208]
- swap - noop mean donor-margin change：-0.00443

第二个独立 seed 因而复现了“任务可用、state 可读、但 donor-directional control 不成立”。
换成单 token `<|vision_pad|>` 足以提供稳定的信息地址，却不能替代 paired causal loss、
state transition 和持续的 task-retention rehearsal。完整记录见
`results/method_small_state_only_strict_masked_full_20260802/c4_swap_seed20260802_n300.json`
与 `c4_causal_effect_seed20260802_n300.json`。
