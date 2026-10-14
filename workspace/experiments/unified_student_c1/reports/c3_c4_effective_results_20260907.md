# C3 / C4 当前有效证据汇总（2026-09-07）

## 1. 问题、结论与非目标

本次按用户要求仅整理现有结果，不新训模型、不提交作业、不重启 recurrent。C4 在本文特指 one-shot 状态因果效应，不沿用旧 taxonomy 文档中以 recurrent State8 为主的定义。数据截点包含 2026-09-07 05:07 UTC 刚收齐的历史长度诊断。

- C3：存在任务/数据集相关的正迁移证据。最值得保留的是真实 Eedi T3：同骨干、目标任务曝光匹配下，Task-only 44.52% → Unified 49.66%，+5.15 个百分点。但仅一个训练 seed、总更新数不匹配；不能写成普遍正迁移或特定辅助任务的已识别因果效应。
- C4：原模型的候选支持状态干预成立；后续 candidate-free 训练使已训练规则在新数值、新历史上的自由生成迁移明显改善。新规则归纳/迁移没有可靠成功，增加历史到 12 条也没有救活。
- 新训练 A/B 都有正向状态干预效应，但 B 的显式 state-exchange 训练没有显示优于匹配 SFT A 的增益。
- 原始 checkpoint、后续训练 A/B、Unified 多任务 checkpoint 是不同模型。不得把它们的最好数字拼成同一个模型的能力。

## 2. 样本构造与具体样本

### C3

同骨干基线均为 Qwen3-32B QLoRA，rank、上下文长度和冻结评测一致。每个 Task-only 一轮训练恰好看到 Unified 一轮中的同一组该任务 example IDs、target 与 target_label，各一次；训练目标签名没有不匹配。总更新量不匹配。

| 任务 | 训练目标行数 | 冻结测试行数 | 主指标 |
|---|---:|---:|---|
| T1 正确性预测 | 110,525 | 10,913 | Accuracy；另报 AUROC |
| T2 作答预测 | 55,569 | 3,693 | 数学等价率 |
| T3 错因诊断 | 6,776 | 1,494 | 独立语义判断准确率 |
| T4 前瞻作答/错误预测 | 78,600 | 4,600 | 正确性与作答同时正确的 JointMath |
| T5 错误持续性预测 | 8,375 | 976 | Accuracy；另报 AUROC |

Unified 总训练行数 259,845；五任务测试共 21,676 行。T3 包含真实 Eedi 894、MalRule 同模板 300、跨模板 300。Eedi 的 894 行对应 373 道题的不同选项，不能一律看作独立学生。

### C4

- 历史 CF1：训练 6,000、验证 600 行，原始 response 与同 family 反事实重组各一半；22 个标签。原 seed 13/42/97 使用同一冻结 150-pair manifest，不是不同 pair 采样。
- 新执行训练：六个已知算术规则；160 个无序历史对构造 1,920 训练行，8 对构造 96 验证行；测试 16 对、32 个方向，每方向三个新问题，共 96 directional queries。10 个条件 × baseline/A/B = 2,880 输出。每段历史三条观察。
- 未见规则：六个新构造规则、两个已知规则对照；四类别各四无序历史对，各 24 directional queries；九条件 × 两模型 = 1,728 输出。
- 历史长度诊断：复用上述问题，对比 3、6、12 条独特历史，以及原三条重复四遍。七条件 × 四长度 × 两模型 = 5,376 输出。这是同一面板的事后诊断，不是独立复现。

新执行训练排除所核验 CF1/MalRule 和先前面板的数值组合，也排除新 train/valid/test 之间的组合重叠。当前 donor/recipient 目标不同、与数学正确答案不同，且不出现于可见历史数值中；同一 donor 的三个目标也互不相同。新颖性不涵盖预训练，已知规则仍然是训练过的规则。

具体真实 donor 历史：3/25 × 18/25 → 54；15/28 × 26/28 → 390；16/19 × 15/19 → 240。该学生遵循“忽略分母”。状态抽取只看历史前缀，不看当前问题、目标答案或候选。

## 3. 坐标、训练目标与指标定义

C4 干预坐标：Qwen3-32B 第 0–63 个 decoder block 输出处，同一个 `vision_pad` token 的 residual；总形状 64 × 5120。没有筛选神经元或仅挑一个最佳层来实施主干预。

旧候选指标：

`D_candidate = mean_clean(1[pred_swap = donor_answer] - 1[pred_control = donor_answer])`

其中 clean 要求原生模型选 recipient 答案，随 checkpoint 改变。旧 control 是范数匹配 Gaussian noise；9 月 4 日新增 real-random 则是真实、不同规则的历史状态，两者必须分开命名。候选按完整目标序列的 log-probability 和评分。

新自由生成指标：

`D_open = mean_all_queries(1[generated_swap = donor_target] - 1[generated_wrong_rule_state = donor_target])`

这是全测试问题上的 donor 答案命中率差，不是只对原生正确 recipient 的 flip。不能把 D_open 或 donor hit 与旧 0.411 flip 当作同一指标。

新训练的共同损失为目标答案/EOS token 的平均负对数似然：

`L = -mean_(t in target/EOS) log p_theta(y_t | prefix, y_<t)`

A 在 donor 历史上做 candidate-free SFT；B 在 history 阶段把 donor 状态以可微方式换入 recipient 路径并监督 donor 答案。source/prompt 不计 loss。主训练对比为 `D_B - D_A`，不是 B 相比零是否为正。

C3 使用 `Delta_task = metric_Unified - metric_Task-only`。Eedi 新增审计按题目 ID 聚类，把同题不同选项一起重采样，统计比例之差。

## 4. 指标符号的数值例子

实测原 seed 42：directed 39/95、noise 18/95，因此 D = 21/95 = 0.2211。新 A：directed 56/96、wrong-rule 6/96，因此 D = 50/96 = 0.5208。

仅作损失说明：目标 token 概率从 0.2 提至 0.5，NLL 从 1.609 降到 0.693。这不是测得的 neuron attribution，也不代表每个答案都执行正确。

## 5. 真实成功例与固定 checkpoint

新 A 的 `test:fractions:000:0` 在同一个 donor 状态下：

| 新题 | donor 规则目标 | swap 实际自由生成 |
|---|---:|---:|
| 11/29 × 14/29 | 154 | 154 |
| 18/30 × 13/30 | 234 | 234 |
| 5/24 × 9/24 | 45 | 45 |

三题均无候选、无当前正确答案，目标不在 donor 历史；这是可跨问题复用的已知规则条件化行为证据，不是生成新规则。

固定模型哈希：

| 模型 | Adapter SHA256 |
|---|---|
| 原 CF1 seed 42 | `c9b463ed12f11ef84d346c1399d2588b9b396ecb9e15c5f4ca85c7f1d7fe991f` |
| A，final step 240 | `7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6` |
| B，final step 240 | `87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32` |

## 6. 筛选流程与预算

原候选：150 冻结对 → checkpoint 自身 clean 子集 → 全部预定对照。三历史 run 的共同 clean 子集为 53，复算仍保留 seed 97 的失败。

新训练：冻结全部数据 → A/B 相同监督目标次序、26,524 个答案/EOS token → 240 次参数更新 → 仅评预定 final checkpoint。前 120 步显式规则，后 120 步 history；只 B 使用可微交换。rank 8、LR 2e-5、12 步 warmup、AdamW、gradient accumulation 8、长度上限 768。A 被平台中断后重放部分日志；最终轨迹仍 240 步，审计保留 274 条日志与重放标记，不作为额外独立 seed。

新规则与历史诊断保留全部类别、所有预先纳入的问题和失败输出。没有按结果筛掉难规则；未见规则每类只有四独立 history pairs。

## 7. C3：已完成的比较与因果边界

### 7.1 Unified 对 Task-only

以下差值均为 Unified − Task-only，pp 为百分点；原完整任务区间为逐样本 paired bootstrap，未估计训练 seed 不确定性。

| 任务 | n | Task-only | Unified | 差值（pp） | 95% paired CI（pp） |
|---|---:|---:|---:|---:|---|
| T1 | 10,913 | 69.17% | 68.75% | −0.41 | [−1.02, +0.19] |
| T2 | 3,693 | 50.56% | 51.83% | +1.27 | 未提供可用逐题配对 |
| T3 | 1,494 | 54.55% | 56.63% | +2.07 | [−0.60, +4.69] |
| T4 | 4,600 | 76.98% | 76.67% | −0.30 | [−0.70, +0.09] |
| T5 | 976 | 86.27% | 86.17% | −0.10 | [−0.92, +0.72] |

已精确匹配目标任务数据/监督曝光；没有匹配总更新数。一训练 seed。T2 的 Unified 归档逐题数据被脱敏、没有 raw_output/target，不把缺字段判成正确、不伪造配对区间。T3 整体 McNemar p=0.128。

### 7.2 最有价值的正向子集：真实 Eedi 诊断

| T3 来源/协议 | n | Task-only | Unified | 差值（pp） |
|---|---:|---:|---:|---:|
| 真实 Eedi | 894 | 398/894 = 44.52% | 444/894 = 49.66% | +5.15 |
| MalRule 同模板 | 300 | 67.33% | 69.00% | +1.67 |
| MalRule 跨模板 | 300 | 71.67% | 65.00% | −6.67 |

本次对现有 T3 判断逐行核对 example ID、target hash、TRAPI 模型与 rubric，真实 Eedi 上 Unified-only correct=138、Task-only-only correct=92。按 373 道题聚类的 20,000 次配对 bootstrap，增益 95% CI = [+1.57, +8.74] pp。逐行 McNemar p=0.002923 仅作辅助，不能忽视同题选项相关性。

这是事后来源分组分析，未做多重比较修正；不能冒充预注册 endpoint 或多 seed 复现。该结果支持这次训练的真实 Eedi 子集受益，不识别是哪一个辅助任务贡献，亦不能概括“语义越丰富必然迁移越好”。跨模板 MalRule 的负差值必须一起报告。

### 7.3 T2 来源分解

| 数据来源 | n | Task-only | Unified | 差值（pp） |
|---|---:|---:|---:|---:|
| DBE | 223 | 58.74% | 57.85% | −0.90 |
| Eedi Response | 984 | 51.52% | 51.93% | +0.41 |
| FoundationalASSIST | 1,886 | 50.85% | 51.86% | +1.01 |
| MalRule 同模板 | 300 | 45.33% | 50.00% | +4.67 |
| MalRule 跨模板 | 300 | 44.67% | 48.67% | +4.00 |

这些是描述性正/负迁移信号，无可用配对区间，不能仅按点估计称显著。

### 7.4 必须保留的 C3 负结果和旧 smoke

T1 + interaction-matched T2 add-one 已完成：T1 Accuracy=0.686521、AUROC=0.759587；T1-only Accuracy=0.691652。差 −0.005131，95% paired CI [−0.011271,+0.001008]，p=0.103。未支持 T2 帮助 T1，亦未确立显著负迁移。

旧 Eedi/MalRule smoke（n=300、120 步、候选 logprob 评分、非曝光匹配）：

| 条件 | Accuracy | Macro-F1 |
|---|---:|---:|
| Eedi-only | 0.437 | 0.269 |
| +identity | 0.470 | 0.314 |
| +response | 0.423 | 0.271 |
| +identity+response | 0.487 | 0.314 |
| all mixed | 0.493 | 0.320 |

后续 exposure-matched auxiliary 三臂诊断固定 Eedi 500 行、MalRule underlying 500 行和训练次序，仅改 auxiliary target；使用另一个 layerwise/recurrent 路径。三个 arm 的真实 Eedi endpoint 都为 0，不能用于支持正迁移。

| Auxiliary | 汇总器的 overall T3 | 真实 Eedi | C4 D（该诊断 checkpoint） |
|---|---:|---:|---:|
| correctness | 0.241 | 0.000 | 0.216 |
| misconception | 0.233 | 0.000 | 0.195 |
| response | 0.137 | 0.000 | 0.253 |

重要分母修正：每臂实际只 judge 75 条（Eedi、MalRule 同/跨各 25），overall 0.241/0.233/0.137 是按正式来源支持数加权的估计，不是实测完成 1,494 条；真实 Eedi 为 0/25，而非 0/894。三个 arm 均未通过 T3 与全部 non-T3 preservation gate，单训练 seed，保留为特定配方/路径的诊断负结果。不能用它否定后来的 unrestricted Unified 的 Eedi 正增益。

## 8. C4 干预操作与对照

全量替换：`h_recipient[l,state] <- s_donor[l]`，l=0..63。生成时只在 prefill patch；一个捕获轨迹复用到三个问题，每次 fresh KV。新训练/诊断的 late-fusion 限制历史读取，问题不能读取状态，answer anchor/答案可读取状态。后续 A/B 的修正版 mask 与原实验分开评价；原 baseline 已按同一修正版重评。

原候选的剂量实验用 `h_alpha=(1-alpha)*h_recipient+alpha*h_donor`，alpha=0/0.25/0.5/0.75/1，不是新增训练 loss。

对照区分：Gaussian noise；real-random third-rule history；新 A/B 的独立 recipient-rule history（wrong-rule 相对 donor）；同 donor-rule 不同历史；同 recipient-rule；own-state no-op；显式规则 oracle；open history。各实验没有全部同一组对照，不混称。

## 9. 评测协议与独立裁判

C3 T3 是开放生成 + 独立 TRAPI 语义裁判，实际模型 `gpt-4o_2024-11-20`，rubric `t3-semantic-equivalence-v2-substantive-schema`；配置中别的请求模型名不能替代实际 judge identity。T2/4 主指标为数学等价，不是只比字符串。

C4 新实验：NF4/bfloat16、eager attention、frozen rank8、dropout off；greedy、48 token 上限、EOS 或闭合标签停止，768 token 总长度内不截断。单数字/单响应标签严格解析，以有理数等价评分；非法、多答案、超限输出计失败。答案来自合成规则，无 LLM judge，不是人类真实作答。

新执行训练使用 16 无序历史对聚类 bootstrap；新规则/历史长度每类别仅四对，10,000 次，均探索性未校正区间。同 pair 的两个方向、三个问题保留相关性。经验零宽区间不代表总体无不确定性。

## 10. C4：有效结果总表

### 10.1 原 one-shot 与 seed 修复

| Run | clean n | 跨模板 response | 最佳跨模板 probe | directed | Gaussian noise | same-rule | no-op flip | D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 13 | 101 | 0.920 | 0.913 | 0.455 | 0.267 | 0.059 | 0.030 | 0.188 |
| 旧 42 | 95 | 0.907 | 0.933 | 0.411 | 0.189 | 0.042 | 0.000 | 0.221 |
| 旧 97 | 66 | 0.643 | 0.692 | 0.045 | 0.030 | 0.015 | 0.045 | 0.015 |
| 严格 seed 修复后 13 | 50 | 0.833 | 0.806 | 0.000 | 0.020 | 0.000 | 0.020 | −0.020 |
| 严格 seed 修复后 97 | 92 | 0.907 | 0.909 | 0.500 | 0.228 | 0.076 | 0.011 | 0.272 |

历史 D≥0.15 的 two-of-three 数值规则为 2/3，但历史 seed 只在 Trainer 初始化时设置，LoRA 先构造，未完整控制 adapter 初始化。因此不能包装成严格可重现的三训练 seed 证据。旧 97 val loss=0.0710，旧 13/42=0.0147/0.0158；新 13=0.03323、新 97=0.01663。支持优化路径敏感性，不允许事后删除旧 97。

新 42 有明确取消排队记录，未形成完整新版三 seed 表；不得把旧 42 与新 13/97 拼成新的 2/3 gate。`seed13_audit.json` 中 passed=true 是 artifact/协议审计通过，不是 D gate 通过。

probe=0.933 是九个被测层中的 best cross-template layer（旧 42 为 layer 56），不是固定预选层结果，不是自由生成准确率。

### 10.2 有效机制支持与旧模型边界

| 实验 | 结果 | 可用解释 |
|---|---|---|
| 原 masked exploratory sweep | directed flip 0.4944，noise 0.2921 | 有探索性效应；0.494 不是 D，也非最新主估计 |
| 原 seed42，真实随机学生状态对照 | directed 39/95=0.4105，real-random 16/95=0.1684；D=0.2421，paired CI [0.1263,0.3579] | 不只优于高斯噪声；仍是同一 frozen panel 的候选指标 |
| 剂量 alpha=0/0.25/0.5/0.75/1 | flip=0/0.0421/0.1684/0.3368/0.4105 | 定向干预强度与效应同向，探索性 |
| strict history-free layerwise bridge | clean n=97；swap/noise/same/no-op=0.4021/0.1959/0.0722/0；D=0.2062 | 接收解码阶段无 raw history、无 persistent history KV 仍有效；仅该候选任务/接口，不是一般充分统计量证明 |
| 原模型，无候选新数值自由生成 | directed=3/81，real-random=3/81；D=0，CI [−0.0370,+0.0370] | 原模型不支持可靠开放规则执行 |
| 原模型，候选可见但无当前正确答案 | directed=61/81，real-random=29/81；差0.3951，CI [0.2222,0.5556] | 候选支持效应仍在；不是无候选执行 |
| 固定相同候选、两题联合命中 | 5/46 vs 1/46；差0.0870，CI [−0.0217,0.2174] | 未排除跨题固定候选偏好 |

81 题旧面板另有 12 题语义适用性/标签问题；事后有效69题敏感性仍为候选可见49/69 vs22/69，无候选1/69 vs1/69。原表保留，不借此改主 endpoint。固定候选的46个单位是23 donor × 两顺序，不是46独立 donor。

### 10.3 新 candidate-free 执行训练：当前最强 one-shot 正结果

测试六个已训练规则、全新历史/数值；每模型96题，16独立对。

| 模型 | directed donor hit | wrong-rule real-state donor hit | D_open [95% pair-cluster CI] | 同状态三题全中 |
|---|---:|---:|---|---:|
| 原 checkpoint，同新协议重评 | 4/96=4.17% | 3/96=3.13% | 0.0104 [−0.0312,0.0521] | 0/32 |
| A：candidate-free SFT | 56/96=58.33% | 6/96=6.25% | 0.5208 [0.3021,0.7396] | 14/32 |
| B：增加 state-exchange supervision | 56/96=58.33% | 8/96=8.33% | 0.5000 [0.2604,0.7396] | 14/32 |

wrong-rule 三题全中均0/32；A/B 同 donor-rule 另段历史也均14/32。A/B donor-native 各57/96、56/96；显式规则oracle均60/96。

主对比 B−A = −0.0208，CI [−0.0521,0.0000]：不支持 state-exchange loss 的增量优势。A/B相对原baseline的D增益为0.5104/0.4896，区间均正，但只是单seed修复训练 pilot，不能反推原checkpoint早已具备相同能力。

效果主要来自分数规则：A/B均 directed46/48、wrong-rule0/48；减法为10/48 vs6/48（A）或8/48（B）。因此不能称六类规则均稳定成功。

### 10.4 未见规则与历史长度边界（最新已完成）

以下原3条历史结果A/B相同，每类别24题、四独立对。

| 类别 | directed | wrong-rule | D | 三题全中 |
|---|---:|---:|---:|---:|
| 新 offset 规则 | 0/24 | 0/24 | 0.0000 | 0/8 |
| 新 composition 规则 | 1/24 | 0/24 | 0.0417 | 0/8 |
| 新 conditional 规则 | 14/24 | 13/24 | 0.0417 | 0/8 |
| 已训练规则对照 | 24/24 | 0/24 | 1.0000 | 8/8 |

新规则没有类别的D区间下界严格大于零；三类新规则共0/24方向组三题全中。partial conditional 命中不代表执行了条件规则。

2026-09-07 历史长度诊断8/8作业pass、5,376/5,376输出完整，每分片672唯一条。每模型各长度结果：

| 类别 | 3条 swap/random | 6条 swap/random | 12条 swap/random | 3条重复4遍 swap/random |
|---|---:|---:|---:|---:|
| offset | 0/0 | 0/0 | 0/0 | 0/0 |
| composition | 1/0 | 0/0 | 0/0 | 0/0 |
| conditional | 14/13 | 13/12 | 12/11 | 13/13 |
| 已训练规则 | 24/0 | 24/0 | 24/0 | 24/1 |

每格分母均24，A/B表中命中数一致，不是两个独立训练seed。所有新规则/长度的三题全中仍0/8，已训练对照一直8/8。offset 的 open-history 0/24→3/24 有小幅改善，但 donor-native/swap仍0/24；不足以恢复可靠新规则迁移。oracle同prompt跨长度输出完全一致；对照不是整体失效。

结论是“在已测3→12条范围内，加历史不足以解决”，不是证明任何数量历史都无效。增加历史也移动state位置、超出三观察训练格式，未精确匹配token数。

## 11. 失败、no-op 与不能再用的论断

- 新执行训练A/B各1/96 no-op文本/停止差异。A为180/21与60/7，数值等价；B为4732与4632，数值不同，但均不改donor命中。不能宣传全部新实验no-op=0。
- 新规则B有1/96 no-op差异；最新history诊断A/B各1/384对比有文本差异。保留原分母与失败，未知量化误差/实现差异的具体来源。
- 已知规则的新数值成功反驳了“只复制历史里某个答案”的解释，但不能唯一确定state内存着程序；state编码规则身份、共享decoder实现算术仍是合理替代解释。
- 新offset例：16/20 × 13/20，目标211，swap和wrong-rule均208，显式规则oracle能给211。composition连oracle也弱；不同类别可能同时有归纳和执行问题，不能唯一归因于state容量。
- 末层block OUTPUT patch对过去state token的更改发生在最后一次attention之后，后续位置logits/KV没有读取路径；其零效应不能证明单向量无效或“状态本质上必须是跨层轨迹”。保留有效trajectory传输结果，撤回该强比较结论。
- recurrent/State8失败保留为另一研究线的负结果，不再作为one-shot的成功门槛。本次不声称任何模型已通过真人因果干预或recurrent闭环自更新。

## 12. 可直接采用的论文表述

C3：在目标任务监督曝光相同的同骨干比较中，联合教育训练呈现任务依赖的迁移；真实Eedi诊断有正增益，部分合成跨模板任务下降。当前为单seed、非总更新匹配证据。

C4：历史诱导的one-shot学生状态能够因果性地控制作答。候选支持条件下原模型已有明确效应；补充candidate-free监督后，状态可跨新历史、新数值复用，诱导已训练错误规则一致的自由生成，且明显强于真实错误规则状态对照。该能力未泛化到新构造规则，增加到12条观察仍未解决；显式exchange训练未优于匹配SFT。

不应写：普遍正迁移；未见规则执行器；自发发明新规则；已证明程序本体存于单token；单层/单向量不可能；统计意义的一般充分统计量；同一checkpoint同时具有所有C3/C4最好结果。

## 13. 复现、来源与本次核验

本次读取原始代码、冻结manifest、逐题结果与审计JSON；用jq从原始输出重计A/B/baseline的2,880条、新规则1,728条命中数；历史诊断核对8个complete文件、各672条唯一记录与总数5,376。未重新运行32B推理。新增C3来源分组审计脚本只读现有判断，验证输入SHA256和全部配对字段，重算题目聚类区间，输出到stdout。

```bash
cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
node scripts/audit_c3_eedi_question_transfer_20260907.mjs
```

关键来源（相对以上项目根目录）：

- C3完整表：`results/taskonly_qwen_formal_eval_20260813/formal_specialist_v3/formal_evaluation_matrix.json`。
- C3 paired：`results/taskonly_qwen_formal_eval_20260813/paired/complete.json`，SHA256 `31af47c95fc89dcf4c017d2c9406d782c98333532ea1097b352a685ae93ed6df`。
- 目标曝光：`results/taskonly_qwen_formal_eval_20260813/target_exposure_audit.json`。
- 新增Eedi审计：`scripts/audit_c3_eedi_question_transfer_20260907.mjs`；内部固定两份原始判断SHA256。
- T1+T2：`results/t1_t2_paired_addone_recovery_20260901/comparison.json`。
- C3三臂：`results/c3_exposure_matched_world2_20260809/summary.json`及`t3_semantic/*/judged/metrics.json`；对应`scripts/summarize_c3_exposure_matched.py`。
- 历史C4：`results/five_trajectory_audit_20260824/audit.md`、`reports/c4_seed97_failure_audit_20260823.md`。
- 新seed13/97：`results/c4_strict3seed_v2_seed13_download_20260824/seed13_audit.json`、`results/c4_strict3seed_v2_seed97_download_20260824/seed97_audit.json`。
- 真实随机/剂量：`results/c4_real_random_dose_seed42_20260904.summary.json`。
- bridge：`results/layerwise_state_bridge/raw_20260823/recurrent-cf1-layerwise-state-bridge-c4-seed42-20260731/cf1-layerwise-state-bridge-compact-n150-seed42/gate/layerwise_state_bridge.json`。
- 原模型自由生成/固定候选：`reports/c4_one_shot_delivery_conclusion_20260905.md`；末层审计：`reports/c4_final_layer_interpretation_audit_20260905.md`。
- 新训练：`reports/c4_execution_training_results_20260906.md`、`results/c4_execution_training_v1_recovery1_20260906.delivery_audit.json`。
- 新规则：`reports/c4_unseen_rules_results_20260906.md`、`results/c4_unseen_rules_v1_feedsllm_20260906.summary.json`。
- 历史长度：`reports/c4_history_diagnostic_results_20260907.md`、`results/c4_history_diagnostic_v1_20260907.summary.json`，SHA256 `a34fffba445cd70d3078e0f3d7f72285f86e1a6c9d148daabe379cc8fe653ca2`。

新训练实验ID：`c4-execution-training-v1-seed42-recovery1-20260906`；新规则：`c4-unseen-rules-v1-seed42-feedsllm-20260906`；历史诊断：`c4-history-diagnostic-v1-seed42-feedsllm-20260907`。均为既有Feeds-LLM任务，本轮没有新提交。详细镜像/UAI/数据路径/作业receipt见各execution报告，不在此混成一个运行协议。

`neuron-discovery-reporting`技能要求促使本汇总区分探索性probe、候选flip、自由生成命中、no-op文本差异、artifact通过与科学gate通过，并保留新旧seed和负结果。旧paper notes仍含recurrent主线和已失效的末层解释；本次不改写那些历史文件。
