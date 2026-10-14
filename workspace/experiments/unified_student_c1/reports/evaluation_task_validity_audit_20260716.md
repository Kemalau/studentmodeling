# T1–T6 评测任务合理性审计

日期: 2026-07-16

## 1. 总结论

当前六个任务不能作为同一种评测协议直接汇总。

| 任务 | 构念是否保留 | 当前评测是否可作主结果 | 处理决定 |
|---|---|---|---|
| T1 correctness | 是 | 部分可用 | 保留; 区分 temporal KT 与 non-temporal profile prediction |
| T2 response | 是 | 当前不完整 | 拆成 closed-choice 与 open-response 两个协议 |
| T3 misconception | 是 | 否 | 保留任务, 重做 evaluator; row-specific oracle candidates 只能进 appendix |
| T4 mastery | 仅作 proxy | 否 | 改名 future-KC-performance band, 放 appendix |
| T5 prospective response | 是 | FoundationalASSIST 协议合理, 尚未正式评测 | 保留为 C2 核心, 做真正开放生成并按距离报告 |
| T6 persistence | 来源相关 | 不能混合 | FoundationalASSIST strict recurrence 单独报告; XES proxy 改名; DBE test 太小 |

论文主表继续使用 T1/T2/T3/T5, 但必须先修复协议。T4 不进主表。T6 作为独立 secondary analysis, 不进入 headline macro average。

## 2. 当前矩阵已经过期

当前 `results/strong_baseline_matrix/evaluation_matrix.md` 基于旧的 `mixed_t1_t6_balanced/test.jsonl`:

- 共 10,544 条 test;
- 不含 FoundationalASSIST;
- T5 只有 DBE-KT22 的 200 条;
- strict T6 只有 DBE 12 条, 其余 2,000 条来自定义不同的 XES proxy;
- Qwen 使用 candidate-label log-probability, 不是自由生成。

FoundationalASSIST 新增 9,779 条 test, 包括 4,695 条 T5 和 1,084 条 strict T6。旧矩阵因此不能代表当前总数据或当前 C2 能力。

## 3. 跨任务共同问题

### 3.1 测试集几乎都是 warm-item

所有 longitudinal 数据都实现了学生级隔离, 但测试题大多在训练集中出现过:

| 来源 | test 学生与 train 重叠 | test 题目在 train 出现比例 |
|---|---:|---:|
| DBE-KT22 | 0 | 100.00% |
| Eedi Response | 0 | 100.00% |
| XES3G5M | 0 | 96.99% |
| FoundationalASSIST | 0 | 99.45% |

这是一种合理的 **new-student / warm-item** 协议, 但不能解释为新题泛化。正式结果至少应并列报告:

1. warm-item student-disjoint;
2. cold-item 或 cold-template;
3. 若任务允许, cold-student + cold-item。

question-majority 在 T2/T5/T6 上很强, 主要反映 item overlap, 不是充分的学生状态建模证据。

### 3.2 outcome-aware 平衡抽样破坏自然发生率

T1 的多个测试集按真实 correct/incorrect 标签平衡位置。FoundationalASSIST T1 test 为 1,000/1,000, XES T1 test 为 3,613/3,613。

该协议适合比较 discrimination, 但不适合解释:

- 自然部署 accuracy;
- prevalence-sensitive precision;
- 原始分布上的 calibration/ECE/Brier/NLL。

需要额外保留不按标签采样的 chronological natural-prevalence test。

### 3.3 JSON 结构含 target 字段, prompt 虽隐藏但 baseline 可读取

旧 DBE、XES、Eedi Response 和 Eedi-Mining rows 的 `query` 或 `metadata` 保留了 `correctness`, `student_response`, `correct_response`, `target_correctness` 或 gold misconception 字段。Qwen 的 prompt formatter 通常不显示这些字段, 但其他 evaluator/baseline 可以直接读取。

现有 `train_pending_task_baselines.py` 在 T2/T5 推理时读取 query 中的正确答案 ID, 而 Qwen prompt 没有提供 answer key。这造成输入不对齐, 不能作为公平模型比较。

正式评测文件应使用 sanitized query; gold 只保存在 evaluator 单独读取的 target 区域。所有模型必须从同一个 visible-input view 取特征。

### 3.4 不应计算六任务 accuracy macro average

六个任务的标签空间和协议不同:

- binary classification;
- 3-class ordinal proxy;
- 4-option或多选 closed choice;
- 数千标签 retrieval;
- open structured generation。

把六个 accuracy 直接平均没有统计或构念意义。只能按任务分别报告, 再用预注册的标准化分数做辅助总览。

## 4. T1 `predict_correctness`

### 合理部分

- DBE-KT22、XES3G5M、FoundationalASSIST 都有时间顺序和完整题目;
- train/test 学生隔离;
- FoundationalASSIST 已修正 `discrete_score=0` 不必然表示首答错误的问题;
- 传统 BKT/DKT/AKT/IRT/PFA baseline 与任务相符。

### 限制

- Eedi Response 没有时间戳, 只能称 non-temporal profile-held-out correctness, 不能混称 next-step KT;
- Malrule T1 中全部历史正确几乎决定下一题正确, 全部系统错误历史几乎决定下一题错误, 是受控合成 sanity task, 不能进入真实 T1 headline;
- 测试位置按标签平衡且大多是 warm-item。

### 决定

保留 T1。主结果按来源分别报告 temporal T1; Eedi profile 和 Malrule synthetic 单列。

指标使用 AUROC、AUPRC、accuracy、macro-F1、NLL/Brier。校准指标只在 natural-prevalence test 上解释。

## 5. T2 `predict_response`

当前 T2 实际包含三种不同构念:

1. DBE-KT22/Eedi: 已知选项中的 choice prediction;
2. FoundationalASSIST: 隐私过滤后的结构化数学首答开放生成;
3. Malrule: 合成规则下的可执行错误回答生成。

它们不能共享一个 accuracy 后直接平均。

### 当前问题

- 旧 Qwen evaluator 从每行 candidate set 中打分, 不是开放生成;
- Malrule prompt 列出所有可由 malrule 产生的候选答案, 真实部署中不会拥有这组 oracle-derived responses;
- DBE candidate 文本长度不同, 当前 sum log-prob scoring 有长度偏差;
- specialist 在 DBE/Eedi 上读取了 prompt 未提供的正确答案字段;
- Eedi Response 无时间顺序, 只能解释为 profile-held-out choice。

### 决定

保留 T2, 但拆成:

- `T2-choice`: DBE/Eedi, 输出 option ID/letter, 使用 exact choice accuracy、macro-F1、top-k;
- `T2-open`: FoundationalASSIST/Malrule, 自由生成后做规范化 EM、数学等价、字符或 token 相似度、首答正确性一致率。

Malrule T2 主要用于 C3/C4 和 controlled generalization, 不与真实学生 response accuracy 合并。

## 6. T3 `diagnose_misconception`

### 构念

Eedi-Mining 的题目、正确答案、学生错误答案到专家 misconception 的映射是合理任务。它没有学生历史, 因而是 response-level misconception diagnosis, 不是 longitudinal student-state 预测。

Malrule identity 是受控 synthetic gold, 适合机制、跨模板和 transfer 实验。

### 当前 evaluator 不合理之处

Eedi 每行人为构造 `gold + 31 random negatives`; evaluator 已知 gold 必在 32 个候选中。BGE 0.644 和 Qwen T3 结果只说明 oracle-candidate ranking, 不能代表真实开放诊断。

### 决定

保留 T3, 但主评测改为二条独立协议:

1. global-taxonomy retrieval: 在完整 misconception corpus 中做 Recall@k、MRR、MAP@25;
2. open generation: 不提供候选, 由规范化映射、语义 judge 和人工抽检评估。

当前 32-candidate 结果只保留为 closed-set appendix diagnostic。

## 7. T4 `estimate_mastery`

标签是未来最多 5 次同 KC 表现的正确率分箱, 并非直接观测的 latent mastery。未来样本数至少只需 2 条, 实际时间跨度和题目难度也不固定。

XES test 中 high 占 1,872/2,808; DBE test 中 high 占 94/131。accuracy 很容易被多数类主导。

### 决定

- 代码名可暂时保留, 论文统一改称 `future-KC-performance band`;
- 不作为 C1/C2 headline;
- 放 appendix 或辅助分析;
- 报 macro-F1、balanced accuracy、ordinal MAE、quadratic weighted kappa;
- 同时报告未来机会数、时间跨度和题目难度敏感性。

## 8. T5 `predict_prospective_error`

### 合理部分

任务给定未来题目, 隐藏中间交互, 预测该题上的正确性与具体首答。它测的是 **conditional prospective behavior**, 不是预测学生未来会遇到哪一道题。

FoundationalASSIST 已提供 `Δ=1/3/5`, 共有 4,695 条 test, 是当前最强的 C2 数据基础。

### 限制

- DBE 只有固定 `Δ=3` 和 218 条原子 test; 旧 mixed 进一步只取 200 条;
- 旧 evaluator 是候选选项打分;
- FoundationalASSIST 的开放任务 `candidate_targets` 为空, 现有 candidate evaluator 无法运行;
- FoundationalASSIST 的 Δ 是经过隐私过滤后轨迹的交互距离;
- exact string 会把 `1`, `1.0` 等数学等价回答当作不同。

### 决定

保留为 C2 核心。分别按 source 和 `Δ=1/3/5` 报告:

- correctness AUROC/AUPRC/F1;
- response normalized EM;
- 数学等价或 option accuracy;
- joint exact match;
- 随 Δ 增长的性能衰减。

## 9. T6 `predict_error_persistence`

当前有三个不同任务:

| 来源 | 定义 | test | 判定 |
|---|---|---:|---|
| DBE-KT22 | 同题同一错误选项在未来最多 3 次同题记录中复现 | 12 | 定义好, 规模不可用 |
| XES3G5M | 同题未来再次错误, 无具体选项 | 3,751 原子 test | 只能叫 repeat-incorrectness risk |
| FoundationalASSIST | 同题后续暴露中精确错误首答复现 | 1,084 | 可作 strict secondary |

FoundationalASSIST test 有 165 persistent / 919 transient。accuracy 不能作为主要指标。

### 额外限制

- 只在未来确实存在同题暴露的人群中定义, 有 selection bias;
- “未来 3 次同题记录”没有固定日历时间跨度;
- exact string 对数学等价表达敏感;
- FoundationalASSIST 不记录一次做题过程中的第二/第三次内部尝试, 因而是 later-exposure recurrence。

### 决定

- FoundationalASSIST strict T6 单独报告 AUROC、AUPRC、macro-F1、positive recall、Brier/NLL;
- DBE 只做定性或与 train/valid 合并后的 exploratory analysis, 不做主 test claim;
- XES 改名 `repeat-incorrectness risk`, 不与 strict T6 合并;
- baseline 增加当前错误回答身份、题目×错误回答历史频率和 elapsed-time 特征。

## 10. 当前 baseline matrix 的公平性

### 可以保留

- T1 IRT/PFA/BKT/DKT/AKT, 前提是使用完全相同的 T1 test visible inputs;
- Malrule LinearSVC 作为 synthetic classification baseline;
- majority/question-history 作为 sanity lower bounds。

### 必须修改

- BGE 32-candidate 不能标成开放 T3 baseline;
- T2/T5 specialist 不能读取 Qwen prompt 不可见的 answer key;
- T6 baseline 需要使用 observed wrong response, 否则不是 strict persistence specialist;
- candidate-softmax 的 NLL/ECE 只能解释为候选集合条件概率;
- variable-length candidate 的 sum log-prob 不能直接比较, 应输出等长 option ID 或采用预注册长度处理;
- 不再报告六任务 accuracy macro average。

## 11. P0 修复顺序

1. 新建包含 FoundationalASSIST 的 evaluation manifest, 不覆盖旧 10,544-row matrix。
2. 为所有数据生成 sanitized visible-input rows; gold 字段仅由 evaluator 持有。
3. 实现 T2/T5 free-generation evaluator, 支持 normalized EM、数学等价和 correctness parsing。
4. 将 T3 改为 full-corpus retrieval + open generation; 32-candidate 仅 appendix。
5. 将 T6 拆成 `strict_exact_recurrence` 与 `repeat_incorrectness_proxy` 两个 task/protocol。
6. 增加 natural-prevalence T1 test 和 cold-item test。
7. 让所有 specialist 与 Qwen 使用同一 visible-input schema。
8. 按 task/source/protocol 报告结果, 删除 headline 六任务 macro accuracy。

完成以上 P0 后, 当前任务集合才达到可用于论文主实验的评测标准。
