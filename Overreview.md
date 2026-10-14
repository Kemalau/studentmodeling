# 统一生成式学生模型: Claims、Tasks 与当前实验结果总览

日期: 2026-07-16  
实验根目录: `experiments/unified_student_c1`

## 0. 这个实验在做什么

我们要做的是一个 **统一生成式学生模型**。

传统 AI for Education 里有很多分开的学生建模任务:

- knowledge tracing: 预测学生下一题对不对;
- response prediction: 预测学生具体会答什么;
- misconception diagnosis: 诊断学生为什么错;
- near-future KC performance: 预测学生在某知识点上的近期表现档位;
- prospective error prediction: 预测学生未来会错成什么样;
- repeat-incorrectness risk: 预测学生再次遇到同一道题时是否仍会答错。

这些任务过去通常由不同模型分别做。我们的核心想法是:

> 用一个 LLM, 通过统一的 conditional generation 格式, 同时完成这些学生建模任务; 并进一步检查模型内部是否形成了一个可读、可干预的 student state。

统一输入格式大体是:

```text
instruction
student history
<STUDENT_STATE>
query
```

输出统一用结构化标签, 例如:

```text
<correctness>incorrect</correctness>
<predicted_response>2/5</predicted_response>
<misconception>add numerators and denominators</misconception>
```

`<STUDENT_STATE>` 是一个输入 token; 我们关心的是它在某一层的 hidden state, 记作 `z_t`。  
`z_t` 不是模型输出, 也不是持久化保存的全局学生向量; 它是每次给定 history 前向计算时, 在这个 token 位置形成的内部激活向量。

## 1. C1-C4 是什么

C1-C4 是论文层面的四个 claim。代码保留 T1-T6 六个任务, 但论文主分析聚焦 T1/T2/T3/T5; T4 是辅助 proxy。XES3G5M 的 T6 不进入 headline, FoundationalASSIST 的严格 T6 先作为独立 secondary analysis。

### C1: Generality / 通用性

问题:

> 一个统一模型能不能在多个有效、互补的学生建模任务上都做到 competitive?

也就是说, 模型不能只会 KT, 也不能只会 misconception diagnosis, 而是要在 T1 正确性、T2 具体回答、T3 误解诊断和 T5 未来具体回答上同时有效。T4/T6 不再为了“凑齐六任务”而承担 headline claim。

当前状态:

- C1 smoke 已经跑通;
- fully-open mixed C1 是当前 C1 纯性能最强模式;
- 但这还不是 paper-grade, 需要更完整数据、更正式 baseline 和多 seed。

### C2: Capability Frontier / 新能力边界

问题:

> 统一生成式模型能不能做传统专用模型结构上做不了的任务?

典型例子:

- 传统 KT 通常只输出 `P(correct)`;
- 但它不能自然输出“学生未来会错成哪个具体答案”;
- 也不能自然地在同一接口中组合输出未来正确性、具体回答和文字解释。

所以 C2 的核心不是“传统模型理论上绝对做不到”, 而是:

> 传统 KT 通常需要固定标签空间和任务专用输出头; 统一生成式模型能否在同一接口下输出开放词表、结构化、组合式的学生行为?

C2 的主证据来自:

- T2: 生成学生的具体回答, 而不只输出 `P(correct)`;
- T3: 不依赖 row-specific oracle candidates 的开放式误解诊断;
- T5: 跨过隐藏交互后, 同时生成未来正确性和具体回答。

XES3G5M 的 T6 只能测“再次答错风险”。FoundationalASSIST 现已提供带时间顺序的首答文本, 可测同一题在后续暴露中是否精确复现同一错误回答; 该任务仍是二分类 persistence 风险, 不作为开放生成能力的 C2 headline。

当前状态:

- FoundationalASSIST 已完成 T1/T2/T5/T6 全量预处理, 但 Qwen 与 specialist 尚未在该 split 上完成正式评测;
- 当前 T3 evaluator 仍是隐式候选排序, 不能作为开放诊断主结果;
- 原 DBE-KT22 的 T5 测试规模偏小; FoundationalASSIST 新增 4,695 条 T5 test 样本;
- 严格 T6 已有 10,340 个机会样本, 但它测的是跨同题暴露的 exact-response recurrence, 不是平台内部逐次尝试轨迹。

### C3: Transfer / 正迁移

问题:

> 多任务联合训练是不是比单任务训练更好? 任务之间有没有互相帮助?

例如:

- 学 Malrule 的合成误解规则, 会不会帮助真实 Eedi-Mining 误解诊断?
- 学 response prediction, 会不会帮助 misconception diagnosis?
- 学 future-KC-performance, 会不会帮助 correctness?

当前状态:

- 已经有一个比较清楚的 C3 smoke 结果;
- Malrule `identity` 监督能提升真实 Eedi-Mining diagnosis;
- Malrule `response-only` 不提升, 说明不是“多喂合成数据就涨”, 而是特定任务共享了误解规则结构。

### C4: Interpretability / 可解释性

问题:

> 模型内部有没有一个可读、可因果干预的 student state?

C4 分两部分:

- **C4a 可读**: 从 `<STUDENT_STATE>` 位置的 hidden state `z_t` 里, 能不能线性读出学生正在犯哪条 malrule?
- **C4b 可干预**: 把学生 A 的 state swap 到学生 B 的前向计算里, B 的预测错误风格会不会朝 A 翻转?

当前状态:

- latefusion single-task Malrule 模型上, C4 pilot 很强;
- state probe cross-template acc 到 `0.905`;
- masked swap clean flip_to_A 到 `0.494`;
- open-attention swap clean flip_to_A 到 `0.301`;
- open attention 下效果变弱, 因为 B 的 history 和 A 的 swapped state 是冲突证据。

## 2. 任务清单与论文角色

代码继续保留 T1-T6, 以复用已经生成的数据和正在进行的训练。论文不把六个任务视为同等有效的心理或教育构念。

| ID | 代码任务 | 论文中的读法 | 论文角色 | 当前效度判断 |
|---|---|---|---|---|
| T1 | `predict_correctness` | 下一题正确性预测 | 核心性能任务 | 高; 标准 KT 构念 |
| T2 | `predict_response` | 具体回答预测 | 核心性能 + C2 | 高; FoundationalASSIST 有时间戳和首答文本, Eedi mirror 仍只能称 profile-held-out response |
| T3 | `diagnose_misconception` | 开放式误解诊断 | 核心性能 + C2/C3 | 构念高; 当前 oracle-candidate evaluator 只能作辅助 |
| T4 | `estimate_mastery` | 近期 KC 表现档位 | 辅助/appendix | 中低; 标签是未来同 KC 正确率分箱, 不是 mastery 金标 |
| T5 | `predict_prospective_error` | 隐藏若干交互后的正确性 + 具体回答 | C2 核心 | 中高; FoundationalASSIST 支持 `Δ=1/3/5` 和 4,695 条 test |
| T6 | `predict_error_persistence` | 同题后续暴露的错误回答复现 | secondary analysis | FoundationalASSIST 中高; XES 版本仍只是 repeat-incorrectness proxy |

### T1: predict_correctness

预测学生下一题会不会答对。  
这是最接近传统 KT 的任务。

### T2: predict_response

预测学生下一题具体会答什么。  
这比 T1 更细, 因为不仅要预测错不错, 还要预测错成什么。

### T3: diagnose_misconception

给定学生错误答案, 诊断背后的误解。  
当前数据来自:

- Eedi-Mining: 真实专家标注;
- MalruleLib: 合成金标。

### T4: estimate_mastery / future KC performance band

代码名称保留 `estimate_mastery`, 但论文中不把它解释为直接测得的 latent mastery。当前操作化定义是:

> 学生在未来若干道同 KC 题目上的正确率档位。

标签:

```text
<mastery>high</mastery>
<mastery>medium</mastery>
<mastery>low</mastery>
```

### T5: predict_prospective_error

前瞻性具体错误预测。

给模型当前历史和未来第 Δ 道会遇到的题, 中间 Δ-1 次练习不给模型, 让模型预测未来会不会错、错成什么。

这是 C2 的核心任务之一。

### T6: predict_error_persistence

这里必须区分两个协议:

- XES3G5M 没有具体错误答案, 只能测未来同题是否再次答错, 必须称为 `repeat-incorrectness risk`;
- FoundationalASSIST 有首答文本和时间顺序。当前严格标签要求本次为无提示、未查看答案的错误首答, 并检查未来最多 3 次同题记录是否精确复现同一规范化错误文本。

FoundationalASSIST 不记录一次做题过程中的第二/第三次尝试, 因此严格 T6 表示 **same-question later-exposure exact-response recurrence**, 不能写成 within-session attempt persistence, 也不能直接等同于同一潜在 misconception。

## 3. C1-C4 与任务的关系

| 层级 | 含义 |
|---|---|
| T1/T2/T3/T5 | 论文核心任务 |
| T4 | 未来 KC 表现 proxy, 辅助任务 |
| T6 | FoundationalASSIST 严格 recurrence 作 secondary; XES proxy 不进 headline |
| C1 | 一个模型能不能统一完成多个有效学生建模任务 |
| C2 | 统一生成接口是否支持具体回答、开放诊断和未来具体行为 |
| C3 | 多任务联合训练是否产生正迁移 |
| C4 | 模型内部是否形成可读、可干预的 student state |

一句话:

> 代码任务数量不是贡献; 构念有效的任务、能力边界、迁移和机制才是论文 claim。

## 3.1 Paper-grade construct validity 与评测规则

论文主结果必须遵守以下规则:

1. **T2 按来源解释**: 有真实时间顺序的数据才能称 next-response prediction; Eedi Response mirror 只能称 non-temporal profile-held-out response。
2. **T3 主结果必须开放生成**: 输入只含题目、正确答案和学生错误答案; 不允许 evaluator 使用每行必含 gold 的 oracle candidate set 决定输出。候选排序、BGE retrieval 和 MAP/MRR 只能作为单独的 closed-set appendix protocol。
3. **T4 不宣称 mastery measurement**: 当前标签来自未来表现, 因此报告 future-KC-performance band, 并讨论题目难度、运气和练习机会等替代解释。
4. **T5 报告预测距离**: 至少分别报告 `Δ=1/3/5`, 具体回答指标和正确性指标, 并与带 task-specific head 的序列模型比较。
5. **T6 不直接宣称 misconception persistence**: FoundationalASSIST 可报告同题后续暴露的 exact wrong-response recurrence; 若要宣称同一 misconception, 仍需独立误解标注或等价类判定。
6. **所有核心任务同协议比较**: Base Qwen、single-task Qwen、mixed Qwen 和 specialist 使用相同 split 与可见输入; 至少 3 seeds, 报告置信区间。

论文主表建议只放 T1/T2/T3/T5。T4 放辅助分析; FoundationalASSIST 严格 T6 单独报告, 不与 XES proxy 合并, 暂不参与 headline macro average。

2026-07-16 的代码级审计进一步确认: 当前 10,544-row evaluation matrix 不含 FoundationalASSIST, T3 仍是 row-specific oracle-candidate ranking, T2/T5 specialist 读取了 Qwen prompt 不可见的 answer-key 字段, 且 longitudinal test 几乎都是 warm-item。旧矩阵只能作为 pilot, 正式结果必须按 `reports/evaluation_task_validity_audit_20260716.md` 的 P0 清单重建。

## 4. 当前数据状态

当前已有数据:

| 数据 | 用途 | 状态 |
|---|---|---|
| MalruleLib synthetic | C4 state / malrule 诊断 / next response | 已清洗、过滤、可用 |
| Eedi response + OCR | T1/T2/T4 smoke | 已处理、可用 |
| Eedi-Mining | T3 misconception diagnosis / C3 | 已处理、可用 |
| DBE-KT22 | T2/T5 prospective response | 已处理、可用, 但 T5 test 偏小 |
| XES3G5M | T1/T4/repeat-incorrectness | 已处理、可用, 但没有具体错误答案 |
| FoundationalASSIST | 时间序列 T1/T2/T5 + 同题错误回答 recurrence | 已下载、审计并生成 5,000 学生全量 split |

主 processed 数据:

| path | 用途 |
|---|---|
| `data/processed/mixed_c1_ocr_no_val` | fully-open mixed C1 smoke |
| `data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck` | state-token / latefusion 版本 |
| `data/processed/foundationalassist_core_tasks` | FoundationalASSIST T1/T2/T5/严格 T6, 168,172 条 |
| `data/processed/malrule_c1_clean` | Malrule C1/C4 synthetic |
| `data/processed/eedi_mining_c1_ir` | Eedi-Mining T3 |
| `data/processed/transfer_eedi_malrule_ocr_no_val` | C3 ablation |
| `data/processed/mixed_t1_t6_balanced` | 当前统一训练与全量评测数据; 论文需按核心/辅助任务拆报 |

## 5. 已做实验与结果

## 5.1 C1: 统一多任务 smoke

当前 C1 评估任务包括:

- `diagnose_misconception`
- `future_mastery`
- `malrule_identity`
- `malrule_next_correctness`
- `malrule_next_response`
- `next_correctness`
- `next_response`

主要训练模式:

| 模式 | 平均 acc | 结论 |
|---|---:|---|
| Qwen3-32B base | `0.594` | base 参考 |
| fully-open mixed C1 S120 | `0.647` | 当前 C1 纯性能最好 |
| latefusion mixed C1 S120, open eval | `0.588` | C1 下降, 但保留 state 设计 |
| latefusion mixed C1 S120, masked eval | `0.472` | bottleneck 对 mixed C1 伤害较大 |
| mixed-mask p=0.5 S120, open eval | `0.621` | C1 部分恢复, 但 C4 control 没保住 |

详细任务准确率:

| task | base | fully-open | latefusion-open | mixedmask-p50-open |
|---|---:|---:|---:|---:|
| `diagnose_misconception` | `0.460` | `0.500` | `0.487` | `0.440` |
| `future_mastery` | `0.413` | `0.467` | `0.410` | `0.443` |
| `malrule_identity` | `0.527` | `0.593` | `0.573` | `0.563` |
| `malrule_next_correctness` | `0.813` | `0.977` | `0.927` | `0.993` |
| `malrule_next_response` | `0.627` | `0.780` | `0.697` | `0.807` |
| `next_correctness` | `0.673` | `0.563` | `0.490` | `0.500` |
| `next_response` | `0.643` | `0.647` | `0.533` | `0.600` |

当前 C1 结论:

- fully-open 训练最适合 C1 任务性能;
- latefusion/bottleneck 会降低 C1, 尤其是真实 Eedi log 任务;
- mixed-mask p=0.5 能恢复一部分 C1, 但没有保住 C4 的可控 state。

关键文件:

- `results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter`
- `results/mixed_c1_ocr_lte1536_FULLYOPEN_train_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`
- `reports/mixed_c1_full_open_vs_latefusion_open_20260712.md`
- `reports/mixed_c1_mixedmask_p50_20260712.md`

## 5.2 C2: 新能力边界

正式 C2 尚未完成, 但不再把它等同于“T5/T6 两个标签任务”。

论文问题是:

> 相比只输出 `P(correct)` 的标准 KT, 统一生成式学生模型能否在不增加任务专用输出头的情况下, 生成具体回答、开放式误解解释以及跨隐藏交互的未来行为?

当前证据:

- T2 已证明统一格式可以输出具体 response, 但必须区分 temporal 与 non-temporal 数据源;
- T3 已有 Eedi/Malrule 结果, 但当前候选 logprob 排序不是开放生成主评测;
- T5 已有 DBE-KT22 pilot; FoundationalASSIST 已补齐 `Δ=1/3/5` 和更大测试集, 正式模型评测待跑;
- FoundationalASSIST T6 已能测同题后续暴露的 exact wrong-response recurrence; XES T6 仍只能测 repeat incorrectness。

下一步:

1. T2/T3/T5 全部增加真正 free-generation evaluation;
2. 在 FoundationalASSIST 上分别报告 T5 `Δ=1/3/5`;
3. 比较标准 KT、带 task-specific response head 的序列模型和统一生成模型;
4. 做 held-out response / held-out misconception, 检查开放词表泛化;
5. 单独评测 FoundationalASSIST strict T6, 并明确其 later-exposure 而非 within-session attempts 的边界;
6. 表述为“传统模型通常不原生支持、需要任务专用扩展”, 不写成理论上绝对做不到。

## 5.3 C3: 正迁移

C3 做了一个 targeted ablation:

> Malrule 合成监督是否帮助真实 Eedi-Mining misconception diagnosis?

目标任务固定为 Eedi-Mining `diagnose_misconception`, `n=300`。

| 训练条件 | acc | Δ acc | macro-F1 | Δ F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Eedi-only | `0.437` | `+0.000` | `0.269` | `+0.000` | `6.551` | `0.418` |
| Eedi + Malrule identity | `0.470` | `+0.033` | `0.314` | `+0.045` | `6.286` | `0.393` |
| Eedi + Malrule response | `0.423` | `-0.013` | `0.271` | `+0.003` | `6.474` | `0.427` |
| Eedi + identity + response | `0.487` | `+0.050` | `0.314` | `+0.045` | `5.961` | `0.386` |
| all mixed C1 | `0.493` | `+0.057` | `0.320` | `+0.051` | `6.374` | `0.379` |

当前 C3 结论:

- 有初步正迁移;
- 迁移主要来自 Malrule `identity` / 误解规则监督;
- `response-only` 不提升, 说明不是简单“多喂合成数据”;
- 这支持“任务之间共享了误解规则 latent”的说法, 但还只是 smoke。

关键文件:

- `reports/c3_eedi_malrule_ablation_20260712.md`
- `results/eedi_mining_plus_malrule_identity_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`
- `results/eedi_mining_plus_malrule_response_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`
- `results/eedi_mining_plus_malrule_ir_ocr_no_val_qwen3_32b_qlora_s120_len1536_diagnose_n300.json`

## 5.4 C4: Student State 可读与可干预

### 最强 C4 模型

当前最强 C4 模型:

```text
results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter
```

训练设置:

- base: `/data1/liyuejia/models/Qwen3-32B`
- task: `malrule_next_response`
- data: `data/processed/malrule_c1_clean/train.jsonl`
- max length: `1536`
- steps: `800`
- LoRA: r=8, alpha=16
- special token: `<STUDENT_STATE>`
- mask: latefusion bottleneck

### 任务性能

Malrule same-template `next_response`, `n=300`:

| test attention | accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| masked latefusion | `0.880` | `0.785` | `0.300` | `0.039` |
| open standard | `0.867` | `0.744` | `0.284` | `0.032` |

结论:

- latefusion 模型在 masked 下表现很好;
- test-time 打开 history 后, acc 只掉 `1.3` 个点;
- 所以 “训练 bottleneck, 测试 open” 在任务性能上是可行的。

### C4a: probe 可读性

线性 probe 读取 `<STUDENT_STATE>` hidden state 中的 malrule identity:

- label classes: `22`
- best cross-template acc: `0.905`
- best layer: `48`
- chance: 约 `0.045`

结论:

> `z_t` 里强烈编码了学生错误规则, 且能跨模板迁移。

这说明它不是只读题面表面特征。

### C4b: state swap 可干预性

实验:

> 把学生 A 的 state swap 到学生 B 的前向里, 看 B 的预测错误是否翻成 A 的错误风格。

clean subset = none 条件下模型本来预测 B 的 malrule answer。

| test attention | clean n | condition | flip_to_A | stay_B | other | changed top-1 |
|---|---:|---|---:|---:|---:|---:|
| masked | 89 | noop | `0.000` | `1.000` | `0.000` | `0.000` |
| masked | 89 | samemal | `0.045` | `0.865` | `0.090` | `0.135` |
| masked | 89 | random | `0.292` | `0.506` | `0.202` | `0.494` |
| masked | 89 | swap | `0.494` | `0.225` | `0.281` | `0.775` |
| open | 93 | noop | `0.011` | `0.989` | `0.000` | `0.011` |
| open | 93 | samemal | `0.065` | `0.871` | `0.065` | `0.129` |
| open | 93 | random | `0.140` | `0.699` | `0.161` | `0.301` |
| open | 93 | swap | `0.301` | `0.473` | `0.226` | `0.527` |

结论:

- masked latefusion 下, swap 有强因果效果: `0.494` flip_to_A;
- open attention 下, swap 仍然有效, 但变弱: `0.301` flip_to_A;
- 变弱是合理的, 因为 B 的 history 和 A 的 swapped state 是冲突证据;
- noop 几乎不变, 说明 hook/patch 管线基本可信。

### C4 负结果与教训

| 模式 | 结果 |
|---|---|
| fully-open state swap | history 旁路太强, state 难以控制输出 |
| standard bottleneck | probe 可读, 但 swap 不如 latefusion |
| mixed-mask p=0.5 | C1 部分恢复, 但 C4 controllability 消失 |
| response-only C3 | 不提升 Eedi diagnosis |

当前 C4 结论:

> latefusion single-task 模型中, `<STUDENT_STATE>` 已经表现出“可读 + 部分可控”的性质。  
> 但一个同时兼顾 C1 高性能和 C4 强可控的统一模型还没解决。

关键文件:

- `reports/c4_latefusion_open_attention_swap_20260712.md`
- `results/c4_response_latefusion_s800_probe_malrule_layers_smoke.json`
- `results/c4_response_latefusion_s800_swap_response_all_layers_n150_controls.json`
- `results/c4_response_latefusion_s800_swap_response_OPEN_all_layers_n150_controls.json`

## 6. 当前总判断

| claim | 当前状态 | 读法 |
|---|---|---|
| C1 | 已做 smoke | fully-open mixed C1 当前最好, 但还需正式 baseline 和 full data |
| C2 | 数据已补齐, 正式模型实验未完成 | 缺开放生成、specialist 对照和 open-vocabulary 泛化评测 |
| C3 | 有初步正迁移 | Malrule identity 帮 Eedi diagnosis |
| C4 | 有强 pilot | latefusion state 可读且部分可干预 |

| task | 当前状态 |
|---|---|
| T1 `predict_correctness` | 已做 smoke |
| T2 `predict_response` | 已做 smoke |
| T3 `diagnose_misconception` | 已做 smoke, Eedi + Malrule |
| T4 `estimate_mastery` | 已做 future-KC-performance proxy; 仅作辅助 |
| T5 `predict_prospective_error` | FoundationalASSIST 已有 `Δ=1/3/5` 全量 split; 待正式训练评测 |
| T6 `predict_error_persistence` | FoundationalASSIST strict recurrence 可作 secondary; XES 仍是 proxy |

## 7. 迁服后建议

不要继续无限调同一个 smoke setup。下一步应该切成四条线:

1. **C1 正式化**
   - 用 fully-open mixed C1 作为当前 recipe;
   - 论文主表聚焦 T1/T2/T3/T5, 不再计算包含低效度 T4/T6 的 headline macro average;
   - 补 Base Qwen、single-task Qwen 和正式 specialist baselines;
   - full test, 多 seed, 置信区间。

2. **C3 正式化**
   - 做 exposure-matched ablation;
   - 多 seed;
   - full Eedi-Mining test;
   - 尽量接官方 MAP@25 retrieval protocol。

3. **C4 定版**
   - 以 latefusion single-task 为主;
   - 补更完整 probe 控制;
   - 补更多 swap pair / seed;
   - 明确写成: bottleneck/latefusion 产生可读可控 state, open history 会削弱但不消除因果作用。

4. **C2 正式实验**
   - 先完成 T2/T3/T5 的 open-generation protocol;
   - 在 FoundationalASSIST 上跑 T2/T5 open generation 与 `Δ=1/3/5`;
   - 做 held-out response / misconception 泛化;
   - 单独跑 strict T6, 不与 XES repeat-incorrectness 合并;
   - 当前最大缺口已从数据变成正式模型与 specialist evaluation matrix。

## 8. 一句话版本

当前实验已经证明:

> 统一生成式学生建模可以跑通多个任务; Malrule 误解规则监督对真实 Eedi diagnosis 有正迁移; latefusion 训练能产生一个跨模板可读、并且通过 swap 可部分控制输出的 student state。

当前还没有证明:

> 一个单一模型同时达到最佳核心任务性能和最强 C4 可控性; 也还没有证明开放式误解诊断、开放词表未来回答生成或真正的同一错误模式持续性。
