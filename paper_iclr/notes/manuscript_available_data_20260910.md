# 当前稿件可用数据清单

核对日期：2026-09-10。范围：本服务器 studentmodeling 工作区内已归档的结果及报告。没有提交训练、重新生成答案、调用新裁判或查询云端作业实时状态；未同步到本机的云端结果不在本次完整性声明内。原论文和插图未修改。

## 1. 先区分“已有数据”和“能直接填入新主表”

服务器已有五任务正式测试矩阵、任务单训对照、专用基线、部分生成指标、较新的 KT 基线、C3 配对区间及 C4 机制结果。但新版主表的 EM / NM / SA / BS-F1 不是已有任务主指标矩阵，不能把 Accuracy、MathEq、JointMath 换个列名直接填进去。

- 主结果和任务单训矩阵共 21,676 个测试例：正确性 10,913；Written response 3,693；Diagnosis 1,494；Future response 4,600；Persistence 976。
- 以下 Unified 是正式矩阵中的 `Unified Qwen3-32B QLoRA` 行，不能仅因稿件改名就认定它已经验证了所写的 late-fusion learner-state 架构。
- 现有报告明确区分 unrestricted/full-open Unified、原 CF1 one-shot 和后续 A/B checkpoint。必须保留各自模型身份。
- 当前 T3 语义分数使用 TRAPI 实际模型 `gpt-4o_2024-11-20`，rubric `t3-semantic-equivalence-v2-substantive-schema`。不要把配置字段中的 `gpt-5.5` 当作实际裁判模型。
- 本次只读出已有裁判结果，没有重新判断文本语义。未新增去退化后的语义评估，不把已有全样本分数表述成完成了新的退化审计。

## 2. 正式五任务主指标

| 模型 | Correctness Acc | Written MathEq | Diagnosis SA | Future JointMath | Persistence Acc |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-32B | 0.544580 | 0.223666 | 0.634538 | 0.246304 | 0.841189 |
| Unified Qwen3-32B QLoRA | 0.687529 | 0.518278 | 0.566265 | 0.766739 | 0.861680 |
| Task-only Qwen3-32B QLoRA | 0.691652 | 0.505551 | 0.545515 | 0.769783 | 0.862705 |
| 旧版传统 specialist suite | 0.672043 | 0.464121 | 0.348059* | 0.760435 | 0.859631 |

来源：[正式 v3 矩阵](../../workspace/experiments/unified_student_c1/results/taskonly_qwen_formal_eval_20260813/formal_specialist_v3/formal_evaluation_matrix.json)。

*specialist 的 T3=0.348059 是按来源拼接的分类准确率：Eedi 的 BGE 与 MalRule 的 LinearSVC 加权。它不是统一语义裁判 SA，也不是一套单模型结果，不能放在新版 SA 列里伪装成同协议。

## 3. 附录目前能补的空格

### 3.1 Correctness prediction

| 模型 | n | Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| Base Qwen3-32B | 10,913 | 0.544580 | 0.474429 | 0.633796 |
| AKT | 10,913 | 0.672043 | 0.671627 | 0.742101 |
| Unified | 10,913 | 0.687529 | 0.687526 | 0.760078 |
| Rasch IRT | 10,913 | 0.640246 | 0.640182 | 0.704415 |
| PFA | 10,913 | 0.620269 | 0.620122 | 0.678952 |
| DKT item-LSTM | 10,913 | 0.669202 | 0.668919 | 0.730981 |

Base/Unified 的 Macro-F1 是本次复用仓库正式 `classification_summary` / `binary_metrics` 对脱敏分类预测汇总所得，不是重新推理。四个分类单元的 n、Accuracy、AUROC 已与正式矩阵逐项核对。

来源：[恢复的总体分类指标及预测哈希](server_classification_scores_20260910.json)、[AKT](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/akt.json)、[IRT/PFA](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/irt_pfa.json)、[DKT](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/dkt.json)。BKT 已有 Accuracy=0.541142、AUROC=0.590199，但只有 9,929 个 temporal/synthetic 测试例，不能标为完整 10,913。

### 3.2 Written response

| 模型 | n | Response MathEq | Normalized exact match |
|---|---:|---:|---:|
| Base Qwen3-32B | 3,693 | 0.223666 | 0.215272 |
| Conditional-response baseline | 3,693 | 0.464121 | 0.446791 |
| Unified | 3,693 | 0.518278 | 0.507717 |
| Task-only | 3,693 | 0.505551 | 0.497157 |
| Response-aware neural specialist | 3,693 | 0.434064 | 0.416734 |

这里 NM 为现有 `normalized_exact_match`，MathEq 为数学等价判定，二者不能混称。不是全答案序列原样 EM。

来源：[数值清单](server_data_inventory_20260910.json) 中对应 `base_predict_response`、`adapter_predict_response`、Task-only T2 的 `pooled_additive_metrics`；[传统任务基线](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/task_specialists.json)；[神经 T2](../../workspace/experiments/unified_student_c1/results/neural_response_specialists_20260814/final/T2/metrics.json)。

### 3.3 Misconception diagnosis

同一独立语义裁判口径：

| 模型 | Eedi SA (894) | MalRule 同模板 SA (300) | MalRule 跨模板 SA (300) | MalRule 合并 SA (600) | 总体 SA (1,494) | 总体 NM |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3-32B | 0.667785 | 0.590000 | 0.580000 | 0.585000 | 0.634538 | 0.046185 |
| Unified | 0.496644 | 0.690000 | 0.650000 | 0.670000 | 0.566265 | 0.050201 |
| Task-only | 0.445190 | 0.673333 | 0.716667 | 0.695000 | 0.545515 | 0.087684 |

来源：[Base T3 v2](../../workspace/experiments/unified_student_c1/results/formal_specialist_v2/unrestricted_qwen_t3_v2/base_diagnose_misconception/metrics.json)、[Unified T3 v2](../../workspace/experiments/unified_student_c1/results/formal_specialist_v2/unrestricted_qwen_t3_v2/adapter_diagnose_misconception/metrics.json)、[Task-only T3](../../workspace/experiments/unified_student_c1/results/taskonly_qwen_formal_eval_20260813/T3/t3_semantic/metrics.json)。

分类/检索基线的独立口径：

| 模型/来源 | Accuracy | Macro-F1 | MRR | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|
| BGE-small / Eedi | 0.110738 | 0.053866 | 0.204426 | 0.285235 | 0.392617 |
| TF-IDF + LinearSVC / MalRule | 0.701667 | 0.727610 | 未提供 | 未提供 | 未提供 |
| TF-IDF + LinearSVC / Eedi | 0.051454 | 0.016372 | 未提供 | 未提供 | 未提供 |

来源：[T3 specialist 原始结果](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/t3_specialists.json)。LLM 的全 taxonomy 检索在结果中仍标记 `full_taxonomy_retrieval_pending=true`；不能据 SA 推出 MRR 或 Macro-F1。

### 3.4 Future response

| 模型 | n | JointMath | Correctness Acc | Response MathEq | Joint exact match | Response NM |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3-32B | 4,600 | 0.246304 | 0.379565 | 0.264783 | 0.231522 | 0.249130 |
| PFA + conditioned response | 4,600 | 0.760435 | 0.797174 | 0.760870 | 0.736087 | 0.736304 |
| Unified | 4,600 | 0.766739 | 0.799565 | 0.768913 | 0.742174 | 0.744130 |
| Task-only | 4,600 | 0.769783 | 0.800870 | 0.771522 | 0.746304 | 0.747826 |
| Neural specialist | 4,600 | 0.760435 | 0.795217 | 0.762609 | 0.735435 | 0.737391 |

以上保留原始指标名。`joint_exact_match` 是现有字段级协议指标，不自动等于新版表格的全输出 EM。分来源/距离的数值也在导出的 TSV 中。

来源：[数值清单](server_data_inventory_20260910.json) 中 `base_predict_prospective_error`、`adapter_predict_prospective_error` 与 Task-only T4；[传统任务基线](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/task_specialists.json)；[神经 T4](../../workspace/experiments/unified_student_c1/results/neural_response_specialists_20260814/final/T4/metrics.json)。旧文件的 T5 在这里对应公开任务 T4。

### 3.5 Error persistence

| 模型 | n | Accuracy | Macro-F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-32B | 976 | 0.841189 | 0.481220 | 0.642749 | 0.357657 |
| Logistic baseline | 976 | 0.859631 | 0.620083 | 0.717378 | 0.454680 |
| Unified | 976 | 0.861680 | 0.635231 | 0.677613 | 0.425801 |

来源：[恢复的分类指标](server_classification_scores_20260910.json)、[传统任务基线](../../workspace/experiments/unified_student_c1/results/formal_specialist_v1/task_specialists.json) 的旧 T6。正类是 persistent，总体正类比例 0.162910。Unified 的 AUROC/AUPRC 不如 Logistic，不能对这些列也加粗为最佳。

## 4. 更新的正确性基线

下列本地指标均为 `status=complete`、n=10,913；目录名包含 partial 不代表该指标文件只评了一部分。只列本次实际找到的 seed，不暗示其余 seed 已完成。

| 模型 | seed | Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| simpleKT | 123 | 0.672409 | 0.672404 | 0.743921 |
| UKT | 42 | 0.674792 | 0.674644 | 0.745256 |
| UKT | 97 | 0.676807 | 0.676227 | 0.742352 |
| DTransformer | 42 | 0.669202 | 0.668671 | 0.733421 |
| DTransformer | 123 | 0.676349 | 0.676132 | 0.740791 |
| ReKT | 42 | 0.681389 | 0.680766 | 0.755340 |
| ReKT | 97 | 0.681023 | 0.680644 | 0.752091 |
| ReKT | 123 | 0.682031 | 0.680743 | 0.752674 |

ReKT 三 seed Accuracy 均值/样本 SD 为 0.681481 / 0.000510；AUROC 为 0.753369 / 0.001732。三模型 ensemble Accuracy=0.684230、AUROC=0.755013，不能当成第四个独立 seed。Unified 对各 ReKT seed 的 Accuracy 配对区间均跨零。

来源：[ReKT confirmatory summary](../../workspace/experiments/unified_student_c1/results/modern_kt_t1_rekt_confirmatory_20260824/summary.json)，其他逐 seed 指标的完整路径、状态和哈希见[来源索引](server_data_inventory_20260910.md)。现代基线的 real-only / real-temporal 分组也在 JSON/TSV 中。

## 5. Cross-task transfer 可用结果

| 能力 | Task-only | Unified | 差值 | 95% 区间 |
|---|---:|---:|---:|---|
| Correctness | 0.691652 | 0.687529 | -0.004124 | [-0.010171, +0.001924] |
| Written response | 0.505551 | 0.518278 | +0.012727 | 现有 Unified 逐题记录缺 raw_output/target，未提供可用配对区间 |
| Diagnosis pooled | 0.545515 | 0.566265 | +0.020750 | [-0.006024, +0.046854] |
| Eedi only | 0.445190 | 0.496644 | +0.051454 | [+0.015713, +0.087444]，question-clustered |
| Future response | 0.769783 | 0.766739 | -0.003043 | [-0.006957, +0.000870] |
| Persistence | 0.862705 | 0.861680 | -0.001025 | [-0.009221, +0.007172] |

Eedi 的 398/894 对 444/894，本次重跑现有只读审计，核对 example ID、来源哈希、裁判一致性并复算 373 道题聚类的 20,000 次 bootstrap。其余全任务区间是逐样本配对区间，不是训练 seed 标准差。没有匹配总更新数；Eedi 是事后子集分析。

负结果也有：MalRule 跨模板 diagnosis 为 0.716667 -> 0.650000，差 -0.066667；T1+T2 add-one 对 T1-only 的差为 -0.005131，95% CI [-0.011271, +0.001008]。不能只保留正迁移子集。

来源：[正式矩阵](../../workspace/experiments/unified_student_c1/results/taskonly_qwen_formal_eval_20260813/formal_specialist_v3/formal_evaluation_matrix.json)、[Eedi 审计脚本](../../workspace/experiments/unified_student_c1/scripts/audit_c3_eedi_question_transfer_20260907.mjs)、[T1+T2 对照](../../workspace/experiments/unified_student_c1/results/t1_t2_paired_addone_recovery_20260901/comparison.json)。

## 6. C4 现有机制数据

| 协议/模型 | Directed | Control | 差值 | 区间/备注 |
|---|---:|---:|---:|---|
| 原 CF1，候选，real-random 对照 | 39/95 | 16/95 | 0.242105 | [0.126316, 0.357895] |
| 原 CF1，候选，Gaussian noise 对照 | 39/95 | 18/95 | 0.221053 | 不是稿件中由舍入值相减得到的 0.222 |
| 原 checkpoint，新无候选协议 | 4/96 | 3/96 | 0.010417 | [-0.0312, 0.0521] |
| A，无候选训练 | 56/96 | 6/96 | 0.520833 | [0.3021, 0.7396]，pair-clustered |
| B，另加 state-exchange | 56/96 | 8/96 | 0.500000 | [0.2604, 0.7396]，pair-clustered |

- 原 CF1 最佳跨模板 probe=0.933，为九个被测层中的最佳层，不是固定层预注册结果。
- A/B 同状态三题全中均为 14/32，对照均 0/32；B-A=-0.020833，不支持额外 exchange loss 的增量优势。
- 未见规则三类分别是 offset、composition、conditional：每类三题全中为 0/8，三类合计 0/24；已训练对照为 8/8。稿件若写整体新规则 0/8，必须明确它只对应单类分母。
- 历史 3/6/12 条及 3 条重复四遍诊断已完整归档，共 5,376 输出；未见规则仍未恢复三题一致迁移。
- 原候选旧 seed 的 D 为 0.188、0.221、0.015；修复初始化后已完成新 seed13=-0.020、seed97=0.272。新 seed42 没有完整结果，不把旧 seed42 拼成新三 seed。
- 仅末层 output patch 的零效应不能证明单向量状态不可能有效，原实现存在读取路径限制。

来源：[9 月 7 日 C3/C4 汇总及逐项来源](../../workspace/experiments/unified_student_c1/reports/c3_c4_effective_results_20260907.md)、[real-random 原始汇总](../../workspace/experiments/unified_student_c1/results/c4_real_random_dose_seed42_20260904.summary.json)、[A/B 执行结果](../../workspace/experiments/unified_student_c1/reports/c4_execution_training_results_20260906.md)、[未见规则结果](../../workspace/experiments/unified_student_c1/reports/c4_unseen_rules_results_20260906.md)、[历史长度结果](../../workspace/experiments/unified_student_c1/reports/c4_history_diagnostic_results_20260907.md)。C4 数值为候选标签或可执行规则答案命中，不是通用语义质量 SA。

## 7. 辅助 future-KC 指标

现有 n=2,939（DBE 131、XES 2,808）。class-balanced ordinal PFA：Accuracy 约 0.540，Macro-F1 约 0.439，Balanced accuracy 约 0.487，QWK 约 0.260。已有 student-cluster bootstrap；这些不是 latent mastery 金标，不能移入五任务主表。

来源：[metrics](../../workspace/experiments/unified_student_c1/results/public_t6_future_kc_baseline_20260814/metrics.json)、[bootstrap](../../workspace/experiments/unified_student_c1/results/public_t6_future_kc_baseline_20260814/bootstrap.json)。完整精度和其他基线见导出清单。

## 8. 仍未找到/不能替代的项目

| 稿件需要的项目 | 本地检查结论 |
|---|---|
| 所有模型 x 五任务 x 四统一指标 | 没有发现完整的 EM/NM/SA/BS-F1 同协议矩阵 |
| BERTScore-F1 | 在 results 中 metrics/matrix/summary JSON 文件检索未发现 BERTScore 字段；本次未运行 BERTScore |
| 全输出原样 EM | 当前标签分类与字段级 NM/joint exact match 不能直接替代 |
| 全任务 SA | 独立语义裁判目前可确认的是 T3；其余是分类/数学等价等协议 |
| Qwen 诊断 MRR/Macro-F1 | v2 T3 结果仍标记 full-taxonomy retrieval pending |
| T2 paired CI | 原 Unified 脱敏逐题数据缺 raw_output/target，不能伪造配对比较 |
| Unified 多训练 seed 的 std | 本次正式矩阵为单训练 seed，ReKT std 不能移植给 USSM |
| 新严格 CF1 seed42 | 本机报告没有完整结果，不能声称严格三 seed 已收齐 |

## 9. 与本稿直接相关的修正提醒

1. Base T3 总体 0.634538、Eedi 0.667785 高于 Unified 的 0.566265 / 0.496644。必须报告，不能用 Task-only 的较低结果替代 Base 行。
2. “all five pooled capabilities” 包含 MalRule 合成例；Eedi-Mining 无 longitudinal student history。把 pooled 与 real-only、状态机制证据分开描述。
3. Unified 正式性能与 CF1/A/B 机制来自不同 checkpoint。现有材料不足以把所有最好数值绑定到同一个 late-fusion USSM。
4. Persistence 的 AUROC/AUPRC 是 Logistic 更高，不是 Unified 所有指标都更强。
5. 稿件 `vs. noise` 的差值应按 39/95 - 18/95 = 0.221053，而不是先舍入 0.411 - 0.189 得到 0.222。
6. Eedi 2587 对实际 BGE 2586 标签、MalRule 文本 21 对现有 classifier 22 标签存在统计口径差异，需要核对，不直接改成同一个数。
7. 附录仍引用 `tab:joint`，但粘贴稿中该表已替换为 `fig:cross-task-transfer`，这些引用需要后续统一。

## 10. 数据包与复验

- [可读来源索引与正式矩阵](server_data_inventory_20260910.md)
- [JSON 指标包](server_data_inventory_20260910.json)：71 个来源文件；包含原始指标路径、来源哈希、状态、裁判信息和可加指标的样本量加权汇总。
- [TSV 指标包](server_data_inventory_20260910.tsv)：用于按任务、来源、模型和指标路径筛选；历史/诊断条目不自动等于可发论文的主结果。
- [恢复的总体分类指标](server_classification_scores_20260910.json)：只含汇总统计和来源哈希，不含学生逐题记录。

在仓库根目录运行：

```bash
python workspace/experiments/unified_student_c1/scripts/inventory_classification_scores_20260910.py
node workspace/experiments/unified_student_c1/scripts/inventory_paper_results_20260910.mjs
```

Eedi 独立复验须在实验根目录运行 `node scripts/audit_c3_eedi_question_transfer_20260907.mjs`。本次已执行成功。