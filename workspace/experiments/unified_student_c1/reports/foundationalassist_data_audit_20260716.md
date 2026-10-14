# FoundationalASSIST 数据审计与任务构造

日期: 2026-07-16

## 1. 本地来源与使用边界

- 原始目录: `/home/v-yuejiali/FoundationalASSIST/Data`
- 输入表: `Interactions.csv`, `Problems.csv`, `Skills.csv`
- 数据许可: CC-BY-NC-4.0; 仅用于研究与教育用途。
- 数据卡明确提醒 `answer_text` 可能残留 PII。处理后的开放回答仍属于敏感教育行为数据, 不应直接公开或复制到日志。
- 预处理脚本: `scripts/prepare_foundationalassist_tasks.py`
- 输出目录: `data/processed/foundationalassist_core_tasks`

## 2. 原始表审计

| 项目 | 数量 |
|---|---:|
| Interactions 原始行 | 1,753,384 |
| 去重后的 interaction ID | 1,721,943 |
| 删除的重复 interaction 行 | 31,441 |
| 学生 | 5,000 |
| Problems 原始行 | 3,395 |
| 唯一 problem ID | 3,368 |
| Skills 覆盖的 problem ID | 3,368 |
| skill | 224 |

interaction 重复 ID 审计中未发现同 ID 内容冲突, 因而按 ID 保留首行。Problems 有一个 ID 存在字段版本冲突, 预处理按完整行签名的多数版本保留。Skills 对完全相同的映射去重。

## 3. 隐私与质量过滤

处理顺序为 interaction 去重、题目连接、时间戳检查、二值 `discrete_score` 检查、回答隐私过滤。最终保留 1,652,083 条可用于构造轨迹的记录。

| 删除原因 | 数量 |
|---|---:|
| 缺时间戳 | 3,390 |
| 非法或非二值 score | 8,952 |
| 空回答 | 75 |
| 长度超过 128 字符 | 17,400 |
| email/URL/美式电话号码模式 | 313 |
| 超长连续字母 token | 2,664 |
| 非结构化 answer type | 7,883 |
| 非题目选项的自由文本 | 29,183 |

隐私规则先检查 PII 模式, 再允许与题目作者给定选项完全一致的文本。非选项回答仅保留结构化数学回答、单字母变量和少量常见数学词。这个策略是保守过滤, 不能替代人工 PII 审核。

重要限制:

- 处理后 T2/T5 是开放词表的 **结构化数学首答生成**, 不是任意自然语言回答生成。
- JSONL 中仍包含经过过滤的真实首答文本, 必须在受控环境使用。
- 报告与 `dataset_stats.json` 不枚举原始回答; 开放目标只报告数量、长度和规范化哈希去重数。

## 4. `discrete_score` 构念修正

数据卡说明 `discrete_score=1` 表示学生第一次回答正确且在首次作答前未请求支持; `discrete_score=0` 也可能只是请求过提示、查看过答案或经历多次尝试, 不能直接解释为首答错误。

当前构造使用:

- `score=1`: 首答正确;
- `score=0`, `hint_count=0`, `saw_answer=false`: 首答错误;
- 其余 `score=0`: 首答正确性未知, 不作为 T1/T5 目标; 若进入历史, 明确显示为 `unknown because support was used`。

因此 T1 不再把所有零分记录误标成 incorrect。

## 5. 学生划分

- 随机种子: 42
- train/valid/test: 4,000 / 500 / 500 名学生
- 三个集合学生完全不重叠。
- history 长度上限: 12 条。
- T1/T2 每位训练学生最多 8 个位置, 每位 valid/test 学生最多 4 个位置。
- T1/T2 位置按可判定的 correct/incorrect 尽量平衡采样, 因此 T1 评测集不是自然发生率分布。

## 6. 任务定义

### T1 `predict_correctness`

给定最多 12 条历史和当前完整题目, 预测可判定的首答正确性。总计 36,000 条, correct/incorrect 各 18,000 条。

### T2 `predict_response`

在与 T1 相同的位置开放生成隐私过滤后的精确首答。prompt 不列出 gold 或候选回答, `candidate_labels` 与 `candidate_targets` 均为空。

### T5 `predict_prospective_error`

给定截止位置的可见历史和未来题目, 隐藏中间交互, 同时生成未来首答正确性与精确回答。分别构造保留轨迹上的 `Δ=1/3/5`。

`Δ` 是隐私与质量过滤后轨迹中的交互距离, 不是原始表绝对行距离。三个距离分别有 29,397 / 28,361 / 28,074 条样本。

### T6 `predict_error_persistence`

当前事件必须满足:

- 同一题的首答可判定为错误;
- `hint_count=0`;
- `saw_answer=false`;
- 回答非空且通过隐私过滤。

检查该学生未来最多三条同题记录。若规范化后的同一错误文本再次出现, 标为 `persistent`, 否则为 `transient`。

FoundationalASSIST 不包含一次解题过程中的第二或后续尝试文本。因此这里测量的是 **same-question later-exposure exact wrong-response recurrence**, 不是 within-session attempt persistence, 也不能在没有误解等价类标注时直接解释为同一 misconception。

## 7. 最终规模

| task | train | valid | test | total |
|---|---:|---:|---:|---:|
| T1 correctness | 32,000 | 2,000 | 2,000 | 36,000 |
| T2 exact response | 32,000 | 2,000 | 2,000 | 36,000 |
| T5 prospective response | 76,361 | 4,776 | 4,695 | 85,832 |
| T6 exact recurrence | 8,152 | 1,104 | 1,084 | 10,340 |
| 合计 | 148,513 | 9,880 | 9,779 | 168,172 |

T6 中 persistent 1,503 条, transient 8,837 条, 阳性率约 14.54%。该任务必须报告 AUPRC、AUROC、F1/recall 和 majority/persistence baselines, 不能只报告 accuracy。

## 8. 泄漏与一致性验证

全量逐行验证结果:

- 168,172 个 `example_id` 全部唯一;
- train/valid/test 学生集合完全不重叠;
- query 不包含目标回答、正确答案、首答正确性、hint 或 saw-answer 字段;
- metadata 不包含 target correctness 或 target response hash;
- T2/T5 prompt 不含 gold 候选列表;
- T2/T5 的 `candidate_labels` 和 `candidate_targets` 为空;
- T6 prompt 只暴露当前观察到的错误回答, 不暴露未来 recurrence 标签;
- 各 task-specific JSONL 行数与对应 split 总表完全一致;
- 更严格的非选项短自由文本扫描为 0。

## 9. 可复现命令

```bash
/home/v-yuejiali/.venvs/studentmodeling/bin/python3 \
  scripts/prepare_foundationalassist_tasks.py \
  --out-dir data/processed/foundationalassist_core_tasks \
  --history-k 12 \
  --prospective-deltas 1 3 5 \
  --max-train-per-student 8 \
  --max-eval-per-student 4
```

本次全量构造耗时约 111 秒, 峰值内存约 3.6 GB, 输出约 4.6 GB。输出体积包含每个 split 总表和按 task 拆分文件的重复存储。

## 10. 下一步实验

1. Base Qwen3-32B 在同一 test 上做真正自由生成, 禁止候选 log-prob 评测。
2. 分别训练 single-task 与 mixed Qwen3-32B, 与 Base 使用完全相同 prompt 和 split。
3. T1 对比 BKT/DKT/AKT/SAKT 等 correctness specialists。
4. T2/T5 对比带 response vocabulary head 的序列模型, 同时报 exact match、规范化 exact match、token/character 指标和 correctness 指标。
5. T5 按 `Δ=1/3/5` 分开报告。
6. T6 使用 majority、学生历史 recurrence、题目 recurrence、logistic/GBDT 和序列模型 baseline, 并处理 14.54% 阳性不平衡。
