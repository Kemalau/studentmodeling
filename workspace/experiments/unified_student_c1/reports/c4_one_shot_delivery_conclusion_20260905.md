# C4 one-shot 收尾结论（2026-09-05）

## 1. 问题与非目标

结论：当前证据支持候选支持条件下的学生状态因果效应；本次补实验不足以升级为可靠、可组合的规则执行器。没有训练新模型，没有重启 recurrent，也不以跑出正结果为停止标准。

本次接续既有 `c4-rule-followup-seed42-v2-20260905`，重新核验全部原始记录和代码。实验输出完整性与集群终态分别记录，不能混称。

## 2. 样本与具体实例

沿用 seed-42 CF1 vision-pad checkpoint。原面板 27 个 donor、81 道新数组合题；新颖性只针对已检查的微调/评测文件，不涵盖预训练。来源文件行数为 CF1 train/valid 6,000/600、MalRule train/valid/same-test/cross-test 9,000/900/900/900，不是互不重叠的独立样本数。

显式规则实验保留全部 81 题，同时按推理前的语义检查报告有效 69 题；12 题涉及规则不适用或标签不一致。固定候选实验用 23 个有效 donor，每个两题，两种候选顺序。两题展示相同的 A/B/数学正确答案并集，不提供当前正确答案标注。所有 23 个 donor 的目标答案均通过全部干预历史的答案排除检查。

实例：`11/20 × 4/20` 和 `15/19 × 8/19`，donor 的忽略分母规则目标分别为 44 和 120。两题都展示 44、120 及其他候选。

## 3. 干预坐标与指标

固定候选实验把 donor 的 64 个 decoder-block 输出 residual，逐层替换到 recipient 的同一 vision-pad token 地址，每层 5120 维。状态只从 instruction/history 前缀抽取，不读取新问题或候选。无层/神经元搜索。

`J(g,o,c) = 1[题0输出=A0 且 题1输出=A1]`

`D_joint = mean_(g,o)(J(g,o,directed) - J(g,o,real_random))`

其中 g 为 donor，o 为候选顺序，c 为干预条件。旧候选可见/不可见诊断使用逐题 A 命中率差，不可与 joint 指标或历史 0.411 flip 直接混比。

## 4. 数值符号例子

在 46 个题对×顺序条件中，一个 directed-only 联合成功贡献 +1/46；一个 random-only 成功贡献 −1/46。两题始终选 44，虽然单题命中一半，但联合成功为零。

## 5. 真实成功与失败

Group 1 的 stops-borrow-at-zero donor 在两种顺序下，对 `59009−47173`、`3001−1650` 分别输出规则目标 11976、1451；存在个别成功。

Group 0 的忽略分母 donor：一种候选顺序两题都输出 44，反序后两题都输出 120，而非随问题分别输出 44/120。成功和失败均保留，不能用成功案例替代整体结果。

## 6. 样本筛选与完整性

显式规则：27 donor × 3 题 × 3 条件 = 243 输出。

固定候选：23 donor ×（2 题 + 无问题对照）× 2 顺序 × 6 状态条件 = 828 输出。

共 1,071 条唯一预期记录。三个分片的原始输出、逐 donor 文件、prompt hash、状态 hash、adapter/code/tokenizer lineage、token 解码和严格解析已重新核验。旧格式诊断的 3,240 条记录也重新聚合，与原结果精确一致。

## 7. 因果对照

保留 native、own-state no-op、directed、另一段同 donor 规则历史、另一段同 recipient 规则历史、规则适用于两题的 real-random 历史。额外历史使用相同 response-task 指令且内容不同。

部分 random 规则答案不在固定候选并集中；支持集敏感性在完整英文报告单独保留。旧格式诊断的 random 仅 family-matched，不能声称它满足新实验的语义适用性标准。

## 8. 干预与剂量

`h_B[layer,state] <- h_source[layer,state]`，全量替换，无剂量调参；同一 source 张量复用于两题、两顺序及无问题对照。每次生成使用新 KV cache。历史物理上仍存在但受原 late-fusion mask 限制，本实验不是严格 history-free bridge。

## 9. 评测协议

Qwen3-32B、原 seed-42 adapter、NF4/bfloat16/eager，greedy、最多 48 token、EOS/闭合标签停止，无答案预填或词表约束。严格单数字/响应标签解析，精确有理数匹配，非法输出计失败。显式规则为一般算法说明，无当前答案或算例；这是指令诱发测试，不是理论 oracle 上界。

## 10. 核验结果

| 检验 | Directed / donor 描述 | 对照 | 结论 |
|---|---:|---:|---|
| 原格式诊断：候选可见、不给正确答案，生成 A | 61/81 | real-random 29/81 | 差 +0.3951，95% donor-cluster CI [0.2222, 0.5556] |
| 原格式诊断：候选与正确答案均隐藏，生成 A | 3/81 | real-random 3/81 | 差 0，CI [−0.0370, 0.0370] |
| 显式规则、无候选，有效算术题 A 命中 | 11/69 | 无规则 1/69 | 描述有帮助，但绝对执行率仅 15.9% |
| 固定相同候选，两题联合命中 A | 5/46 | real-random 1/46 | 差 +0.0870，CI [−0.0217, 0.2174]，不能确立正效应 |

固定候选的其他联合命中：native 0/46、no-op 1/46、另一段同 donor 规则历史 5/46、同 recipient 规则历史 2/46。区间使用 10,000 次 donor-cluster paired bootstrap，保留题目和顺序相关性；46 不是 46 个独立 donor。仅一个 checkpoint、沿用开发面板，属于探索性后续诊断，不是独立复现或新的预注册通过。

旧格式结果的事后有效 69 题敏感性：候选可见 49/69 vs 22/69，无候选 1/69 vs 1/69。原始 81 题表不删除、不替换。

## 11. 失败与数值边界

额外描述性复算：directed 有 26/46 个题对×顺序条件对两道题输出同一答案；仅 4/23 donor 至少在一种顺序上联合成功，1/23 在两种顺序都成功。这是事后失败诊断，不是新主要终点。

固定候选中交换前后顺序导致 23/46 个题目输出变化；native/no-op 有 3/92 处文本不一致、1/92 处 A 命中变化，不能搬用旧实验的 no-op=0。该实验没有非法或候选外输出。显式规则实验没有 token-limit 停止，宽松数字格式重解析仍为 11/69，因此不能把低表现只归咎于严格解析。

## 12. 机制结论、需撤回的推断与后续顺序

可以写：历史诱导的学生状态干预能在候选支持条件下定向改变模型响应，并且效应有清晰的输入支持边界。不能写：已证明模型能自主执行 donor 的错误算法，或已完全排除候选偏好/验证式选择。

显式描述也不可靠，说明开放执行的失败不能单独定位为状态缺信息；指令诱发、算术能力及读出接口仍是替代解释。固定候选主区间跨零不证明真实效应为零，但不能据此升级 claim。

另一个必须修正的结论：末层 block OUTPUT 处的过去 token 替换没有通向后续答案 token 的计算路径。其零效应不能证明“表示本质上是跨层轨迹、单向量不可能承载”。CPU 小 Qwen3 测试已重跑：末层替换不改变后续 logits/KV，前一层替换能改变答案 logits。这是架构路径测试，不是训练模型的新增行为证据。

本轮以该边界收尾，不继续换 seed 追正结果。若另行推进更强机制主张，先在独立开发集建立无候选显式规则的可靠执行正对照，再冻结协议检验新 histories/新数值上的状态干预；轨迹与单向量比较必须保证双方均有下游读出路径。上述新增训练/评测本轮未执行。

## 13. 复现与交付记录

- 详细完整报告：`reports/c4_rule_followup_results_20260905.md`。
- 原格式诊断：`reports/c4_format_diagnostic_results_20260905.md`。
- 末层路径审计：`reports/c4_final_layer_interpretation_audit_20260905.md`。
- 本轮独立复算：`results/c4_rule_followup_v2_seed42_20260905.delivery_audit.json`、`results/c4_format_diagnostic_semantic_valid_20260905.delivery_audit.json`，均与既有对应 JSON 精确一致。
- 冻结 15 个文件的 SHA256 全部通过；未修改冻结推理脚本、manifest 或 endpoints。
- 本轮完整相关测试集：18 passed，34.15 秒（CPU 协议/评分/解码与架构路径验证，不是新增 32B 推理结果）。
- Feeds 实验 `c4-rule-followup-seed42-v2-20260905`，3 × Standard 80G1-A100，UAI `rankfun_aml`，镜像 `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`。
- Jobs：`cyan_bell_k8778jy8h3`、`elated_shark_cd34fgg71s`、`dynamic_tomato_5kpsglrkjy`。远端输出 `AMLT_OUTPUT_DIR/rule_followup`；模型、adapter 远端路径见完整实验报告。
- 本轮已实时查询确认集群 **3 jobs — 3 pass**（各时长约 31/44/40 分钟）。这是作业执行成功，不是科学假设通过。
- 集群终态及收尾：以 `results/c4_rule_followup_v2_seed42_20260905.monitor.json` 和 `.finalizer.json` 为准。
- 2026-09-05 18:23:15 UTC 收尾完成：monitor/finalizer 均为 `complete`；最终下载后的完整审计与本轮 delivery audit 再次比较一致。

```bash
cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
/home/v-yuejiali/DeepGraph/.venv/bin/python scripts/audit_c4_rule_followup_extended.py --download results/downloads/c4-rule-followup-seed42-v2-20260905 --output results/c4_rule_followup_v2_seed42_20260905.delivery_audit.json
/home/v-yuejiali/DeepGraph/.venv/bin/python scripts/audit_c4_format_semantic_subset.py --output results/c4_format_diagnostic_semantic_valid_20260905.delivery_audit.json
/home/v-yuejiali/DeepGraph/.venv/bin/python -m pytest -q tests/test_c4_freegen_protocol.py tests/test_c4_freegen_decode.py tests/test_audit_c4_freegen_results.py tests/test_c4_format_diagnostic.py tests/test_c4_rule_followup.py tests/test_c4_rule_extended.py tests/test_c4_final_output_path.py
```
