# 统一生成式学生模型: 实验总览与结果账本

日期: 2026-07-13  
实验根目录: `experiments/unified_student_c1`  
核心模型: `/data1/liyuejia/models/Qwen3-32B`

这份文档的目的不是只解释 C1-C4 claim，而是把当前已经做过的实验按“为什么做、怎么做、结果是什么、能支撑什么结论”整理成一个论文写作用总览。

## 0. 一句话总览

我们在做一个 **统一生成式学生模型**:

> 用一个 LLM 和统一 conditional generation 格式，同时完成 correctness prediction、response prediction、misconception diagnosis、mastery estimation 等传统学生建模任务，并检查模型内部是否形成了可读、可干预的 student state。

统一输入大体是:

```text
instruction
student history
<STUDENT_STATE>
query
```

统一输出是结构化标签:

```text
<correctness>incorrect</correctness>
<predicted_response>2/5</predicted_response>
<misconception>add numerators and denominators</misconception>
```

`<STUDENT_STATE>` 是输入中的状态 token。我们关心它在某一层的 hidden state，记作 `z_t`。`z_t` 不是显式输出，也不是持久化学生向量，而是模型每次根据 history 前向计算得到的内部激活。

当前最短结论:

> 统一多任务 C1 已经跑通；Malrule 误解规则监督对真实 Eedi-Mining diagnosis 有正迁移；late-fusion bottleneck 训练能产生一个跨模板可读、并且通过 state swap 可部分控制输出的 student state。  
> 但 real-log KT 仍弱，C2 的未来错误/错误持续性还没有正式做，一个同时兼顾最高 C1 性能和最强 C4 可控性的单一模型还没有定版。

## 1. Claim Map: C1-C4 与 T1-T6

| 层级 | 含义 | 当前状态 |
|---|---|---|
| T1 `predict_correctness` | 预测下一题是否答对 | 已做 Eedi smoke；KT baseline 显示当前 LLM 不赢 |
| T2 `predict_response` | 预测下一题具体答案 | 已做 Eedi/Malrule；Malrule 强，Eedi 接近 item prior |
| T3 `diagnose_misconception` | 诊断错误背后的误解 | 已做 Eedi-Mining/Malrule；是当前真实数据上最强信号 |
| T4 `estimate_mastery` | 估计 KC 掌握档位 | 已做操作化 smoke；接近 baseline |
| T5 `predict_prospective_error` | 预测未来第 delta 题会错成什么 | 未正式做，缺长序列数据 |
| T6 `predict_error_persistence` | 预测错误模式是否持续 | 未正式做，缺长序列数据 |

| Claim | 问题 | 当前读法 |
|---|---|---|
| C1 Generality | 一个统一模型能否覆盖多种学生建模任务 | 已跑通，fully-open mixed C1 是当前性能最强 smoke；KT cell 是明显短板 |
| C2 Capability Frontier | 能否做传统 KT 不自然支持的未来具体错误/持续性预测 | 未正式证明 |
| C3 Transfer | 多任务/合成误解监督是否帮助真实任务 | 有正向 smoke: Malrule identity 提升 Eedi-Mining diagnosis |
| C4 Interpretability | 内部是否有可读、可干预的 student state | late-fusion single-task C4 pilot 很强 |

## 2. 实验版图

当前实验不是一个大 run，而是一条逐步收敛的实验链:

| 实验组 | 主要问题 | 结果一句话 |
|---|---|---|
| E0 数据与预处理 | 数据能不能构造成统一生成式任务 | Eedi、Eedi-Mining、Malrule 已可用；FoundationalASSIST 不可用 |
| E1 初始 Eedi C1 pilot | 统一格式能不能跑 KT/response/mastery | 32B 比 0.6B 强，但 KT 不足以 paper-grade |
| E2 KT baseline | real-log correctness 是否能赢传统 baseline | 当前 Qwen3-32B 候选打分输给 question_rate/DKT |
| E3 Malrule clean C1 | 可执行误解规则上是否有强学习信号 | 强；response/identity 都大幅超过 baseline，跨模板也强 |
| E4 OCR mixed C1 | 把 Eedi OCR、Eedi-Mining、Malrule 混在一起是否可行 | 可行；diagnosis/Malrule 强，KT 仍弱 |
| E5 单任务 vs 混合/C3 | 多任务是否帮助真实 misconception diagnosis | Malrule identity 对 Eedi-Mining 有正迁移 |
| E6 C4 marker/strict probe | state 是否可线性读出 malrule | 可读；非严格 marker 0.841，strict token 0.788 |
| E7 C4 bottleneck swap | 可读 state 是否能因果控制输出 | standard bottleneck 可读但 swap 弱/负 |
| E8 C4 late-fusion swap | 改 mask 后 state 是否可控 | 强正；masked flip_to_A 0.494，open flip_to_A 0.301 |
| E9 mixed-mask C1/C4 | 能否同时保 C1 和 C4 | C1 部分恢复，但 C4 controllability 消失 |

## 3. 数据与任务构造

| 数据 | 用途 | 状态 |
|---|---|---|
| Eedi response / NeurIPS logs | T1/T2/T4，真实学生日志 | 已处理；history up to 12 |
| Eedi OCR text | 给 Eedi 题目补文本 | 已处理；948 题中 782 OCR ok，166 weak |
| Eedi-Mining | T3 真实 misconception diagnosis | 已处理成 closed-set candidate scoring |
| MalruleLib synthetic | 可执行误解规则、C1/C3/C4 | 已清洗；支持 same/cross-template |
| FoundationalASSIST | C2 T5/T6 长序列 | 没有正式可用；HF gated/download failed |

主要 processed 数据:

| path | 用途 |
|---|---|
| `data/processed/eedi_c1_k12` | 初始 Eedi KT/response/mastery |
| `data/processed/malrule_c1_clean` | Malrule C1/C4 clean 数据 |
| `data/processed/mixed_c1_no_val` | 早期 mixed C1 |
| `data/processed/mixed_c1_ocr_no_val` | OCR 后 mixed C1 |
| `data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck` | state-token / late-fusion mixed C1 |
| `data/processed/transfer_eedi_malrule_ocr_no_val` | C3 Eedi+Malrule ablation |

## 4. E0: 数据构造实验

### Eedi C1 数据

来源: `experiments/learnerlm_exp1/data/processed/eedi_neurips_mcq_canonical.csv`

构造:

- student-held-out split;
- history length up to 12;
- tasks: `next_correctness`, `next_response`, `future_mastery`;
- total examples: 76,581.

| split | next_correctness | next_response | future_mastery |
|---|---:|---:|---:|
| train | 23,604 | 23,604 | 20,853 |
| valid | 1,473 | 1,473 | 1,301 |
| test | 1,479 | 1,479 | 1,315 |

### Malrule clean 数据

目录: `data/processed/malrule_c1_clean`

每个 raw example 生成三个任务:

- `malrule_next_response`
- `malrule_identity`
- `malrule_next_correctness`

| split | raw examples | task rows |
|---|---:|---:|
| train | 3,000 | 9,000 |
| val | 300 | 900 |
| test_same_template | 300 | 900 |
| test_cross_template | 300 | 900 |

Malrule pool:

- families: `subtraction`, `decimals`, `fractions`;
- discovered / pooled / cross-capable malrules: 21;
- random malrule-label chance: about 1/22 = 0.0455.

关键报告:

- `reports/c1_pilot_report_20260707.md`
- `reports/malrule_c1_data_report_20260707.md`
- `reports/eedi_response_ocr_report_20260709.md`

## 5. E1: 初始 Eedi C1 Pilot

问题:

> 统一生成式 schema 能不能在 Eedi real-log 上跑通 T1/T2/T4？

模型:

- Qwen3-0.6B base + LoRA S200;
- Qwen3-32B base + QLoRA S100.

评测: candidate likelihood，512 examples per task。

### Qwen3-0.6B

| task | base acc | unified LoRA acc | strongest baseline |
|---|---:|---:|---:|
| future_mastery | 0.4766 | 0.4629 | 0.4746 skill-majority |
| next_correctness | 0.5625 | 0.5527 | 0.6738 global-majority |
| next_response | 0.6348 | 0.3750 | 0.6309 question-majority |

读法: 0.6B 学会格式，但不构成能力结果。

### Qwen3-32B

| task | strongest baseline | base acc | unified QLoRA acc |
|---|---:|---:|---:|
| future_mastery | 0.4746 | 0.4316 | 0.4648 |
| next_correctness | 0.6738 | 0.6680 | 0.6719 |
| next_response | 0.6309 | 0.6719 | 0.6719 |

读法:

- 32B base 在 `next_response` 上已经强于 question-majority；
- QLoRA 改善 NLL/calibration，但 top-1 accuracy 提升有限；
- 这是 C1 pipeline pilot，不是最终 C1 证明。

关键报告: `reports/c1_pilot_report_20260707.md`

## 6. E2: Eedi KT Baseline Comparison

问题:

> 在 real-log `next_correctness` 上，当前 LLM 方案能不能赢传统/统计 KT baseline？

结论: 不能。当前 LLM 候选 target likelihood 在 KT cell 上是弱项。

| model | accuracy | macro-F1 | AUC(correct) | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| global_rate | 0.6734 | 0.4024 | 0.5000 | 0.6881 | 0.1661 |
| question_rate | 0.6295 | 0.6102 | 0.6724 | 0.6421 | 0.0274 |
| dkt_skill_lstm | 0.6011 | 0.5795 | 0.6284 | 0.6571 | 0.0092 |
| dkt_item_lstm_mlp | 0.6288 | 0.6029 | 0.6474 | 0.6474 | 0.0181 |
| qwen3_32b_unified_qlora_s100 | 0.6660 | 0.4254 | 0.5661 | 0.7464 | 0.1446 |
| qwen3_32b_kt_only_s150 | 0.6024 | 0.5213 | 0.5642 | 0.6455 | 0.0394 |
| qwen3_32b_kt_only_s300 | 0.6018 | 0.5107 | 0.5638 | 0.6435 | 0.0295 |

读法:

- Qwen3-32B unified accuracy 接近 global majority，是因为多数预测 `correct`；
- AUC、macro-F1、NLL 都落后于 question_rate 和 item-aware DKT；
- KT-only S150/S300 改善 NLL/ECE，但 AUC 基本不动；
- 这是真实 C1 短板，不能在论文里装成 win。

关键报告: `reports/eedi_kt_comparison_20260707.md`

## 7. E3: Malrule Clean C1

问题:

> 在可执行数学误解规则数据上，统一生成式模型能不能学到 response prediction 和 misconception identity？

模型:

- base: Qwen3-32B;
- adapter: `results/qwen3_32b_malrule_unified_qlora_s250/final_adapter`;
- training: all 9,000 Malrule task rows, 250 steps.

### Same-template

| task | best baseline acc | base acc | QLoRA acc | QLoRA macro-F1 |
|---|---:|---:|---:|---:|
| malrule_identity | 0.4900 | 0.5600 | 0.7300 | 0.6235 |
| malrule_next_response | 0.4267 | 0.6567 | 0.8600 | 0.7488 |
| malrule_next_correctness | 1.0000 | 0.8267 | 1.0000 | 1.0000 |

### Cross-template

| task | best baseline acc | base acc | QLoRA acc | QLoRA macro-F1 |
|---|---:|---:|---:|---:|
| malrule_identity | 0.1867 | 0.5367 | 0.7333 | 0.6559 |
| malrule_next_response | 0.4833 | 0.6233 | 0.8600 | 0.7440 |
| malrule_next_correctness | 1.0000 | 0.8100 | 1.0000 | 1.0000 |

读法:

- 这是第一个清楚的 C1 能力信号；
- Malrule response 和 identity 都显著强于 simple baselines；
- cross-template 结果说明不是只背模板；
- `malrule_next_correctness` 由构造决定，不能作为 headline。

关键报告: `reports/malrule_c1_data_report_20260707.md`

## 8. E4: Mixed C1 与 OCR C1

问题:

> 把 Eedi OCR、Eedi-Mining、Malrule 任务混到一个统一模型里，是否仍然可学？

主模型:

```text
results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter
```

设置:

- Qwen3-32B 4-bit QLoRA;
- max length 1536;
- 120 steps;
- response-token-only loss;
- candidate likelihood evaluation;
- mostly `n=300` smoke.

### OCR mixed C1 headline

| data/task | n | model acc | macro-F1 | best simple baseline | read |
|---|---:|---:|---:|---|---|
| Eedi OCR `future_mastery` | 300 | 0.463 | 0.380 | question majority 0.470 | roughly baseline |
| Eedi OCR `next_correctness` | 300 | 0.573 | 0.525 | global majority 0.703 | not competitive |
| Eedi OCR `next_response` | 300 | 0.653 | 0.636 | question majority 0.650 | tiny win |
| Eedi-Mining `diagnose_misconception` | 300 | 0.493 | 0.320 | simple majority 0.083 | strong closed-set signal |
| Malrule `malrule_identity` | 300 | 0.587 | 0.453 | majority/history 0.187 | strong |
| Malrule `malrule_next_correctness` | 300 | 0.980 | 0.968 | history-state 1.000 | saturated |
| Malrule `malrule_next_response` | 300 | 0.793 | 0.657 | history-state 0.483 | strong |

OCR vs previous thin-text:

| task | thin-text acc | OCR-text acc | change |
|---|---:|---:|---:|
| Eedi `future_mastery` | 0.473 | 0.463 | -0.010 |
| Eedi `next_correctness` | 0.463 | 0.573 | +0.110 |
| Eedi `next_response` | 0.620 | 0.653 | +0.033 |
| Malrule `malrule_identity` | 0.447 | 0.587 | +0.140 |
| Malrule `malrule_next_response` | 0.793 | 0.793 | +0.000 |

读法:

- 统一格式能同时承载 real-log response、real misconception diagnosis、synthetic malrule；
- 当前 C1 最强不是 KT，而是 Malrule 和 Eedi-Mining diagnosis；
- OCR 帮 Eedi response/correctness，但没有解决 KT。

关键报告:

- `reports/mixed_c1_no_val_smoke_20260708.md`
- `reports/mixed_c1_ocr_smoke_20260709.md`
- `reports/eedi_response_ocr_report_20260709.md`

## 9. E4b: Fully-Open、Late-Fusion、Mixed-Mask C1 对齐

问题:

> C1 纯性能与 C4 state bottleneck 是否能兼得？

同一批 mixed C1 tasks 的主要模式:

| mode | avg acc | read |
|---|---:|---|
| Qwen3-32B base | 0.594 | base reference |
| fully-open mixed C1 S120 | 0.647 | 当前 C1 纯性能最好 |
| latefusion mixed C1 S120, open eval | 0.588 | C1 降，但保留 state 设计 |
| latefusion mixed C1 S120, masked eval | 0.472 | bottleneck 对 mixed C1 伤害大 |
| mixed-mask p=0.5 S120, open eval | 0.621 | C1 部分恢复，但 C4 control 没保住 |

任务准确率:

| task | base | fully-open | latefusion-open | mixedmask-p50-open |
|---|---:|---:|---:|---:|
| `diagnose_misconception` | 0.460 | 0.500 | 0.487 | 0.440 |
| `future_mastery` | 0.413 | 0.467 | 0.410 | 0.443 |
| `malrule_identity` | 0.527 | 0.593 | 0.573 | 0.563 |
| `malrule_next_correctness` | 0.813 | 0.977 | 0.927 | 0.993 |
| `malrule_next_response` | 0.627 | 0.780 | 0.697 | 0.807 |
| `next_correctness` | 0.673 | 0.563 | 0.490 | 0.500 |
| `next_response` | 0.643 | 0.647 | 0.533 | 0.600 |

读法:

- C1 纯性能目前选 fully-open；
- late-fusion/state bottleneck 不适合直接作为 mixed C1 最强模型；
- mixed-mask p=0.5 是折中尝试，但 C4 swap 失败，所以不是最终解。

关键报告:

- `reports/mixed_c1_full_open_vs_latefusion_open_20260712.md`
- `reports/mixed_c1_latefusion_s120_20260712.md`
- `reports/mixed_c1_latefusion_open_eval_20260712.md`
- `reports/mixed_c1_mixedmask_p50_20260712.md`

## 10. E5: Single vs Mixed Transfer

问题:

> 混合训练是否比单任务训练更好？

设置:

- base: Qwen3-32B;
- all runs: QLoRA, 120 steps, max length 1536, seed 42;
- eval: n=300 candidate scoring;
- comparison: mixed OCR model vs task-specific single-task adapters.

| task | single acc | mixed acc | delta | single F1 | mixed F1 | read |
|---|---:|---:|---:|---:|---:|---|
| Eedi `next_correctness` | 0.577 | 0.573 | -0.003 | 0.553 | 0.525 | no positive transfer |
| Eedi `next_response` | 0.650 | 0.653 | +0.003 | 0.634 | 0.636 | tiny positive |
| Eedi-Mining `diagnose_misconception` | 0.437 | 0.493 | +0.057 | 0.269 | 0.320 | positive transfer candidate |
| Malrule `malrule_next_response` | 0.870 | 0.793 | -0.077 | 0.760 | 0.657 | task dilution |

读法:

- broad “multi-task helps everything” 不成立；
- 最清楚的正迁移是 Eedi-Mining diagnosis；
- Malrule response 单任务更强，说明 exact synthetic response 需要足够 task exposure。

关键报告: `reports/single_vs_mixed_transfer_smoke_20260709.md`

## 11. E6: C3 Eedi-Malrule Targeted Ablation

问题:

> Malrule 的哪种监督在帮真实 Eedi-Mining misconception diagnosis？

目标任务固定: Eedi-Mining `diagnose_misconception`, n=300。

| condition | acc | delta acc | macro-F1 | delta F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Eedi-only | 0.437 | +0.000 | 0.269 | +0.000 | 6.551 | 0.418 |
| Eedi + Malrule identity | 0.470 | +0.033 | 0.314 | +0.045 | 6.286 | 0.393 |
| Eedi + Malrule response | 0.423 | -0.013 | 0.271 | +0.003 | 6.474 | 0.427 |
| Eedi + identity + response | 0.487 | +0.050 | 0.314 | +0.045 | 5.961 | 0.386 |
| all mixed C1 | 0.493 | +0.057 | 0.320 | +0.051 | 6.374 | 0.379 |

读法:

- C3 有正向 smoke；
- transfer 主要来自 Malrule `identity`，不是“多喂合成数据”；
- response-only 不提升，说明有效共享结构是 misconception/rule identity；
- 仍需 exposure-matched、多 seed、full test、官方 MAP@25 风格评测。

关键报告: `reports/c3_eedi_malrule_ablation_20260712.md`

## 12. E7: C4 早期 Probe 与 Strict Token

问题:

> `<STUDENT_STATE>` 位置是否线性编码了学生 malrule identity？

### Marker-only probe

模型: response-only single-task adapter，不是直接训练 identity。

结果:

- best STATE cross-template acc: 0.841;
- best history-last cross-template acc: 0.779;
- chance: 0.0455.

读法: state 位置可读，但当时 `<STUDENT_STATE>` 还不是单 token。

### Strict state token probe

修复: 把 `<STUDENT_STATE>` 作为 single special token。

结果:

- best STATE cross-template acc: 0.788;
- best history-last cross-template acc: 0.812;
- chance: 0.0455.

读法:

- 可读性仍明显存在；
- history-last control 也强，说明 representation 是 history-derived distributed signal，不应声称 state token 唯一承载全部信息。

关键报告:

- `reports/student_state_probe_smoke_20260710.md`
- `reports/student_state_strict_token_probe_20260710.md`

## 13. E8: C4 Misconception/Standard Bottleneck Swap 负结果

这一组实验回答:

> probe 能读出 state，不代表 state swap 一定能控制输出。我们需要因果测试。

### Misconception identity swap smoke

模型在 diagnosis 上有 sanity:

- model accuracy: 0.700;
- macro-F1: 0.617;
- question-majority baseline: 0.458;
- random: 0.045.

但 state swap 没有干净地把 recipient 误解翻成 donor 误解。

### Standard bottleneck S200/S800

S200:

- response acc only 0.5917；
- probe 可读，但 swap 不稳定/不方向性。

S800:

| model | n | accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| S200 bottleneck | 120 | 0.5917 | 0.4304 | 0.9577 | 0.1216 |
| S800 bottleneck | 300 | 0.7200 | 0.5687 | 0.6464 | 0.0695 |

S800 probe:

| position | best layer | same acc | cross acc | cross macro-F1 |
|---|---:|---:|---:|---:|
| STATE | 48/56/63 region | up to 0.9038 | up to 0.8696 | up to 0.8628 |
| history-last control | 40 | 0.8243 | 0.8379 | 0.8193 |

S800 clean swap subset where `none == stay_B`, n=54:

| condition | flip_to_A | stay_B | changed top-1 |
|---|---:|---:|---:|
| noop | 3/54 | 49/54 | 5/54 |
| samemal | 2/54 | 51/54 | 3/54 |
| random | 4/54 | 47/54 | 7/54 |
| swap | 2/54 | 51/54 | 3/54 |

读法:

- standard bottleneck 证明了 “readable but not controllable”；
- 行为 gate 过了以后 swap 仍不方向性，所以问题不是简单欠训练；
- 这促成了 late-fusion mask 设计。

关键报告:

- `reports/state_swap_misconception_smoke_20260710.md`
- `reports/bottleneck_mask_swap_smoke_20260710.md`
- `reports/bottleneck_s200_filtered_swap_20260710.md`
- `reports/bottleneck_layer_scan_and_token_budget_20260710.md`
- `reports/c4_response_bottleneck_s800_20260711.md`
- `reports/c4_response_bottleneck_controls_n150_20260711.md`

## 14. E9: C4 Late-Fusion 正结果

问题:

> 如果让 answer tokens 必须通过 `<STUDENT_STATE>` 获取 history-derived student information，state swap 是否会变成有效控制？

模型:

```text
results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter
```

Late-fusion mask:

- `<STUDENT_STATE>` 可以 attend instruction 和 history；
- question tokens after `<STUDENT_STATE>` 不能 attend history；
- question tokens 也不能 attend `<STUDENT_STATE>`；
- answer/target tokens 可以 attend `<STUDENT_STATE>`。

### Prediction gate

| model variant | response accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| standard bottleneck S800 | 0.7200 | 0.5687 | 0.6464 | 0.0695 |
| late-fusion bottleneck S800 | 0.8800 | 0.7849 | 0.3002 | 0.0385 |

Open-attention eval on same model:

| test attention | accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| masked latefusion | 0.880 | 0.785 | 0.300 | 0.039 |
| open standard | 0.867 | 0.744 | 0.284 | 0.032 |

读法: test-time 打开 history 后，任务性能只掉 1.3 acc points。

### C4a probe

| position | best layer | same acc | cross acc | cross macro-F1 |
|---|---:|---:|---:|---:|
| `<STUDENT_STATE>` | 48 | 0.8828 | 0.9051 | 0.8946 |
| history-last | 48 | 0.8410 | 0.8617 | 0.8482 |

读法:

- `z_t` 强烈线性编码 malrule identity；
- cross-template acc 0.905，远高于 chance 0.0455；
- history-last 也强，所以要写成 “state position is a strong readable site”，不要写成唯一信息载体。

### C4b masked state swap

Clean subset = no-intervention 本来预测 recipient answer，n=89。

| condition | flip_to_A | stay_B | other | changed top-1 | delta p_A | delta p_B |
|---|---:|---:|---:|---:|---:|---:|
| noop | 0.000 | 1.000 | 0.000 | 0.000 | -0.0003 | -0.0005 |
| samemal | 0.045 | 0.865 | 0.090 | 0.135 | +0.0175 | -0.0304 |
| random | 0.292 | 0.506 | 0.202 | 0.494 | +0.1413 | -0.2912 |
| swap | 0.494 | 0.225 | 0.281 | 0.775 | +0.2881 | -0.4279 |

读法:

- swap 是方向性因果效果；
- noop 完全稳定，hook 不是原因；
- samemal 几乎不翻，说明不是任意 student vector；
- random 有扰动，但明显弱于 directed swap。

### C4b open-attention state swap

Clean subset n=93。

| condition | flip_to_A | stay_B | other | changed top-1 |
|---|---:|---:|---:|---:|
| noop | 0.011 | 0.989 | 0.000 | 0.011 |
| samemal | 0.065 | 0.871 | 0.065 | 0.129 |
| random | 0.140 | 0.699 | 0.161 | 0.301 |
| swap | 0.301 | 0.473 | 0.226 | 0.527 |

读法:

- open attention 下效果变弱但不消失；
- 变弱合理，因为 recipient history 和 donor state 是冲突证据；
- 这是当前最强 C4 结果。

关键报告:

- `reports/c4_response_latefusion_s800_20260711.md`
- `reports/c4_latefusion_open_attention_swap_20260712.md`

## 15. E10: Mixed-Mask p=0.5 负结果

问题:

> 能不能用 mixed-mask 训练让同一个 mixed C1 模型既有不错 C1，又保留 C4 controllability？

结果:

- open C1 avg acc: 0.621，比 latefusion-open 0.588 好，但低于 fully-open 0.647；
- C4 masked swap clean subset 只有 `1/40 = 0.025` flip_to_A；
- controllability 基本消失。

读法:

> mixed-mask 是一个失败但有用的桥接实验：它说明恢复 C1 性能会让 history bypass 重新变强，state 不再是清晰控制旋钮。

关键报告: `reports/mixed_c1_mixedmask_p50_20260712.md`

## 16. 当前可写进论文的正结果

可以相对自信写:

1. **统一生成式格式可行**  
   同一 prompt/target schema 能覆盖 Eedi response/mastery/correctness、Eedi-Mining diagnosis、Malrule identity/response/correctness。

2. **Malrule 可执行误解任务强**  
   Qwen3-32B QLoRA 在 Malrule response/identity 上显著超过 simple baselines，且 cross-template 仍强。

3. **真实 misconception diagnosis 有 transfer 信号**  
   Eedi-Mining diagnosis 中，加入 Malrule identity 从 0.437 提升到 0.470；identity+response 到 0.487；all mixed 到 0.493。

4. **student state 可读**  
   Late-fusion C4 模型中，`<STUDENT_STATE>` hidden state 对 malrule identity 的 cross-template linear probe acc 达 0.905。

5. **student state 可因果干预**  
   Late-fusion masked swap clean flip_to_A 为 0.494；open attention 下仍有 0.301，明显高于 samemal/noop。

## 17. 当前不能写过头的地方

不要声称:

- 已经完成 paper-grade C1；
- real-log KT 已经赢过传统 KT baseline；
- C2 已经证明；
- 一个单一模型同时达到最强 C1 和最强 C4；
- `<STUDENT_STATE>` 是唯一 student information carrier；
- 所有多任务训练都正迁移。

更准确写法:

> The current evidence supports a unified generative formulation and strong synthetic misconception/response modeling, with targeted positive transfer to real misconception diagnosis. It also shows that a late-fusion bottleneck can induce a readable and partially controllable student-state representation. However, real-log KT remains weaker than item-aware baselines, and future-error/persistence tasks require longer-sequence data before C2 can be claimed.

## 18. 当前主结果表

| claim/experiment | strongest positive number | main caveat |
|---|---:|---|
| Eedi C1 initial 32B response | next_response acc 0.6719 vs baseline 0.6309 | correctness/mastery not strong |
| Eedi KT baseline | Qwen AUC 0.5661 | loses to question_rate 0.6724 and DKT 0.6474 |
| Malrule C1 same-template | response acc 0.8600, identity acc 0.7300 | synthetic |
| Malrule C1 cross-template | response acc 0.8600, identity acc 0.7333 | synthetic |
| OCR mixed C1 | Eedi-Mining diagnosis acc 0.493; Malrule response acc 0.793 | n=300 smoke |
| Fully-open mixed C1 | avg acc 0.647 | not C4-controllable |
| C3 targeted transfer | all mixed +0.057 acc over Eedi-only | one seed, not exposure matched |
| C4 latefusion probe | cross-template acc 0.905 | history-last control also strong |
| C4 latefusion masked swap | clean flip_to_A 0.494 | single-task Malrule, closed-set scoring |
| C4 latefusion open swap | clean flip_to_A 0.301 | weaker due to conflicting history |
| Mixed-mask p=0.5 | C1 avg acc 0.621 | C4 swap collapses |

## 19. 论文行文建议

建议把论文实验分成三层:

1. **Task generality layer**  
   展示统一生成式格式覆盖 T1-T4，重点用 Malrule/Eedi-Mining positive cells，诚实承认 KT weakness。

2. **Transfer layer**  
   用 Eedi-Mining + Malrule identity ablation 支撑 “misconception rule supervision transfers to real diagnosis”。

3. **State layer**  
   用 late-fusion single-task 模型证明 student state 的可读性和因果可干预性。把 fully-open/mixed-mask 作为负结果，说明可控 state 需要结构性约束。

## 20. 下一步最值得补的实验

1. C1 正式化:
   - full test；
   - 多 seed；
   - 更强 KT baselines；
   - 找更好的 real-log content 或 FoundationalASSIST。

2. C3 正式化:
   - exposure-matched；
   - 多 seed；
   - full Eedi-Mining；
   - 尽量接官方 MAP@25 retrieval protocol。

3. C4 定版:
   - late-fusion 多 seed；
   - shuffled-label probe control；
   - layer-band/single-layer swap；
   - more swap pairs；
   - cross-template swap if legal pairs sufficient。

4. C2 数据:
   - 找 full FoundationalASSIST 或等价长序列；
   - 构造 T5 prospective error；
   - 构造 T6 persistence；
   - 与 KT baseline 对齐 correctness 部分，同时强调 response/persistence 是生成式模型的新能力接口。

## 21. 主要报告索引

| report | 内容 |
|---|---|
| `reports/c1_pilot_report_20260707.md` | 初始 Eedi C1 pilot |
| `reports/eedi_kt_comparison_20260707.md` | KT baselines vs Qwen |
| `reports/malrule_c1_data_report_20260707.md` | Malrule clean 数据与结果 |
| `reports/eedi_response_ocr_report_20260709.md` | Eedi OCR 数据 |
| `reports/mixed_c1_ocr_smoke_20260709.md` | OCR mixed C1 |
| `reports/single_vs_mixed_transfer_smoke_20260709.md` | single vs mixed transfer |
| `reports/c3_eedi_malrule_ablation_20260712.md` | C3 targeted ablation |
| `reports/student_state_probe_smoke_20260710.md` | marker state probe |
| `reports/student_state_strict_token_probe_20260710.md` | strict state token probe |
| `reports/c4_response_bottleneck_s800_20260711.md` | standard bottleneck positive probe / negative swap |
| `reports/c4_response_latefusion_s800_20260711.md` | late-fusion positive probe/swap |
| `reports/c4_latefusion_open_attention_swap_20260712.md` | open-attention swap |
| `reports/mixed_c1_full_open_vs_latefusion_open_20260712.md` | C1 fully-open vs late-fusion |
| `reports/mixed_c1_mixedmask_p50_20260712.md` | mixed-mask p=0.5 |

