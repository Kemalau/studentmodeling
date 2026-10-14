# Unified Student Model C1-C4 Server Migration Status

Date: 2026-07-13  
Workspace: `/data1/liyuejia`  
Main experiment root: `experiments/unified_student_c1`

## One-Screen Summary

当前这一轮属于 **smoke / pilot 阶段**，但 C1、C3、C4 已经有可交接的结果。

- **C1**: 统一多任务训练已经跑通。当前 C1 纯性能最强的是 fully-open mixed C1 adapter，平均 acc `0.647`。
- **C2**: 还没有正式做。原因是 T5/T6 需要 FoundationalASSIST 这类长序列真实学生数据；当前本地只有一个很小的 `foundationalassist_attempt`，不能支撑正式 C2。
- **C3**: 有正迁移 smoke 证据。Malrule `identity` 监督能提升真实 Eedi-Mining 误解诊断，response-only 不提升。
- **C4**: 最强结果来自 single-task Malrule `next_response` late-fusion state 模型。STATE probe cross-template acc `0.905`；masked swap clean flip_to_A `0.494`；open-attention swap clean flip_to_A `0.301`。

最重要的当前结论：

> Fully-open 训练最适合 C1 性能；late-fusion/bottleneck 训练最适合 C4 可读可控 state。随机 p=0.5 mixed-mask 没能同时保住二者。

## Directory Size

Current sizes:

| path | size | note |
|---|---:|---|
| `experiments/unified_student_c1/data` | `6.7G` | raw + processed data |
| `experiments/unified_student_c1/results` | `158G` | adapters + eval JSON; most space here |
| `experiments/unified_student_c1/reports` | `148K` | all summary reports |
| `experiments/unified_student_c1/scripts` | `404K` | training/eval/probe/swap scripts |
| `StudentModeling` | `16K` | task definition docs |

The base model is not inside this experiment root:

- `/data1/liyuejia/models/Qwen3-32B`

New server must have this model path or all commands must be updated.

## Data Status

### Raw Data

Raw directories currently present:

| path | size | status |
|---|---:|---|
| `data/raw/eedi_response_hf` | `290M` | Eedi-style real student response data with OCR text reconstruction |
| `data/raw/eedi_mining_2024_hf_ir` | `8.1M` | Eedi-Mining misconception diagnosis data |
| `data/raw/malrulelib` | `40M` | cloned MalruleLib repo |
| `data/raw/foundationalassist_attempt` | `28K` | not usable as full FoundationalASSIST |

Important: **FoundationalASSIST is still effectively missing** for the formal C2/T5/T6 plan.

### Processed Data

Main processed datasets:

| path | size | use |
|---|---:|---|
| `data/processed/mixed_c1_ocr_no_val` | `920M` | main fully-open mixed C1 smoke data |
| `data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck` | `873M` | length-filtered state-token/latefusion data |
| `data/processed/malrule_c1_clean` | `103M` | clean Malrule C1/C4 synthetic data |
| `data/processed/eedi_mining_c1_ir` | `24M` | Eedi-Mining diagnose data |
| `data/processed/eedi_c1_k12_ocr` | `1.7G` | Eedi response data after OCR/text processing |
| `data/processed/transfer_eedi_malrule_ocr_no_val` | `160M` | C3 ablation train files |
| `data/processed/single_task_transfer_ocr_no_val` | `574M` | single-task C3/C1 train files |

Main mixed C1 dataset counts:

| split | total | task counts |
|---|---:|---|
| `mixed_c1_ocr_no_val/train` | `80,537` | `next_response 23604`, `next_correctness 23604`, `future_mastery 20853`, `diagnose_misconception 3476`, `malrule_identity 3000`, `malrule_next_correctness 3000`, `malrule_next_response 3000` |
| `mixed_c1_ocr_no_val/test` | `6,067` | `next_correctness 1479`, `next_response 1479`, `future_mastery 1315`, `diagnose_misconception 894`, plus 3 Malrule tasks x `300` |
| `mixed_c1_ocr_no_val_lte1536_bottleneck/train` | `76,914 kept` | length-filtered for max length 1536 |
| `mixed_c1_ocr_no_val_lte1536_bottleneck/test` | `5,835 kept` | length-filtered for max length 1536 |

## Scripts

Core scripts:

| script | purpose |
|---|---|
| `scripts/train_c1_lora.py` | QLoRA SFT training; supports state token, bottleneck, late-fusion, mixed-mask prob |
| `scripts/evaluate_c1.py` | closed-set candidate-target logprob eval for C1/C3 tasks |
| `scripts/probe_student_state_smoke.py` | linear probe on hidden states at `<STUDENT_STATE>` |
| `scripts/state_swap_misconception_smoke.py` | activation patch / state-swap experiments |
| `scripts/prepare_malrule_c1.py` | build Malrule synthetic data |
| `scripts/prepare_eedi_c1.py` | build Eedi C1 data |
| `scripts/prepare_eedi_mining_c1.py` | build Eedi-Mining diagnosis data |
| `scripts/rebuild_eedi_c1_with_ocr.py` | rebuild Eedi with OCR text |
| `scripts/make_mixed_c1_dataset.py` | mix processed task datasets |
| `scripts/train_eedi_kt_baselines.py` | KT baseline training/eval helper |

Important implemented flags:

- `--add-state-token`
- `--bottleneck-mask`
- `--bottleneck-style standard|late_fusion`
- `--bottleneck-mask-prob <p>`
- `--supervise-eos`

## C1 Current State

C1 = unified multi-task student modeling. Current smoke rows are:

- `diagnose_misconception`
- `future_mastery`
- `malrule_identity`
- `malrule_next_correctness`
- `malrule_next_response`
- `next_correctness`
- `next_response`

### C1 Performance Table

All rows below use the same filtered `lte1536` test style and `n=300` per task.

| model / training mode | avg acc | read |
|---|---:|---|
| base Qwen3-32B | `0.594` | zero/few-shot-ish baseline under our candidate scoring |
| fully-open mixed C1 S120 | `0.647` | current best C1 pure performance |
| latefusion mixed C1 S120, open eval | `0.588` | weaker C1, keeps state pathway idea |
| latefusion mixed C1 S120, masked eval | `0.472` | bottleneck hurts C1 on mixed real tasks |
| mixed-mask p=0.5 S120, open eval | `0.621` | recovers some C1, but loses C4 controllability |

Detailed task accuracy:

| task | base | fully-open | latefusion-open | mixedmask-p50-open |
|---|---:|---:|---:|---:|
| `diagnose_misconception` | `0.460` | `0.500` | `0.487` | `0.440` |
| `future_mastery` | `0.413` | `0.467` | `0.410` | `0.443` |
| `malrule_identity` | `0.527` | `0.593` | `0.573` | `0.563` |
| `malrule_next_correctness` | `0.813` | `0.977` | `0.927` | `0.993` |
| `malrule_next_response` | `0.627` | `0.780` | `0.697` | `0.807` |
| `next_correctness` | `0.673` | `0.563` | `0.490` | `0.500` |
| `next_response` | `0.643` | `0.647` | `0.533` | `0.600` |

Current C1 read:

- Fully-open training is best for task performance.
- Latefusion hurts real-log Eedi tasks but is important for C4.
- p=0.5 mixed-mask helps some Malrule task performance, but not the state-swap story.

Main C1 artifacts:

- Fully-open adapter: `results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter`
- Fully-open eval: `results/mixed_c1_ocr_lte1536_FULLYOPEN_train_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`
- Latefusion mixed adapter: `results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_latefusion_qlora_s120/final_adapter`
- Latefusion open eval: `results/mixed_c1_ocr_lte1536_latefusion_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`
- Mixed-mask p50 adapter: `results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_mixedmask_p50_qlora_s120/final_adapter`
- Mixed-mask p50 eval: `results/mixed_c1_ocr_lte1536_mixedmask_p50_qwen3_32b_qlora_s120_all_tasks_OPEN_n300.json`

Main C1 reports:

- `reports/mixed_c1_full_open_vs_latefusion_open_20260712.md`
- `reports/mixed_c1_mixedmask_p50_20260712.md`
- `reports/mixed_c1_ocr_smoke_20260709.md`
- `reports/c1_pilot_report_20260707.md`

## C2 Current State

C2 = capability frontier: prospective error / persistence tasks that classic KT cannot structurally do.

Status: **not formally done**.

Reason:

- Formal T5/T6 requires long real student histories and future same-KC windows.
- The intended main source was FoundationalASSIST.
- Current local `data/raw/foundationalassist_attempt` is only `28K`, so it is not the actual usable full dataset.

What exists:

- `future_mastery` smoke task exists in mixed C1.
- It is not enough to claim formal C2.

Next C2 action after migration:

1. acquire/restore full FoundationalASSIST or another long-sequence dataset;
2. build T5 `predict_prospective_error`;
3. build T6 `predict_error_persistence`;
4. evaluate against KT baselines for correctness part and mark response/persistence columns as structurally unavailable for classic KT.

## C3 Current State

C3 = transfer / positive migration between tasks.

Current best C3 evidence is a targeted Eedi-Malrule transfer ablation:

Target task: real Eedi-Mining `diagnose_misconception`, `n=300`.

| condition | acc | delta acc | macro-F1 | delta F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Eedi-only | `0.437` | `+0.000` | `0.269` | `+0.000` | `6.551` | `0.418` |
| Eedi+Malrule identity | `0.470` | `+0.033` | `0.314` | `+0.045` | `6.286` | `0.393` |
| Eedi+Malrule response | `0.423` | `-0.013` | `0.271` | `+0.003` | `6.474` | `0.427` |
| Eedi+Malrule identity+response | `0.487` | `+0.050` | `0.314` | `+0.045` | `5.961` | `0.386` |
| All mixed C1 | `0.493` | `+0.057` | `0.320` | `+0.051` | `6.374` | `0.379` |

Read:

- There is a positive C3 signal, but it is specific.
- Malrule `identity` supervision transfers to real Eedi diagnosis.
- Malrule `response` alone does not help.
- This argues against the weak explanation "more synthetic data always helps"; the useful transfer seems to be rule/misconception supervision.

Caveats:

- smoke only: one seed, `n=300`, `120` steps;
- not exposure-matched;
- not official Eedi MAP@25;
- still enough to justify doing C3 properly after migration.

Main C3 artifacts:

- report: `reports/c3_eedi_malrule_ablation_20260712.md`
- Eedi-only adapter: `results/qwen3_32b_single_diagnose_misconception_ocr_no_val_qlora_s120_len1536/final_adapter`
- identity adapter: `results/qwen3_32b_eedi_mining_plus_malrule_identity_ocr_no_val_qlora_s120_len1536/final_adapter`
- response adapter: `results/qwen3_32b_eedi_mining_plus_malrule_response_ocr_no_val_qlora_s120_len1536/final_adapter`
- identity+response adapter: `results/qwen3_32b_eedi_mining_plus_malrule_ir_ocr_no_val_qlora_s120_len1536/final_adapter`

## C4 Current State

C4 = readable and causally usable student state.

Important conceptual point:

- In open attention, swapping state creates conflict: receiver B history says "use B rule", patched state says "use A rule".
- Therefore open-attention swap is not expected to fully flip.
- The meaningful result is whether patched state still has directional effect under conflict.

### Best C4 Model

Best adapter:

- `results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter`

Training recipe:

- Base: `/data1/liyuejia/models/Qwen3-32B`
- Task: `malrule_next_response`
- Data: `data/processed/malrule_c1_clean/train.jsonl`
- Max length: `1536`
- Steps: `800`
- LoRA: r=8, alpha=16, lr=1e-4
- Special token: `<STUDENT_STATE>`
- Mask: `--bottleneck-mask --bottleneck-style late_fusion`

### C4 Task Performance

Same-template Malrule `next_response`, `n=300`:

| test attention | accuracy | macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| masked late-fusion | `0.880` | `0.785` | `0.300` | `0.039` |
| open standard | `0.867` | `0.744` | `0.284` | `0.032` |

Read:

- Opening history at test time does not collapse task performance.
- Open eval is only `-1.3` acc points below masked eval.

### C4 Probe

Latefusion single-task probe:

- label classes: `22`
- layers scanned: `0,8,16,24,32,40,48,56,63`
- STATE best cross-template acc: `0.905` at layer `48`
- random/chance around `~0.045`

Read:

- The state vector is strongly linearly readable for malrule identity across templates.
- This is currently the strongest C4a evidence.

Main artifact:

- `results/c4_response_latefusion_s800_probe_malrule_layers_smoke.json`

### C4 Swap

All-layer state replacement, `n=150` pairs.

Clean subset means `none` predicts receiver B's malrule answer.

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

Read:

- Masked swap has strong causal effect: `0.494` flip_to_A vs `0.292` random and `0.045` samemal.
- Open swap still has directional effect: `0.301` flip_to_A vs `0.140` random and `0.065` samemal.
- Open effect is weaker because direct B history is a competing evidence path.

### Failed / Weaker C4 Modes

| mode | result |
|---|---|
| standard bottleneck S800 | good probe, weak swap; not the best |
| mixed C1 latefusion S120 | C1 weaker; C4 not main |
| mixed-mask p=0.5 | C1 partially recovers, but swap disappears |
| fully-open C1 | best C1, but not designed for state causality |

Main C4 reports:

- `reports/c4_latefusion_open_attention_swap_20260712.md`
- `reports/c4_response_latefusion_s800_20260711.md`
- `reports/c4_response_bottleneck_s800_20260711.md`
- `reports/mixed_c1_mixedmask_p50_20260712.md`

## Training Modes Tried

| purpose | adapter | rough size | current status |
|---|---|---:|---|
| C1 fully-open mixed | `qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter` | parent `1.8G` | best C1 performance |
| C1 latefusion mixed | `qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_latefusion_qlora_s120/final_adapter` | parent `12G` | weaker C1, useful for tradeoff |
| C1 mixed-mask p50 | `qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_mixedmask_p50_qlora_s120/final_adapter` | parent `12G` | partial C1 recovery, no C4 control |
| C4 latefusion single | `qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter` | parent `12G` | best C4 |
| C4 standard bottleneck single | `qwen3_32b_single_malrule_next_response_state_token_bottleneck_qlora_s800_len1536/final_adapter` | parent `12G` | weaker than latefusion |
| C3 Eedi-only | `qwen3_32b_single_diagnose_misconception_ocr_no_val_qlora_s120_len1536/final_adapter` | parent `1.8G` | C3 baseline |
| C3 identity | `qwen3_32b_eedi_mining_plus_malrule_identity_ocr_no_val_qlora_s120_len1536/final_adapter` | parent `268M` | positive transfer |
| C3 response | `qwen3_32b_eedi_mining_plus_malrule_response_ocr_no_val_qlora_s120_len1536/final_adapter` | parent `268M` | no transfer |
| C3 identity+response | `qwen3_32b_eedi_mining_plus_malrule_ir_ocr_no_val_qlora_s120_len1536/final_adapter` | parent `1.8G` | positive transfer |

Directories probably not worth prioritizing unless debugging old runs:

- `results/qwen3_32b_single_malrule_next_response_state_token_sanity` (`30G`)
- early S40/S60 identity/state-token sanity runs
- old 0.6B sanity runs

## Migration Checklist

### Minimal Bundle: Preserve Scientific Record Only

Move these:

- `experiments/unified_student_c1/reports`
- `experiments/unified_student_c1/results/*.json`
- `StudentModeling/TaskC1.md`

This is tiny and preserves all written conclusions and metric JSONs, but cannot rerun models.

### Recommended Bundle: Continue Experiments

Move these:

- `experiments/unified_student_c1/scripts`
- `experiments/unified_student_c1/reports`
- `experiments/unified_student_c1/data`
- `experiments/unified_student_c1/results/*.json`
- the key adapter directories listed below
- `/data1/liyuejia/models/Qwen3-32B`

Key adapter dirs:

```text
experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536
experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_latefusion_qlora_s120
experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_lte1536_state_token_mixedmask_p50_qlora_s120
experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536
experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_bottleneck_qlora_s800_len1536
experiments/unified_student_c1/results/qwen3_32b_single_diagnose_misconception_ocr_no_val_qlora_s120_len1536
experiments/unified_student_c1/results/qwen3_32b_eedi_mining_plus_malrule_identity_ocr_no_val_qlora_s120_len1536
experiments/unified_student_c1/results/qwen3_32b_eedi_mining_plus_malrule_response_ocr_no_val_qlora_s120_len1536
experiments/unified_student_c1/results/qwen3_32b_eedi_mining_plus_malrule_ir_ocr_no_val_qlora_s120_len1536
```

This should be enough to resume C1/C3/C4 evaluation and continue training from the current useful checkpoints.

### Full Bundle

Move the full directory:

```text
experiments/unified_student_c1
```

This is around `165G` and includes all old sanity checkpoints.

## Environment Notes

Known assumptions:

- Python 3.12 was used.
- Key packages used by scripts: `torch`, `transformers`, `peft`, `bitsandbytes`, `sklearn`, `numpy`.
- 32B runs used 4-bit loading for QLoRA.
- Training/eval commands often used:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 experiments/unified_student_c1/scripts/train_c1_lora.py ...
```

and:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 experiments/unified_student_c1/scripts/evaluate_c1.py ...
```

If the new server model path differs, update `--model /data1/liyuejia/models/Qwen3-32B`.

## Reproduction Commands

### Best C1 Eval

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 experiments/unified_student_c1/scripts/evaluate_c1.py \
  --train-jsonl experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val/train.jsonl \
  --eval-jsonl experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val_lte1536_bottleneck/test.jsonl \
  --output-json /tmp/fully_open_c1_eval.json \
  --model /data1/liyuejia/models/Qwen3-32B \
  --adapter experiments/unified_student_c1/results/qwen3_32b_mixed_c1_ocr_no_val_qlora_s120_len1536/final_adapter \
  --max-samples-per-task 300 \
  --max-length 1536 \
  --batch-size 4 \
  --progress-every 300
```

### C3 Identity Transfer Eval

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 experiments/unified_student_c1/scripts/evaluate_c1.py \
  --train-jsonl experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val/train.jsonl \
  --eval-jsonl experiments/unified_student_c1/data/processed/mixed_c1_ocr_no_val/test.jsonl \
  --output-json /tmp/c3_identity_eval.json \
  --model /data1/liyuejia/models/Qwen3-32B \
  --adapter experiments/unified_student_c1/results/qwen3_32b_eedi_mining_plus_malrule_identity_ocr_no_val_qlora_s120_len1536/final_adapter \
  --tasks diagnose_misconception \
  --max-samples-per-task 300 \
  --max-length 1536 \
  --batch-size 4 \
  --progress-every 300
```

### Best C4 Latefusion Eval

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 experiments/unified_student_c1/scripts/evaluate_c1.py \
  --train-jsonl experiments/unified_student_c1/data/processed/malrule_c1_clean/train.jsonl \
  --eval-jsonl experiments/unified_student_c1/data/processed/malrule_c1_clean/test_same_template.jsonl \
  --output-json /tmp/c4_latefusion_masked_eval.json \
  --model /data1/liyuejia/models/Qwen3-32B \
  --adapter experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_latefusion_qlora_s800_len1536/final_adapter \
  --tasks malrule_next_response \
  --max-samples-per-task 300 \
  --max-length 1536 \
  --batch-size 4 \
  --add-state-token \
  --bottleneck-mask \
  --bottleneck-style late_fusion \
  --progress-every 300
```

### Best C4 Open-Attention Eval

Same as above but remove:

```text
--bottleneck-mask --bottleneck-style late_fusion
```

## Recommended Next Steps After Migration

1. **Do not keep tuning the same smoke setup forever.** Current pilot has answered enough to justify a more formal round.
2. For **C1**, use fully-open mixed training as the performance baseline.
3. For **C4**, use latefusion single-task as the main interpretable/causal-state baseline.
4. For **C3**, rerun the Eedi-Malrule ablation with exposure matching, more seeds, full test, and preferably official Eedi MAP@25.
5. For **C2**, acquire full FoundationalASSIST or equivalent long-sequence data before claiming anything.
6. For a **single model that balances C1 and C4**, try staged training:
   - stage 1: open training for C1 competence;
   - stage 2: latefusion continuation on Malrule/state-heavy tasks;
   - optional small open fraction to prevent C1 collapse.

## Bottom Line

Current experimental state is good enough to migrate:

- the data builders work;
- the QLoRA training/eval loop works;
- C1 has a best-performing open model;
- C3 has a specific positive transfer signal;
- C4 has a readable and partially causal `<STUDENT_STATE>` result;
- the remaining missing piece for the original full plan is real long-horizon student data for C2.

