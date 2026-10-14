# C1 Pilot Report: Unified Generative Student Model

Date: 2026-07-07

## What Ran

This pilot tested the C1 behavior claim only: whether a single conditional-generation model can share one prompt/output format across multiple student-modeling tasks.

Implemented modules:

- `scripts/prepare_eedi_c1.py`: converts Eedi interaction logs into unified C1 JSONL.
- `scripts/evaluate_c1.py`: evaluates trivial/statistical baselines and optional CausalLM candidate likelihood.
- `scripts/train_c1_lora.py`: response-token-only LoRA SFT for unified C1 samples.

## Data Audit

Downloaded or reused:

- Eedi NeurIPS task 3/4 data already present locally, with 948 questions, 4,918 students, and 1,382,727 answers.
- Public Hugging Face `rik1599/eedi-response-data` metadata downloaded into `data/raw/eedi_response_hf`.
- `sonkar-lab/malrulelib` cloned into `data/raw/malrulelib`, commit `8ddedfb`.

Unavailable this run:

- FoundationalASSIST: Hugging Face reports manual gated access; direct download failed with `GatedRepoError 401`.
- Kaggle/MAP misconception data: no `~/.kaggle/kaggle.json`, so the API path is not available.

## Prepared Dataset

Source:

`/data1/liyuejia/experiments/learnerlm_exp1/data/processed/eedi_neurips_mcq_canonical.csv`

Prepared output:

`/data1/liyuejia/experiments/unified_student_c1/data/processed/eedi_c1_k12/`

Split is by student. History length is up to 12 prior interactions.

| Split | next_correctness | next_response | future_mastery |
|---|---:|---:|---:|
| train | 23,604 | 23,604 | 20,853 |
| valid | 1,473 | 1,473 | 1,301 |
| test | 1,479 | 1,479 | 1,315 |

Total examples: 76,581.

Tasks:

- `next_correctness`: output `<correctness>correct</correctness>` or `<correctness>incorrect</correctness>`.
- `next_response`: output `<predicted_answer>A/B/C/D</predicted_answer>`.
- `future_mastery`: output `<mastery>low/medium/high</mastery>`, derived from near-future same-skill correctness.

## Test Baselines

Full test set results:

| Task | Global Majority | Skill Majority | Question Majority | History State |
|---|---:|---:|---:|---:|
| future_mastery acc | 0.4395 | 0.4525 | 0.4570 | 0.4365 |
| next_correctness acc | 0.6734 | 0.5828 | 0.6288 | 0.5835 |
| next_response acc | 0.2921 | 0.3529 | 0.6376 | 0.2623 |

Interpretation:

- `next_correctness` has a strong class imbalance on test; global majority is already 0.6734.
- `next_response` has very strong question/item prior; question-majority is 0.6376.
- This dataset is useful for pipeline sanity, but not enough to prove LLM semantic superiority because question text/options are not fully available in the current adapter.

## Model Runs

### 0.6B Smoke

Base model:

`/data1/liyuejia/models/Qwen3-0.6B`

Unified LoRA:

`/data1/liyuejia/experiments/unified_student_c1/results/qwen3_0p6b_unified_lora_k12_s200/final_adapter`

Training:

- 30k train cap
- 1.5k valid cap
- 200 steps
- LoRA rank 8
- response-token-only loss
- final eval loss: 0.1106

Candidate-likelihood test on 512 examples per task:

| Task | Qwen3-0.6B Base Acc | Unified LoRA Acc | Strong Baseline Acc |
|---|---:|---:|---:|
| future_mastery | 0.4766 | 0.4629 | 0.4746 skill-majority |
| next_correctness | 0.5625 | 0.5527 | 0.6738 global-majority |
| next_response | 0.6348 | 0.3750 | 0.6309 question-majority |

Main read:

- The unified SFT run learned the output format but did not become competitive on this 0.6B smoke.
- Base model matched question-majority on `next_response`, likely because the prompt exposes the known correct answer and the dataset has strong item/answer priors.
- Correctness remains the failure point: both base and 200-step unified LoRA are below simple global majority.

### 32B Run

User-requested model:

`/data1/liyuejia/models/Qwen3-32B`

Unified QLoRA:

`/data1/liyuejia/experiments/unified_student_c1/results/qwen3_32b_unified_qlora_k12_s100/final_adapter`

Training:

- 20k train cap
- 300 valid cap
- 100 steps
- QLoRA 4-bit load for training only
- LoRA rank 8, alpha 16
- response-token-only loss
- final eval loss: 0.1129

Candidate-likelihood test on 512 examples per task:

| Task | Strong Baseline Acc | Qwen3-32B Base Acc | Qwen3-32B Unified QLoRA Acc |
|---|---:|---:|---:|
| future_mastery | 0.4746 skill-majority | 0.4316 | 0.4648 |
| next_correctness | 0.6738 global-majority | 0.6680 | 0.6719 |
| next_response | 0.6309 question-majority | 0.6719 | 0.6719 |

Secondary metrics:

| Task | 32B Base Macro-F1 | 32B QLoRA Macro-F1 | 32B Base NLL | 32B QLoRA NLL |
|---|---:|---:|---:|---:|
| future_mastery | 0.4290 | 0.4514 | 1.7030 | 1.0554 |
| next_correctness | 0.4115 | 0.4237 | 0.9914 | 0.7397 |
| next_response | 0.6651 | 0.6661 | 1.2128 | 1.1786 |

Main read:

- Scaling to 32B matters: the base 32B model already beats the question-majority baseline on `next_response`.
- The 100-step unified QLoRA improves calibration/NLL and slightly improves `future_mastery` and `next_correctness`, but does not materially move top-1 accuracy over base on `next_response`.
- C1 is partly supported on this Eedi pilot: one model is competitive on answer prediction and close to trivial correctness/mastery baselines. It is not yet a clean "all tasks competitive" result.

## C1 Status

This run completes a C1 pilot loop, not the full C1 claim.

What is done:

- Data download/audit policy executed.
- Gated datasets were abandoned rather than blocking.
- Unified generation schema implemented.
- Deterministic parser-free candidate scoring implemented.
- Baselines implemented and run.
- Base LLM and unified LoRA/QLoRA run on three tasks, including a user-requested 32B run.

What is not yet proven:

- No true misconception-label task was evaluated because Kaggle/MAP access was unavailable and MalruleLib was only cloned, not adapted.
- No strong KT baselines such as AKT/SAKT/DKT were run yet.
- No longer 32B unified training or single-task-vs-joint transfer control was run.
- No semantically rich FoundationalASSIST result because the dataset is gated.

## Decision

C1 is not fully established by this pilot. The correct conclusion is:

> The C1 infrastructure is working. Qwen3-32B shows real strength on answer prediction and near-baseline behavior on correctness/mastery, but the current 100-step unified QLoRA does not yet prove a unified model is competitive across all tasks.

## Next Run

Recommended next concrete run:

1. Run longer 32B unified QLoRA, plus 32B single-task SFT controls, to test whether joint training gives positive transfer.
2. Add MalruleLib synthetic misconception/response as the fourth C1 task.
3. Add pyKT baselines for Eedi correctness.
4. Add task-level progress logging and a faster batched scorer for full-test 32B evaluation.
5. If FoundationalASSIST access becomes available, switch it to the primary C1 real-data source.

## Sources Checked

- Eedi NeurIPS Education Challenge: official page describes tasks 1/2 as correctness and answer prediction, and task 3/4 data as 948 questions, 4,918 students, 1,382,727 answers.
- FoundationalASSIST Hugging Face card: gated/manual access; describes 1.7M interactions with natural problem text and student answers.
- MalruleLib GitHub: 101 executable mathematical misconceptions across 22 categories.
