# Method Small v1 Dataset

Date: 2026-07-22

## Purpose

`method_small_v1` is a development dataset for iterating on the recurrent and
causally grounded student-state method before scaling back to the full 259,845
example training mixture. It is not a replacement for the formal full test.

## Size

| Split | Rows | Purpose |
|---|---:|---|
| train | 12,000 | one-epoch method training |
| valid | 2,900 | architecture and loss selection |
| test | 2,900 | quick frozen-protocol comparison |

On four GPUs with per-device batch size 1 and no gradient accumulation, one
epoch is 3,000 optimizer steps. The full mixture requires 64,962 steps under
the same global batch, so the small run is approximately 21.7 times shorter.

## Train composition

| Task | Source | Rows |
|---|---|---:|
| T1 correctness | DBE-KT22 | 500 |
| T1 correctness | Eedi Response | 700 |
| T1 correctness | FoundationalASSIST | 1,000 |
| T1 correctness | Malrule | 700 |
| T1 correctness | XES3G5M | 1,200 |
| T2 response | DBE-KT22 | 500 |
| T2 response | Eedi Response | 700 |
| T2 response | FoundationalASSIST | 1,000 |
| T2 response | Malrule | 700 |
| T3 misconception | Eedi-Mining | 1,000 |
| T3 misconception | Malrule | 700 |
| T5 prospective response | DBE-KT22 | 500 |
| T5 prospective response | FoundationalASSIST | 1,800 |
| strict T6 recurrence | DBE-KT22 | 114 |
| strict T6 recurrence | FoundationalASSIST | 886 |

T4 and the XES repeat-incorrectness proxy remain excluded, matching the formal
training protocol.

## Sampling protocol

- Sampling is deterministic with seed `20260722`.
- Temporal sources are ranked by a source-level student cohort hash shared
  across tasks. This makes the selected students overlap across T1/T2/T5/T6
  within a split while keeping students disjoint between splits.
- Per-student caps prevent a few long FoundationalASSIST trajectories from
  dominating the small set.
- Existing structured `history` lists are retained, allowing recurrent methods
  to unroll over 4, 6, or 12 visible interactions rather than parsing prompts.
- Validation uses the original validation students where available.
- Eedi-Mining has no student sequence and no original validation split. Its 200
  validation rows are response-level holdouts whose labels all have a distinct
  training example. The 1,000 training rows maximize label diversity.
- The quick test is sampled only from `formal_evaluation_v1/test.jsonl`; it is
  never used for architecture or loss selection.

## Leakage audit

- Train/validation/test example-ID overlap: zero.
- Train/validation/test student overlap for every longitudinal or synthetic
  student source: zero.
- Eedi-Mining is excluded from student-overlap checks because every row uses the
  explicit `NO_STUDENT_HISTORY` sentinel; its example IDs are still disjoint.

## Important limitations

- This set is designed to compare methods quickly, not estimate final effect
  sizes.
- Eedi-Mining is extremely long-tailed: the 1,000 selected training rows cover
  1,000 labels once each. Its validation set is warm-label, while the quick
  formal test contains both warm and cold labels.
- T6 preserves its natural class imbalance; accuracy must not be the only
  metric.
- Eedi Response remains a non-temporal profile-held-out task.
- Final claims must be rerun on the complete training mixture and full formal
  test with multiple seeds.

## Files

- `data/processed/method_small_v1/train.jsonl`
- `data/processed/method_small_v1/valid.jsonl`
- `data/processed/method_small_v1/test.jsonl`
- `data/processed/method_small_v1/dataset_stats.json`
- `scripts/build_method_small_dataset.py`
