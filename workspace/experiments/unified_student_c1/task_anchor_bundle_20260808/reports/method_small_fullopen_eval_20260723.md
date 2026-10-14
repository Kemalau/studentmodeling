# Method-Small Full-Open Evaluation

Date: 2026-07-23

## Run

- Experiment: `method-small-fullopen-eval-retry-20260723`
- Job: `qwen3-32b-method-small-full-open-eval`
- Status: `Pass`
- Compute: Feeds, Standard SLA, `80G4-A100` (4 A100 80GB GPUs)
- Evaluation set: frozen `method_small_v1/test.jsonl`, 2,900 rows
- Comparison: one-epoch full-open Qwen3-32B adapter versus the Qwen3-32B base model
- Adapter: `projects/v-yuejiali/amlt-results/7215295067.10143-255b28af-1e91/model/final_adapter`
- Remote result prefix: `projects/v-yuejiali/amlt-results/7215221285.32786-f2e6cda1-33d6`

The first submission was terminated by a backend `SIGTERM` while staging the
61 GiB base model and became a zombie run. It was explicitly killed after its
output had stopped for about 30 minutes. The retry used `max_attempts: 2`,
completed all four evaluation processes, and passed.

## Weighted overview

These aggregates are weighted by the number of examples in each protocol.

| Task | Primary quick-test metric | Adapter | Base | Delta |
|---|---|---:|---:|---:|
| T1 | accuracy, 1,000 rows | 0.6720 | 0.5770 | +0.0950 |
| T2 | response math equivalence, 600 rows | 0.4617 | 0.2533 | +0.2083 |
| T3 | normalized exact taxonomy match, 500 rows | 0.0800 | 0.0460 | +0.0340 |
| T3 | format-valid rate | 1.0000 | 0.8340 | +0.1660 |
| T5 | joint correctness + response math equivalence, 600 rows | 0.7150 | 0.3033 | +0.4117 |
| T5 | correctness accuracy | 0.7983 | 0.4183 | +0.3800 |
| T5 | format-valid rate | 1.0000 | 0.6200 | +0.3800 |
| T6 | accuracy, 200 rows | 0.8650 | 0.8150 | +0.0500 |
| T6 | weighted descriptive macro-F1 | 0.4638 | 0.4397 | +0.0241 |

## T1 correctness

| Source / protocol | n | Adapter accuracy | Base accuracy | Adapter AUROC | Base AUROC |
|---|---:|---:|---:|---:|---:|
| DBE-KT22 temporal warm-item | 200 | 0.6000 | 0.5150 | 0.6952 | 0.5657 |
| Eedi Response profile-heldout | 200 | 0.5400 | 0.5050 | 0.5723 | 0.5182 |
| FoundationalASSIST temporal warm-item | 200 | 0.5850 | 0.4450 | 0.6182 | 0.5645 |
| Malrule cross-template | 93 | 1.0000 | 0.8817 | 1.0000 | 1.0000 |
| Malrule same-template | 107 | 1.0000 | 0.8879 | 1.0000 | 1.0000 |
| XES3G5M temporal warm-item | 200 | 0.6350 | 0.5350 | 0.6589 | 0.6295 |

## T2 response prediction

| Source / protocol | n | Adapter math equivalence | Base math equivalence |
|---|---:|---:|---:|
| DBE-KT22 closed choice | 150 | 0.4800 | 0.2933 |
| Eedi Response closed choice | 150 | 0.4000 | 0.0667 |
| FoundationalASSIST structured open math | 150 | 0.4667 | 0.2267 |
| Malrule cross-template open | 66 | 0.5000 | 0.4545 |
| Malrule same-template open | 84 | 0.5000 | 0.4048 |

The adapter had a 1.0 format-valid rate on every T2 protocol. Base format
validity ranged from 0.1667 on Eedi Response to 0.9524 on same-template
Malrule.

## T3 misconception diagnosis

| Source / protocol | n | Adapter exact match | Base exact match | Adapter format-valid | Base format-valid |
|---|---:|---:|---:|---:|---:|
| Eedi-Mining open taxonomy | 300 | 0.0133 | 0.0000 | 1.0000 | 0.8567 |
| Malrule cross-template open | 93 | 0.1828 | 0.0753 | 1.0000 | 0.7849 |
| Malrule same-template open | 107 | 0.1776 | 0.1495 | 1.0000 | 0.8131 |

This run only provides normalized exact taxonomy matching for T3. The metrics
files explicitly mark full taxonomy retrieval as pending. A semantic judge or
human audit is still required before treating T3 as the paper-grade semantic
diagnosis result.

## T5 prospective prediction

| Source / delta | n | Adapter joint math | Base joint math | Adapter correctness | Base correctness |
|---|---:|---:|---:|---:|---:|
| DBE-KT22, delta 3 | 150 | 0.8133 | 0.4733 | 0.8133 | 0.5000 |
| FoundationalASSIST, delta 1 | 141 | 0.6738 | 0.2270 | 0.7447 | 0.4043 |
| FoundationalASSIST, delta 3 | 152 | 0.6711 | 0.2434 | 0.7961 | 0.3618 |
| FoundationalASSIST, delta 5 | 157 | 0.7006 | 0.2675 | 0.8344 | 0.4076 |

The adapter had a 1.0 format-valid rate for every T5 group. Base weighted
format validity was 0.62.

## T6 strict recurrence

| Source | n | Adapter accuracy | Base accuracy | Adapter AUROC | Base AUROC | Adapter positive recall | Base positive recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBE-KT22 | 12 | 0.9167 | 0.0833 | 0.3636 | 0.0909 | 1.0000 | 0.0000 |
| FoundationalASSIST | 188 | 0.8617 | 0.8617 | 0.5533 | 0.4917 | 0.0000 | 0.0000 |

T6 accuracy is misleading because FoundationalASSIST has a 0.1383 positive
rate and both models predict no positive examples there. The adapter improves
FoundationalASSIST AUROC from 0.4917 to 0.5533 and AUPRC from 0.1654 to 0.1734,
but it does not improve thresholded positive recall. The DBE-KT22 group has
only 12 examples and should not carry a strong conclusion.

## Interpretation

On this development quick test, the one-epoch adapter improves the selected
aggregate metric for every task. The strongest result is T5: weighted joint
math equivalence rises from 0.3033 to 0.7150. T2 also improves substantially,
and output-format validity reaches 1.0 across all generation protocols. T1
gains are broad across sources. T3 improves exact taxonomy matching but remains
low and requires semantic evaluation. T6 remains weak under the meaningful
minority-class diagnostics despite a higher aggregate accuracy.

This is the 2,900-row method-development test, not the full paper-grade formal
evaluation and not a multi-seed result.
