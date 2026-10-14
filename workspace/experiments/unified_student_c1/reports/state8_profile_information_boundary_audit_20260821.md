# State8 profile information-boundary audit (2026-08-21)

## Result

The current future-profile protocol is chronologically clean and student
disjoint, but its primary target is mostly outside the updater's observable KC
support.  This makes all-cell future ownership a mixture of two materially
different questions:

1. **observed-support ownership** — does State8 preserve which student's
   evidenced KC outcomes it encodes? and
2. **unseen-KC transfer** — can a short recent window predict performance on
   KCs absent from that window?

These must be reported separately.  Failure on the second question must not be
interpreted as evidence that the representation has no student ownership.

## Audited information boundary

| dataset / split | students | median cutoff | updater-visible history | future-supported cells seen in updater | future weaknesses seen in updater | students with neither future weakness seen |
|---|---:|---:|---:|---:|---:|---:|
| DBE train | 663 | 90 | 12 | 836 / 4669 (17.91%) | 15.31% | 76.47% |
| DBE valid | 84 | 89.5 | 12 | 97 / 579 (16.75%) | 14.88% | 77.38% |
| DBE test | 88 | 91 | 12 | 114 / 634 (17.98%) | 16.48% | 75.00% |
| XES train | 800 | 132 | 12 | 399 / 3731 (10.69%) | 10.88% | 80.38% |
| XES valid | 200 | 129.5 | 12 | 111 / 941 (11.80%) | 10.50% | 81.50% |
| XES official test | 466 | 125 | 12 | 244 / 2171 (11.24%) | 12.34% | 77.25% |

The DBE input contains only about 17% of the pre-cutoff prefix on average;
the XES cutoff is even longer.  The future target is therefore dominated by
KCs for which the model receives no direct student evidence.

## Data-only positive control

As a sanity check, the exact updater-visible interactions were reduced to a
Beta(1,1) correctness rate per KC.  Evaluation was restricted to future-
supported KCs also seen in that visible history.  Student ownership was then
permuted 2,000 times.  This loads no model and fits nothing on test labels.

| dataset / split | native RMSE | permuted RMSE mean | permuted minus native | one-sided p |
|---|---:|---:|---:|---:|
| DBE train | 0.1934 | 0.2561 | +0.0627 | 0.00050 |
| DBE valid | 0.1686 | 0.2572 | +0.0887 | 0.00050 |
| DBE test | 0.1895 | 0.2801 | +0.0906 | 0.00050 |
| XES train | 0.2061 | 0.2914 | +0.0853 | 0.00050 |
| XES valid | 0.2248 | 0.2950 | +0.0702 | 0.00050 |
| XES official test | 0.2441 | 0.2911 | +0.0470 | 0.00050 |

Thus the supplied histories contain strong student-specific ownership signal
where the relevant KC was actually observed.  The scientific failure of the
current all-cell State8 gate cannot be attributed to a complete absence of
student-specific signal in the source data.

## Required C4 reporting split

- **C4a, primary ownership/readability:** evaluate student-permutation and
  readable KC profiles on future-supported KCs also observed in the updater
  window.  Compare State8 against the exact same-window KC-statistics baseline.
- **C4b, future transfer:** evaluate KCs absent from the updater window and the
  full all-cell target separately.  Population and same-window baselines remain
  mandatory.  This is a stronger generalization result, not a prerequisite for
  saying that the state has student ownership.
- Preserve the strict final route: readout sees State8 only, not raw history,
  student ID, future support, or future outcomes.
- Do not reinterpret the data-only positive control as State8 evidence; State8
  must independently beat or add value over that same-window control.

## Reproducibility

The independent script is
`scripts/audit_state8_profile_information_boundary.py`.  Machine-readable
outputs are:

- `reports/state8_profile_information_boundary_dbe_20260821.json`
- `reports/state8_profile_information_boundary_xes_20260821.json`

The accompanying focused tests pass (`2 passed`).
