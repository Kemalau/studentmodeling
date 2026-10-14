# Revised C4 reusable-state status (2026-08-21 UTC)

## Claim

The primary C4 claim is now **reusable student-state validity**:

> A frozen, history-closed State8 representation supports future student
> prediction and readable weakness profiles for held-out students; pairing a
> state with its owner's outcomes matters, and downstream predictions are
> stable across disjoint views of that student's observed history.

Literal cross-student donor-text transport is diagnostic only.  It is not a
scientific gate because an arbitrary latent transplant need not remain in the
training distribution.

## Primary gates

The frozen evaluator in `scripts/evaluate_state8_reusable_interface.py` requires:

1. at least 80 held-out students;
2. State8 RMSE and weak-profile F1 to beat a train/validation population prior
   with paired-bootstrap probability greater than 0.95;
3. State8 to beat a same-window KC correctness/count baseline on both metrics
   with paired-bootstrap probability greater than 0.95;
4. held-out state-ownership permutation p-value below 0.05;
5. disjoint-history prediction-stability p-value below 0.05; and
6. a closed route: the readout receives neither raw history nor student ID.

The disjoint-history p-value is interpreted only jointly with the predictive,
same-window, and ownership gates above.  Odd/even histories are complementary,
not replicate measurements, and a student-invariant constant state can appear
numerically stable.  The evaluator therefore also reports a fitting-split-
scaled, noise-adjusted reliability diagnostic that exposes this collapse; it
does not promote that post-hoc diagnostic to an additional scientific gate.

Readout alpha is selected on training/validation students only.  The weakness
candidate KC set is frozen from fitting-split support, rather than selected from
each test student's future support.

## Why the repair is necessary

The earlier DBE ownership checkpoints did not establish student ownership:

| checkpoint | native RMSE | permuted minus native RMSE | one-sided p |
|---:|---:|---:|---:|
| 100 | 0.52509 | +0.00184 | 0.1984 |
| 200 | 0.53436 | -0.00011 | 0.5257 |
| 400 | 0.49247 | +0.00131 | 0.1379 |
| 500 | 0.49639 | +0.00001 | 0.4908 |

All are development diagnostics, not confirmatory results.  The lrfix-v4
branch therefore adds the missing objective aligned with the evaluation:

- future-KC rate regression from the exact frozen State8 feature view;
- a hard mismatched-student ownership margin;
- a separate `3e-4` auxiliary-head learning rate;
- deterministic head reset with train-only shrunk KC-rate biases;
- per-example layer normalization; and
- exclusion of KCs supported by fewer than four training students.

The final decoder remains strict state-only.  Post-cutoff labels supervise the
training loss but are not rendered into the updater or decoder input.

## Live execution chain

1. **Bounded DBE development smoke**
   - Experiment: `state8-profile-ownership-lrfix-v4-sparsekc-smoke-r2-20260820`
   - Job: `maroon_neck_4l1h8snx5l`
   - Target: Feeds, `80G1-A100`, Standard, one A100 80 GB
   - Training reached the full 200/200 micro-steps and wrote immutable
     checkpoint artifacts.  The original job then failed in the post-training
     audit because the terminal checkpoint records its step under
     `diagnostic_max_micro_steps`, while the evaluator initially looked only
     for a root-level `step` field.
   - Evaluation-only recovery: experiment
     `state8-profile-ownership-lrfix-v4-smoke-recovery-audit-20260821`, job
     `cool_eagle_l94f2y3wr0`, queued on one `80G1-A100` at 2026-08-21 05:59 UTC.
     It binds the original checkpoint hashes, reruns no training, and leaves
     the XES official test untouched.
   - Strict route observed in logs: `history_dropout=1`,
     `history_dropped=true`, `direct_tokens=0`, recurrent state present
   - The promotion gate requires 88 held-out DBE students, 2,000 ownership
     permutations, positive ownership direction, and p < 0.05.

2. **Formal DBE development run**
   - Experiment: `state8-profile-ownership-lrfix-v4-formal-20260820`
   - Exactly 1,000 micro-steps (one 1,000-row development epoch)
   - Automatically submitted only if the bounded smoke passes its artifact and
     significance contract.
   - At this report timestamp it has not been submitted.

   A separate four-rank, two-round execution preflight previously stopped
   before training because its fail-closed evaluator hash no longer matched
   the audited source.  The hash-only rerun is experiment
   `state8-profile-ownership-lrfix-v4-world4-hashfix-20260821`, job
   `dynamic_plate_qb7c9h4vvj`, queued on `80G4-A100` at 2026-08-21 05:59 UTC.
   Its watcher requires the exact eight global positions, four-rank agreement,
   and history-integrity receipt before replacing the failed preflight marker.

3. **One-shot XES confirmation**
   - Experiment: `state8-xes-profile-confirmatory-lrfix-v4-20260821`
   - Frozen split: 800 train, 200 validation, 466 official-test students
   - The official test is touched only after the formal DBE run passes every
     reusable-interface gate and the independent artifact audit.
   - At this report timestamp official-test metrics have not been inspected.
   - A failed confirmation must be reported as a failure; repairing after
     inspection requires a new untouched confirmatory dataset.

## Verification completed locally

Forty focused end-to-end gate tests plus seventeen lrfix-v4-specific tests
passed for the reusable-interface evaluator,
ownership objective, XES split construction/freeze verification, AMLT config,
recovery paths, and both promotion controllers.  The recovery-stage promotion
validator was also repaired so a valid evaluation-only receipt cannot be
rejected solely due to its watcher-stage name.  The paper abstract,
introduction, method,
related work, RQ4, and conclusion have been updated to describe State8 and the
revised C4.  A local LaTeX compiler is not installed, so only structural TeX
checks were possible in this environment.

The evaluator has additionally been hardened with a diagnostic-only,
fitting-variance-scaled reliability statistic for the odd/even history views.
It reports zero signal for a constant-state collapse.  This does not alter the
preregistered gates or retroactively change the active smoke job; it will be
included in formal/confirmatory evaluations launched from the updated source.

## Current conclusion

The project is not failed, but revised C4 is not yet established.  The active
branch now tests the defensible claim with a closed, student-disjoint,
uncertainty-audited protocol.  The next decisive number is the 200-step DBE
ownership-permutation p-value.  Both missing mechanical preconditions now have
live, monitored recovery jobs; only a significant smoke may unlock the formal
DBE run, and only a passed formal DBE run may unlock the untouched XES
confirmation.
