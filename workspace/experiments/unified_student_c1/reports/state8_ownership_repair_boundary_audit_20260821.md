# State8 ownership repair-boundary audit (2026-08-21 08:25 UTC)

## Scope

This audit separates recoverable execution failures from scientific failures
for the State8 future-profile ownership v3 lineage.  A recovery is admissible
only when it preserves the frozen data, sample order, seed, objective,
checkpoint identity, State8-only information route, and evaluation protocol.

## Authoritative v3 lineage

| experiment | terminal/current state | interpretation |
|---|---|---|
| `state8-future-profile-ownership-v3-20260820` | killed | external termination after model load; no scientific result |
| `state8-future-profile-ownership-v3-recovery1-20260820` | killed | external termination during model copy; no scientific result |
| `state8-future-profile-ownership-v3-recovery2-20260820` | killed after Python failure | implementation failure: the auxiliary ownership loss attempted a second backward through a freed graph |
| `state8-future-profile-ownership-v3-recovery3-20260820` | running, 21 hours at the audit timestamp | aligned single-A100 recovery; training is complete and the job is in evaluation |

The active recovery has already published checkpoint hashes
`e392499c...c9e1` (adapter) and `e5f681a6...c5d84` (recurrent state).  Its
strict profile-generation evaluation completed with 88 held-out students:

- schema-valid rate: `1.0`;
- exact profile accuracy: `0.0`;
- future-weakness macro F1: `0.278409`;
- profile-generation scientific gate: `false`;
- swap diagnostic `D_profile_specific`: `0.08`;
- same-profile control difference: `0.02`.

These profile-generation numbers are failures of that output gate, not
infrastructure errors.  The reusable-interface evaluation is a distinct,
preregistered revised-C4 test and was still extracting the 663 fitting-split
states at the audit timestamp; its result must not be inferred from the
profile-generation result.

## Ownership evidence before lrfix-v4

Independent checkpoint audits show no significant student ownership:

| checkpoint | native RMSE | permuted minus native RMSE | one-sided p |
|---:|---:|---:|---:|
| 100 | 0.52509 | +0.00184 | 0.1984 |
| 200 | 0.53436 | -0.00011 | 0.5257 |
| 400 | 0.49247 | +0.00131 | 0.1379 |
| 500 | 0.49639 | +0.00001 | 0.4908 |

The trend does not support an ownership claim.  Selecting checkpoint 100 or
400 after observing these values would also be post-hoc and would still miss
the preregistered `p < 0.05` gate.

## lrfix-v4 audit

The lrfix-v4 code is evaluator-aligned in structure: it uses the same
layerwise mean-slot random-projection view, a future-KC regression loss, a
mismatched-student margin, train-only bias initialization, a separate head
learning rate, layer normalization, and a train-support-only KC mask.  The
decoder still sees State8 only.

However, the immutable 200-step lrfix-v4 checkpoint has already been audited
locally on all 88 DBE development students with 2,000 ownership permutations:

- native RMSE: `0.195329`;
- permuted mean RMSE: `0.194752`;
- permuted minus native: `-0.000577`;
- one-sided ownership p: `0.945027`;
- ownership direction: negative.

Therefore lrfix-v4 does **not** pass its scientific promotion gate on the
available independent result.  The queued experiment
`state8-profile-ownership-lrfix-v4-smoke-recovery-audit-20260821` is an
evaluation-only reproduction of the same immutable checkpoint.  It can repair
the original terminal-step metadata failure and establish an independent
remote receipt, but it cannot convert the local scientific failure into a
pass.  Its watcher correctly writes `scientific_gate_failed` and does not
launch the formal run when the preregistered gate fails.

The queued four-A100 hash-fix experiment
`state8-profile-ownership-lrfix-v4-world4-hashfix-20260821` is only an
eight-example distributed execution diagnostic.  Passing it establishes
collective ordering and history integrity; it supplies no ownership evidence
and cannot independently unlock formal training.

### Loss-scale and capacity diagnosis

The failure is not an evaluator wiring artifact.  The early audit explicitly
loads the trained `profile_ownership_head` saved inside the immutable recurrent
checkpoint.  Thirty-two focused objective, checkpoint-audit, and promotion
tests pass locally.  The empirical problem is generalization:

- the auxiliary head has `4096 x 30 + 30 = 122,910` trainable parameters for
  only 663 profile-training students;
- its held-out RMSE (`0.195329`) is not merely nonsignificant but is worse than
  the train-KC population-mean baseline (`0.117849`) on the same 88 students;
- the 140 profile rows seen in the first 200 steps report a positive mean
  training hard-negative preference (`+0.06130`), while held-out ownership
  reverses direction, which is consistent with fitting the deterministic hard
  negatives rather than learning reusable student structure; and
- on the 60 profile-pair rows, legacy text/semantic causal objectives produce
  a mean total reported loss of about `73.44`, whereas the numeric
  future-rate-plus-ownership objective averages only `0.0373`.  Normalized
  causal exposure raises the early-label effective weight to `3.333`, so a
  nominal ownership weight of one does not imply comparable representation
  pressure.

This diagnosis rules out running the unchanged 1,000-step lrfix-v4 recipe as
the next scientific action.  A separately named development v5 must remove or
strictly cap the legacy profile-text losses, use a capacity-controlled numeric
readout, replace one fixed hardest negative with broader in-batch ownership
contrasts, and gate promotion on the complete frozen reusable-interface
validation rather than on training-head fit.  The frozen XES test remains
untouched until that new DBE development gate passes.

## Automatic-repair policy

Allowed without changing experimental semantics:

1. retry status/log/result-download operations after transient authentication
   or network failures, using the existing browser-login cache;
2. repair an artifact reader when immutable hashes prove that training output
   is unchanged;
3. update a fail-closed source hash when the audited source is otherwise
   identical in objective and protocol;
4. resume or rerun after preemption/NCCL failure using the same checkpoint,
   seed, data order, objective, and evaluation contract; and
5. rerun an independent audit on the same immutable artifacts.

Not allowed as an automatic recovery:

1. changing the ownership weight, margin, head learning rate, normalization,
   KC-support rule, data split, seed, or objective after seeing a failed metric;
2. choosing a checkpoint because its observed p-value is relatively better;
3. treating a failed profile, ownership, readability, stability, or retention
   gate as an infrastructure failure;
4. promoting an execution-only diagnostic as scientific evidence; or
5. inspecting or tuning on the frozen XES confirmation split before every DBE
   development gate and independent artifact audit passes.

## Verification

Forty focused tests covering the ownership objective, checkpoint audit,
independent audit, smoke promotion gate, four-rank gate, and recovery watchers
passed locally on 2026-08-21.  Test success establishes implementation and
fail-closed behavior only; it is not evidence that revised C4 passes.
