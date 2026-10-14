# C4 execution pilot: infrastructure recovery1 receipt

## Failure and repair

Both original jobs terminated before the training entrypoint. Azure Run History
reports the identical `IndexError: 2` at test line 40: Python eagerly evaluated
`ROOT.parents[2]` as an `os.environ.get` default even though `C4_TEST_TOKENIZER`
was set. `/scratch/amlt_code` has only two parents. This is a startup failure,
not a scientific negative result; neither original job produced optimizer
checkpoints or predictions. Original experiment and hash manifest are preserved.

Repair: resolve the explicitly supplied tokenizer path before evaluating the
local fallback; fail clearly if the fallback is unavailable. Add a regression
test with the exact shallow remote root. The runner now writes stdout/stderr
and its exit code into the output directory. These are the only uploaded source
changes. Frozen data, training/model/decoding code, initial adapter, target order,
240 updates, and all scientific endpoints are unchanged.

CPU tests after repair: **9 passed in 9.36s**, including data isolation, matching,
gradient/checkpoint tests, collector corruption rejection, quota checks and the
new remote-path regression. All 13 recovery source/protocol/data hashes pass;
shell syntax and AMLT parsed-config validation pass. No local optimizer update.

## Submission

Experiment: `c4-execution-training-v1-seed42-recovery1-20260906`.

| Arm | Job ID | Verified at 12:15 UTC |
|---|---|---|
| A, matched candidate-free SFT | `stoic_queen_l2zrtkcpcq` | queued |
| B, differentiable state exchange | `tidy_card_gdqp59xzfl` | queued |

Target: user-approved **Feeds-LLM**, Standard, one `80G1-A100` per arm. Final
quota snapshot: Standard 7/80, Basic 0/80, Premium 140/144, personal cap null
(not a claim of 80 personal slots). Target-scoped fresh personal request audit:
7 A100 units, auditable, no errors. No unrelated job stopped.

Workspace remains `CS-NewsAndFeeds-Singularity` in
`rg-cs-newsandfeeds-singularity`, primary subscription
`b6dc87f3-c479-49c8-8cb5-7896da3ff895`. Target remains in WPP subscription
`078ad34f-9e5b-4962-9d78-c9448646548e`. Explicit UAI `rankfun_aml`, fixed image
`amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`.
Validated current-user CLI cache `/home/v-yuejiali/.azure`; no device-code login.

Remote base `/mnt/amulet/studentmodeling/models/Qwen3-32B`; remote initial adapter
`/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter`.
Read access confirmed; previous execution proves staging succeeded under this UAI.
Only 14 small files (2.6 MB) uploaded, no local base-weight download.

Remote outputs in `csnewsandfeeds4150361735/amulet`:

- A: `projects/v-yuejiali/amlt-results/7211303140.00646-9f11fa46-2ae7/execution_A`
- B: `projects/v-yuejiali/amlt-results/7211303140.00542-f9ce2af5-f3be/execution_B`

## Monitoring and completion requirements

Durable monitor PID 3938648, native AMLT watch PID 3938667, both verified live.
Terminal-only notification; no resubmission or cancellation ability. After pass,
download JSON/JSONL/logs only and run the paired collector. Large adapters and
optimizer checkpoints remain remote (submission host has about 3 GiB free).

Authoritative live-state file:
`results/c4_execution_training_v1_recovery1_20260906.status.json`.
Event log: same prefix `.monitor.log`; eventual audited metrics `.summary.json`.
Neither queued status nor loss reduction is scientific completion. Require both
final 240-step adapters and 960 complete, unique, parse-audited test records for
each of baseline, A and B (2,880 total), then report paired specificity and all
controls regardless of sign. See the unchanged frozen protocol for full methods.

Reproduction files: `amlt/feeds_c4_execution_training_v1_recovery1_20260906.yaml`,
`manifests/c4_execution_training_v1_recovery1_20260906.sha256`,
`scripts/monitor_c4_execution_recovery.py`.

## Live validation after start

At approximately 12:21 UTC, both remote jobs passed all four direct-run
protocol/gradient/path checks and staged 61 GiB of base weights. B loaded all
17 shards and emitted actual optimizer records through step 11/240 (oracle
phase), loss 0.2095735683 at step 11, finite gradient norm 0.5502747, about
9.2 seconds per step, peak allocated VRAM 29.7876 GiB. These are training
progress metrics, not causal efficacy. A runs the 960-record baseline before
its first optimizer update. The native watcher remains live. Detailed AML
stdout is readable with `scripts/read_c4_remote_log.py A` or `B` while temporary
signed log URLs are kept out of console output. The Blob-mirrored runner log
can lag; an empty mirrored file is not proof the remote process is stalled.

An additional CPU delivery audit is prepared in
`scripts/audit_c4_execution_delivery.py`, with its synthetic regression test
passing. It checks all 240 training steps, finite losses/gradients, exact target
exposure, validation-120/240, final checkpoint lineage, per-rule counts, no-op
token mismatches, and all-three-query outcomes without selecting checkpoints.

## Checkpoint and partial-file audit at 12:30 UTC

Both jobs are still running; the monitor and native watch PIDs remain live.
B checkpoint-40 exists remotely with a 268,555,264-byte adapter, 537,639,453-byte
optimizer/RNG checkpoint, and atomic completion receipt. Adapter SHA256:
`fd28f3c310d6aaa813c9a1b5942a4ef43dc1d780a9e8d98971d3f5921da15bab`.
B progressed beyond it to step 65 (loss 0.0643657149). Do not interpret the
single-step loss as a task or causal success metric.

A baseline has five complete group files (150 generation records). A lightweight
integrity check found 30 unique query-condition keys per group, the expected
baseline fingerprint, and no native/no-op token or stopping discrepancy across
15 queries. This is partial pipeline validation only, not a full efficacy audit.

The delivery auditor now additionally requires group-level fingerprints, all
state-hash roles, frozen-baseline identity, and optional raw-token/text equality
using the local tokenizer. Its synthetic corruption tests pass (2 tests).
These local postprocessing changes are outside the uploaded scientific package;
all 13 frozen recovery hashes continue to pass. Remaining work: full 240-step
A/B training, all baseline/final groups, paired collection, and full audit.

## Training completion and platform interruption at 13:04 UTC

B completed **240/240** optimizer updates with **26,524** supervised tokens
and final scheduler LR zero. Training wall time including checkpoint/validation
work was 2,416.31 seconds. Final step loss was 0.03499810 (not an aggregate
accuracy). Checkpoint-240 and final adapter SHA256:
`87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32`.
Final adapter is present remotely (268,555,264 bytes); final evaluation began
and logged group 1/32. Full causal results remain pending.

| B validation checkpoint | Explicit-rule loss | History/exchange loss | Tokens per condition |
|---|---:|---:|---:|
| 120 | 0.1175519905 | 0.2680753307 | 656 |
| 240 | 0.1320816565 | 0.1422244660 | 656 |

These are teacher-forced validation losses, not free-generation hit rates or
causal specificity. Report the explicit-rule increase alongside the history
decrease; do not infer B outperforms matched A before A is trained/evaluated.

A logged baseline group 21/32, then received SIGTERM at 12:54 UTC. Platform
activity reports `ContainerTerminatedDueToCancelation`, exit 143, external
node-agent cancellation. It attempted a new pod, suspended, then automatically
requeued at 13:01:56 and restarted program execution at 13:03:28. No duplicate
experiment was submitted and no agent cancellation was issued. A resume command
at the state transition returned "specified jobs cannot be resumed"; the
subsequent queue/deployment was automatic, not evidence of a successful resume
command. A's 21 completed group files remain in remote output storage. Whether
the restarted process rehydrates them or recomputes them must still be checked;
do not claim successful application-level resume merely from platform Running.
Latest retry stdout is `user_logs/retry_001/std_log_process_0.txt`.

Image provenance clarification: the requested YAML image tag is the June tag
listed above, but both jobs' actual platform activity logs resolve it to
`aiscofficialameacr.azurecr.io/base/job/pytorch/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260818T030356015`.
A's automatic retry resolves to the same actual August image. Preserve both
requested and observed image identities rather than claiming the requested
alias was itself immutable. Scientific source hashes remain unchanged.

## Application-level resume confirmed at 13:10 UTC

A's retry stdout passed all four direct-run checks, reloaded all 17 model
shards, and its **first new baseline completion was group 22/32**. The prior
21 groups were recovered rather than recomputed. Native AMLT watch now reports
both A and B running again. B's final evaluation reached group 5/32. A has not
yet started optimizer updates because the baseline precedes its training.

The downloaded B training log contains exactly steps 1 through 240, without
gaps or duplicate steps, and every loss/gradient norm is finite. Final
supervised-token count equals 26,524. This establishes completed B training,
not completed evaluation or success of the scientific hypothesis. Remaining
requirements are unchanged: finish baseline, A training and final evaluation,
B final evaluation, then full paired audit. Monitor PIDs 3938648/3938667 remain
live; no experiment has been marked complete.

## Completed baseline integrity audit at 13:28 UTC

A completed all 32 baseline groups after application-level resume and has begun
its 240-step training. The downloaded baseline passes `audit_eval` and
`audit_prediction_lineage` with the local hash-pinned Qwen3 tokenizer: 960 unique
expected records, group/merged equality, prompt reconstruction, strict parsing,
frozen initial-adapter fingerprint, and raw token/text equality. Native versus
own-state no-op has **0/96** token-or-stop mismatches on this baseline.

Baseline candidate-free donor hits: directed **4/96**, wrong-rule random **3/96**,
specificity **0.0104167**, exploratory 16-pair cluster 95% CI
**[-0.03125, 0.0520833]**. Native recipient and native donor accuracy are each
**4/96**; explicit-rule donor accuracy **35/96**. Separately, the candidate-visible
control has directed **74/96** versus random **12/96** donor hits. These menu
counts are not unsupported free-generation results and must not be mixed with
historical panels or the primary endpoint. Baseline no-op validity does not
establish no-op validity for either trained adapter.

The current local protocol/collector/delivery-audit/report tests pass (**8 passed
in 10.54s**); all 13 recovery scientific source/data/protocol hashes still match.
This is completed baseline evidence only. B final evaluation and A training/final
evaluation remain incomplete, so no A/B training-effect conclusion is available.

## Durable full-audit handoff at 13:32 UTC

The native monitor and its terminal-only watcher remain live. Added a local
post-collection finalizer, PID **4133559**, which waits for the existing monitor's
complete receipt, verifies that its collector summary exists, then runs the full
token-aware delivery audit and renders the prepared results report. It cannot
submit/cancel/resume remote jobs. A live `observation_error` state is treated as
waiting, not scientific failure. Its lifecycle regression tests pass (2 tests).
Receipt: `results/c4_execution_training_v1_recovery1_20260906.finalizer.json`.
Expected final audit: same prefix `.delivery_audit.json`; expected report:
`reports/c4_execution_training_results_20260906.md`.

The local collector/report now also computes the protocol's secondary paired
specificity changes A-minus-baseline and B-minus-baseline, using the identical
16-pair bootstrap and fixed final endpoints. B-minus-A remains primary; all
secondary intervals are exploratory and unadjusted. These postprocessing edits
are not uploaded and do not alter data, training, decoding, or source fingerprints.
The frozen scientific package is unchanged. Do not confuse the new waiting
finalizer receipt with a completed experiment.

## Matched-initial-phase check at 13:33 UTC

Downloaded A steps 1–44 match B's corresponding steps exactly in loss,
gradient norm, learning rate, phase, and cumulative supervised-token count.
Their checkpoint-40 adapter hashes are also identical:
`fd28f3c310d6aaa813c9a1b5942a4ef43dc1d780a9e8d98971d3f5921da15bab`.
This is the expected shared explicit-rule phase, before exchange training begins
at step 121. It verifies this part of exposure/initialization matching despite
A's pre-training platform retry; it is not a causal-effect result or proof about
future checkpoints. B currently has 20/32 final groups downloaded.

## Second A platform interruption at 13:42 UTC

A reached step 75 with finite loss/gradient logs, then the same job returned to
Queued. Azure Run History reports `CommonRuntime / ServiceError` from
`/opt/microsoft/azureml/common_runtime/data_capability/data_capability.sh`: a
termination signal and unexpected process exit, not a reported model-loss or
Python-training failure. AML v2 shows Queued with no structured error; retain
the more detailed Run History error rather than interpreting that null as
healthy execution. Last-start metadata advances to 13:42:09 UTC.

No new experiment was submitted and no manual cancellation/resume was issued.
Checkpoint-40 has an atomic completion receipt; no later complete checkpoint
was visible in the downloaded artifact listing. If the restart restores it,
steps 41–75 can be replayed and must be disclosed. Application-level restart
has not yet been verified. B continues normally, at 26/32 final evaluation
groups by the next log check. Full A/B completion remains pending.

## Second application-level recovery confirmed at 13:48 UTC

A returned to Running on the same job, restaged the base weights, passed its
preflight, and resumed optimizer records at step 41 after checkpoint-40. The
new step 42/43 loss, gradient norm and learning rate match B's corresponding
original shared-phase records exactly. Baseline groups were not regenerated.
The output log now appends replayed steps; the full audit will retain and report
them. This is restored-checkpoint evidence, not a completed A training result.
The fresh stdout reused `user_logs/retry_001/std_log_process_0.txt`; a probe of
`retry_002` returned ArtifactNotFound, not a job failure. B is at 29/32 groups.

## B complete-output audit at 13:52 UTC — A comparison still pending

B's runner completed group 32/32 and logged exit 0 at **13:51:23 UTC**.
Its complete receipt, final-adapter lineage, all 240 finite training steps,
validation checkpoints, 26,524-token exposure, 960 unique predictions, group
versus merged records, prompt hashes, strict parsing, and raw token/text equality
pass the relevant delivery-audit functions. Both B and baseline have 32 complete
groups; B prediction SHA256 is
`1bb454a0017db601d6ebdd2485721bb6c0411eace34642a65c83081b3130389d`.

| Candidate-free endpoint | Initial adapter | B final |
|---|---:|---:|
| Directed donor-target hits | 4/96 | 56/96 |
| Wrong-rule random donor-target hits | 3/96 | 8/96 |
| Specificity D | 0.0104167 | 0.5000000 |
| D, exploratory pair-cluster 95% CI | [-0.03125, 0.0520833] | [0.2604167, 0.7395833] |
| Native recipient / native donor accuracy | 4/96 each | 56/96 each |
| Same-donor-rule donor-target hits | 4/96 | 53/96 |
| Explicit-rule donor-target hits | 35/96 | 60/96 |
| All-three directed donor matches | 0/32 | 14/32 |
| All-three random donor matches | 0/32 | 0/32 |

B-minus-baseline specificity is **0.4895833**, exploratory 16-pair paired
bootstrap 95% CI **[0.2604167, 0.71875]**. B joint-three difference is 0.4375,
CI [0.21875, 0.65625]. These are secondary contrasts relative to the still
pending primary B-minus-A comparison, with one seed and unadjusted intervals.

The gain is strongly heterogeneous: the two fraction rules contribute directed
**46/48** versus random **0/48** hits; subtraction contributes **10/48** versus
**8/48**. Per-rule B directed/random: ignore-denominators 24/24 vs 0/24;
keep-common-denominator multiplication 22/24 vs 0/24; stops-borrow-at-zero 4/9
vs 1/9; diff-0-n-equals-n 3/12 vs 3/12; smaller-from-larger 2/18 vs 2/18;
borrow-no-decrement 1/9 vs 2/9. Do not claim robust execution of all six rules.

B has **1/96** native/no-op token mismatch (baseline 0/96): for
`8002 − 3374`, native generates 4732 and no-op 4632; donor/recipient targets
are 4378/4772, so both remain misses. This does not change the accuracy counts,
but forbids an exact no-op-invariance claim. B random and same-donor controls
each contain one invalid output; all are retained as misses.

Concrete first joint success in manifest order, `test:fractions:000:0`:
donor history follows ignore-denominators. The same captured donor trajectory
produces 154, 234 and 45 on `(11/29)*(14/29)`, `(18/30)*(13/30)` and
`(5/24)*(9/24)` respectively, without a candidate list. Random-state outputs
are `154/29`, `19.5/15` (strictly invalid), and `15/8`. This is an illustrative
held-out success, not an example-filtered metric or unseen-rule evidence.

Separately, B candidate-visible directed/random hits are 77/96 vs 13/96,
and native recipient hits 80/96; these are menu controls, not open generation.
This establishes a completed B repair-training result on the frozen panel.
It does **not** establish exchange supervision's incremental value over matched
SFT, general executable rules, or retroactive execution ability in the old
model. A training/final evaluation and the full A/B report remain pending;
the active monitoring goal is not complete.

## B platform pass and A matched midpoint at 14:06 UTC

Fresh AMLT status confirms **B pass**, A running. B's platform pass was first
recorded by the native watcher at 13:57:26 UTC, after its 13:51:23 runner exit.
The watcher later experienced a transient `RemoteDisconnected` status fetch
error, then resumed normal polling without restart; independent reads continued
to confirm A training. Monitor/child/finalizer PIDs remain 3938648/3938667/4133559.

A restored the shared trajectory exactly through checkpoint-120. Its adapter
hash equals B checkpoint-120:
`8902f5dccf6f4342be38fc336e57567b3b7d6a2abfc09d118e504036b4f7414f`.
Both explicit-rule validation losses are 0.11755199046214462 (656 tokens).
History validation differs as expected between native A and exchange B routes:
A 0.26748644904142654 versus B 0.26807533065425004 (656 tokens each); these
are loss diagnostics, not free-generation superiority. A has now entered its
history phase and reached step 132. No evaluation-based model selection.

Checkpoint-80 hashes had also matched exactly. A downloaded log at step 95
contained 129 records and 95 unique steps (34 duplicated-step IDs). Platform
logging can lose an unflushed line around an interruption, so the final replay
audit reports actual retained records rather than assuming all 35 interrupted
updates are logged twice. No original remote artifact is deleted or rewritten.

## A training complete; final evaluation started at 14:32 UTC

A reached **240/240** and its final checkpoint completion receipt was retrieved
at 14:28 UTC. Final adapter SHA256:
`7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6`.
The downloaded training log contains **274 records, 240 unique steps**, with
34 repeated-step IDs (41–49 and 51–75). Every logged loss, gradient norm and
learning rate is finite. The last restarted invocation reports 22,081 supervised
tokens; checkpoint-40 accounts for the first 4,443, giving the planned final
trajectory exposure of **26,524**, identical to B. Replay compute is additional
wall time, not extra updates retained in the final optimizer trajectory.

Final validation files and evaluation lineage exist and refer to that adapter.
A validation at 240: explicit-rule loss **0.13208542809072085**, history loss
**0.14141528401426223**, 656 supervised tokens each. B's corresponding values
are 0.13208165652669898 and 0.1422244660099262. These near-equal teacher-forced
losses do not establish equivalent generation or intervention behavior.

At 14:32 UTC, A emitted final evaluation group 1/32; by 14:33 UTC it reached
2/32. B remains platform Pass. The monitor, native watcher, and full-audit
finalizer remain live. The full 2,880-record paired audit/report is pending A's
remaining 30 groups; no primary B-minus-A result or goal completion is claimed.

## Final completion and delivery audit, 15:28 UTC

A completed evaluation group 32/32 and its runner exited 0 at 15:21:25 UTC.
Fresh AMLT status confirmed both jobs Pass; the native watcher recorded A's
Pass at 15:27:24 UTC. Collection completed at 15:27:34 UTC and the finalizer
completed at 15:28:06 UTC. Monitor/watcher/finalizer processes exited normally.
No further submission, cancellation, training, or checkpoint selection occurred.

All baseline/A/B outputs are complete: 32 groups and 960 unique records each,
2,880 total. Full delivery audit verifies raw token/text equality, parsing,
regenerated prompt hashes, group-versus-merged records, final adapter lineage,
240 unique training steps per arm, and matched 26,524-token target exposure.
The final delivery audit equals the independent audit. The report exactly
reproduces its audited data. Ten CPU protocol/gradient/collector/report/finalizer
tests passed, and all 13 frozen source/protocol/manifest hashes passed.

The long-lived monitor had imported the collector before secondary baseline
contrasts were added. Its summary therefore initially contained the correct
primary contrast and all arm results but omitted the two secondary fields.
Those fields were copied from the independently recomputed final audit; no
predictions or primary metrics changed. The summary now equals the audit core.

| Candidate-free endpoint | Baseline | A: matched SFT | B: state exchange |
|---|---:|---:|---:|
| Directed donor-target hits | 4/96 | 56/96 | 56/96 |
| Wrong-rule random donor-target hits | 3/96 | 6/96 | 8/96 |
| Specificity D | 0.0104167 | 0.5208333 | 0.5000000 |
| Three-query directed agreement | 0/32 | 14/32 | 14/32 |
| No-op token/stop discrepancies | 0/96 | 1/96 | 1/96 |

Primary B-minus-A D is -0.0208333, exploratory pair-cluster 95% CI
[-0.0520833, 0.0000000]: no demonstrated incremental advantage of exchange
supervision. A/B both show improvement over the freshly evaluated baseline,
but this is one seed and 16 independent pairs. Fraction rules account for
46/48 directed versus 0/48 random hits in both arms; subtraction accounts for
10/48 directed versus 6/48 random in A and 8/48 random in B. No-op discrepancies
and A's platform retry/replayed logs remain disclosed. This is narrow
repair-training evidence, not a universal or naturally emerging rule executor.

Final report: `reports/c4_execution_training_results_20260906.md`.
Final audit: `results/c4_execution_training_v1_recovery1_20260906.delivery_audit.json`.
Final summary: same prefix `.summary.json`. The frozen pre-training protocol
is unchanged; this section supersedes earlier pending-status entries.
