# C4 unseen-rule Feeds-LLM submission receipt

User explicitly requested Feeds-LLM after local resource inspection. Local
supervisor PID 209733 stopped before any inference; checkpoints are preserved.
The frozen unseen-rule protocol is unchanged. This is execution evidence,
not model-performance evidence.

- Experiment: `c4-unseen-rules-v1-seed42-feedsllm-20260906`.
- A: `c4-unseen-rules-arm-a-seed42`, job `red_kitten_cgvxbp4rhc`.
- B: `c4-unseen-rules-arm-b-seed42`, job `purple_kettle_psfk0nmvl5`.
- Submission returned zero; immediate authoritative status: both `preparing`.
- Target: Feeds-LLM through `CS-NewsAndFeeds-Singularity` in
  `rg-cs-newsandfeeds-singularity`, subscription
  `b6dc87f3-c479-49c8-8cb5-7896da3ff895`.
- Resources: two independent Standard `80G1-A100` jobs, one GPU each;
  four-hour runtime limit per job, inherited two-attempt infrastructure policy.
- Approved inherited image:
  `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`.
- UAI: `rankfun_aml`; Storage Blob Data Contributor/Reader roles verified
  for the existing `csnewsandfeeds4150361735` store.
- Authentication: existing `/home/v-yuejiali/.azure` cache; expected identity,
  tenant and subscription verified. No interactive login or credential change.
- Fresh quota: Premium 140/144, Standard 18/80, Basic 0/80; `user_max=null`
  means no personal limit exposed, not a guarantee of pool ownership. Active
  project request audit found 18 A100s on this target before requesting two.
- Inputs: `/mnt/amulet/studentmodeling/models/Qwen3-32B` and the exact A/B
  final-adapter paths/hashes in `manifests/c4_unseen_rules_v1_20260906.json`.
- Outputs: per-job `AMLT_OUTPUT_DIR/unseen_A` / `unseen_B` on the existing
  remote output store; runtime and runner logs beside those directories.
- Upload: 15 files including source-hash manifest, approximately 0.4 MB;
  no local model weights, adapters, credential caches, or unrelated data.
- Validation: 9 CPU tests passed in 8.82 seconds; Python compilation and
  shell syntax passed; full YAML parsing, frozen source hashes, prompt hashes,
  independent formulas and final-adapter hashes passed.

Infrastructure diagnostics were resolved without changing access policy:
the initial role-list query attempted an unnecessary Graph name resolution and
hit conditional-access restrictions; `--fill-principal-name false` verified
the same identity's roles via ARM. The active-job audit initially reported a
missing JSON table for an unrelated zero-job experiment; refreshed metadata
confirmed `n_jobs=0` and `_all_terminated=true`. Only that evidenced empty case
was resolved; unknown nonempty cases remain fail-closed.

Native terminal-only monitor/collector:
`scripts/watch_c4_unseen_rules_feeds.py`. It performs read-only status checks,
downloads result JSON/logs, verifies matching runtime packages, and uses
`scripts/collect_c4_unseen_rules.py` to reject gaps, duplicates, wrong hashes,
wrong prompts/targets and token/text inconsistencies. Only both audited arms
produce a complete paired summary; failures are preserved, not resubmitted.

Live status: `results/c4_unseen_rules_v1_feedsllm_20260906.status.json`.
Submission/quota/IDs: `results/c4_unseen_rules_v1_feedsllm_20260906.submission.json`.
Complete submit log: `results/c4_unseen_rules_v1_feedsllm_20260906.submission.log`.
Expected final summary: `results/c4_unseen_rules_v1_feedsllm_20260906.summary.json`.
Scientific protocol: `reports/c4_unseen_rules_protocol_20260906.md`.

## Completion monitoring (16:03 UTC)

Both exact submitted jobs were verified Running by AMLT and Azure ML. User
logs now show the frozen-panel/hash preflight passing and the 61 GiB base
copy progressing on the remote nodes. Early missing-log/temporary HTTP fetch
errors were observation issues, not evidence of terminal jobs; no resubmission
or checkpoint modification was performed.

Existing collector/AMLT-watch handles 221995/222002 remain live. Finalizer
handle 228857 waits for the collector, then runs
`scripts/audit_c4_unseen_rules_delivery.py` and writes
`reports/c4_unseen_rules_results_20260906.md` plus
`results/c4_unseen_rules_v1_feedsllm_20260906.delivery_audit.json`.
The finalizer checks that the collector is still live while waiting; a terminal
receipt or missing process triggers attention, never automatic resubmission.

Additional local audit/report code does not change any frozen remote source.
It checks returned source hashes against the submitted snapshot, source-data
lineage, exact prompt token counts, independent reference answers and runtime
packages, then includes all conditions/categories, per-rule breakdowns,
pair-level transitions, all-three outcomes, no-op mismatches and representative
success/failure text. Nine relevant protocol/report tests passed, followed by
three report/finalizer tests after adding live-handle checking. Frozen remote
source checksums still pass. No performance claim is made before generation
and final audit complete.

## First runtime evidence (16:08 UTC)

Both exact jobs have produced two complete groups (54 generations each).
Remote runtime receipts agree on torch 2.8.0+cu126, transformers 4.57.6,
peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.14.0 and safetensors 0.8.0;
both report NVIDIA A100-SXM4-80GB. Model load took about 38 seconds after the
61 GiB copy. First group times were A 77.7/77.1 seconds and B 81.8/80.8
seconds. Peak allocated/reserved VRAM was 29.79/30.40 GiB. Thus the local
17–18 GiB free would not have supported the unchanged endpoint.

Linear extrapolation from two groups suggests approximately 40 minutes for
the remaining 30 groups, excluding interruptions and final collection. This
is a runtime estimate, not performance evidence or a completion guarantee.
Both remote jobs and the collector/native-watch/finalizer processes remain
live. Goal remains active until both complete and all output checks pass.

## Complete output audit (16:58 UTC)

Both arms have produced all 32 groups, 864 records each, plus matching
`complete.json` and merged-record hashes. Independent collection and the full
delivery audit passed on all 1,728 records. The official monitor still reports
both jobs Running while platform teardown finishes; output completion and
platform terminal status are kept separate. No extra GPU run was submitted.

`reports/c4_unseen_rules_results_20260906.md` now contains complete results,
all nine conditions, pair-cluster intervals, per-rule counts and observed
success/failure examples. `results/c4_unseen_rules_v1_feedsllm_20260906.delivery_audit.json`
contains the independent audit. The native terminal monitor and finalizer
remain in place to repeat canonical collection after both jobs pass.

Primary outcomes are identical for A/B: offset swap 0/24 versus wrong-rule
0/24; composition 1/24 versus 0/24; conditional 14/24 versus 13/24; seen
control 24/24 versus 0/24. No constructed category has a specificity interval
strictly above zero; none has a complete three-query swap success (0/8 per
category per arm). Seen controls have 8/8 complete three-query successes.
Offset oracle succeeds on 19/24 in both arms while open history is 0/24,
suggesting a history-induction limitation in this setup; composition oracle
is only 1/24, so execution/instruction-following also remains a limitation.
These diagnostics do not uniquely identify the failure cause.

No-op discrepancies: A 0/96, B 1/96. The B discrepancy is offset:000:0,
query 0: native 104, own-state replacement 208. Both miss the donor and
recipient new-rule targets; the case is retained. All 1,728 outputs parse
as valid numeric responses, so invalid-format filtering does not explain
the negative new-rule endpoint. This is one training seed and four independent
history pairs/category, not a general impossibility result.

The local report renderer was updated to state measured negative categories,
oracle/history diagnostics and no-op discrepancies explicitly; remote frozen
code and the scientific protocol were not modified. Eleven protocol/delivery
tests passed before this reporting edit, and all four delivery/report tests
passed afterward.

## Platform completion (16:59 UTC)

AMLT now confirms both exact submitted jobs as `pass`: A
`red_kitten_cgvxbp4rhc`, B `purple_kettle_psfk0nmvl5`. The existing collector
completed at 16:59:18 UTC and wrote the canonical summary after downloading
both terminal outputs. It agrees byte-for-byte with the independently audited
complete-output snapshot. No retry experiment, additional training or
outcome-based exclusion was used in this unseen-rule evaluation.
