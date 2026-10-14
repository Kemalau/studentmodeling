# C4 execution-training pilot: execution receipt

## Prepared and verified

Two matched rank-8 continuation arms, 240 updates each; training manifest is
frozen separately from prior experiments. Each arm has 1,920 rows and **26,524
supervised target/EOS tokens**, in identical target order. Actual longest
two-row sequence: A 233, B 230 tokens; no truncation. Six known arithmetic rules,
not the full T1-T5 data. Candidate-free primary evaluation uses 16 independent
test pairs (96 directional queries), ten conditions per checkpoint.

Local CPU tests: **4 passed in 20.58s** using PyTorch 2.8.0+cu126,
Transformers 4.57.6, PEFT 0.19.1. These check split isolation, matching target
exposure, true source gradients with checkpointing, equivalence between batched
exchange and separate capture/patch, and collector corruption rejection.
Thirteen frozen source/protocol/data hashes pass. AMLT config `--dump` passed.
Remote base and initial adapter Blob objects exist and are readable.

## Authentication and compute

Expected user/tenant/subscription validated using existing cache
`/home/v-yuejiali/.azure-studentmodeling-borrowed-keyflip-20260903`.
Management and storage tokens refreshed without printing tokens or modifying
global caches. No device-code login.

Feeds A100 snapshot: Standard 73/97; Premium 0/0; Basic 0/0; user_max 32, actual
personal usage not exposed by the quota row. The conservative project request
audit reports 42 requests including alternate-target and hyperdrive parent
entries, so it does not justify assuming two free personal slots. No unrelated
job was stopped. Feeds-LLM snapshot Standard 3/80, Basic 0/80, Premium 140/144,
user_max null; alternate target awaits explicit user approval.

Approved immutable project image and UAI preserved in config. Intended run:
`c4-execution-training-v1-seed42-20260906`, two Standard `80G1-A100` jobs,
each max 8 hours/attempt and two attempts. Initial adapter and base remain
remote project inputs. Source package is about 2.7 MB, no model weights.

## Live actions and status

A one-backward/no-optimizer-update 32B diagnostic is running locally using
already-present weights and the hash-verified initial adapter. It must verify
nonzero source-history gradient and finite loss before cluster submission.
This is not a trained checkpoint or an experiment result. It runs separately
from another user's local process and does not stop or alter that process.

`scripts/watch_c4_execution_training.py` was launched as a durable supervisor
(initial PID 2772658). It waits for the backward diagnostic, validates hashes,
checks current Feeds quota under the project submission lock, submits once
only if at least two conservative slots are available, uses native terminal-
notification watch, downloads results and runs the paired collector. Ambiguous
submission/failures stop with `needs_attention`, never duplicate jobs.

As of this receipt, **not yet submitted and no optimizer updates performed**.
Authoritative live state: `results/c4_execution_training_v1_20260906.status.json`.
Logs: same prefix `.log`; eventual summary: `.summary.json`.
Diagnostic: `results/c4_execution_training_20260906_backward_diagnostic/diagnostic.json`.
The supervisor does not silently switch to Feeds-LLM.

## Update at 2026-09-06 03:37 UTC

The local 32B backward-only diagnostic completed successfully with **no optimizer
update**: loss 0.2022589594, parameter gradient norm 0.1480428755, source-row
embedding gradient norm 0.0387195833, peak allocated VRAM 29.7876 GiB. The
backward stage took 3.13 seconds; loading 17 local weight shards took 12m34s.
The GPU was released after the diagnostic. This is engineering evidence, not
training improvement, and the source-row norm includes the state and instruction
positions; history-specific gradient isolation is covered by the tiny-model test.

Provenance caveat: this process started before the manifest gained additional
history-identifiability metadata and before the non-truncation assertion and
resume-validation guard were added to source. Its original lineage is preserved,
not rewritten. The 1,920 training targets have the same ordered SHA256
`a6d3a2049010a18cd7b376dd573f87aaf6b8b190ae3c035934fc42b1ce5ac558`,
the same 26,524 supervised tokens, and the same initial adapter. No exchange
or loss operation changed in those edits. All current source hashes pass and
the latest CPU suite reports **6 passed in 25.95s**.

The quota audit was corrected in `scripts/c4_feeds_quota.py` without modifying
the shared legacy counter: fresh status plus HyperDrive-child refresh, filter
to Feeds, exclude orchestration parents, deduplicate job IDs, include queued
workers, and fail closed for unreadable/unknown rows. Two unit tests cover these
rules. The latest authoritative snapshot still has **32/32 personal Feeds A100
requests**. Pool Standard usage is 71/97; Premium and Basic are both 0/0.
Other experiments continue unchanged; no alternate cluster was authorized.

The supervisor was restarted as PID **2820126** to use that corrected audit.
It still waits for two free Feeds slots, then submits the two named arms once
and uses native AMLT completion notifications. Only JSON/JSONL artifacts will
be retrieved for collection; adapters and optimizer checkpoints stay remote to
avoid exhausting the nearly full local root disk. Live status remains the
authoritative source; **no submission receipt or trained result exists yet**.

## User-approved Feeds-LLM submission, 2026-09-06 03:42 UTC

The user explicitly approved Feeds-LLM. The Feeds-only waiting supervisor was
stopped before its first submission attempt; refreshed project listing showed
no experiment with the intended name. No running training or unrelated job was
cancelled. Only the execution target changed, not the frozen experiment data,
training objective, budget, checkpoint selection, or evaluation protocol.

Submitted experiment: **c4-execution-training-v1-seed42-20260906**.

| Arm | Job ID | Verified state at 03:42 UTC | GPU |
|---|---|---|---|
| A: candidate-free SFT | zen_hamster_g6h9sg7lzk | queued | 1 × 80G1-A100 |
| B: differentiable exchange | khaki_cart_v8pv4cg1c0 | queued | 1 × 80G1-A100 |

Both use Standard SLA on Feeds-LLM through the original
CS-NewsAndFeeds-Singularity workspace and primary subscription. The final
pre-submit quota was Standard 3/80, Basic 0/80, Premium 140/144; no personal cap
was exposed. Three existing personal Feeds-LLM requests were audited. A null
personal cap is explicitly preserved, not replaced by a claim of 80 personal
slots. The controller requires both an auditable request count and two Standard
pool slots. Four quota/controller tests passed; config `--dump` and all frozen
source hashes passed. Updated config SHA256:
`f1069b6c8dc942bdcc5786c35de5ae075e7febf6f9b20c037aa8234bcc8dc13b`.

Approved fixed image remains
`amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`;
explicit UAI remains `rankfun_aml`. Remote model and initial adapter paths remain
those in the frozen protocol. Each job writes its `execution_A`/`execution_B`
directory under `AMLT_OUTPUT_DIR` on `csnewsandfeeds4150361735/amulet` storage.
Fourteen package files (2.6 MB) were uploaded; no base weights were uploaded.

Authentication reused the validated
`/home/v-yuejiali/.azure-studentmodeling-borrowed-keyflip-20260903` cache. Direct
Azure CLI target-subscription lookup could not resolve that subscription in the
cache; the authoritative AMLT target query exposed Feeds-LLM under WPP, and
successful submission plus queued job status verified workspace-to-target
access without changing the active subscription or account.

Durable supervisor PID **2836999** is now running native `amlt watch` child PID
**2843098**, terminal-only notifications. After successful jobs, it retrieves
JSON/JSONL and computes the paired result automatically. Failures stop with
`needs_attention` and no blind duplicate submission. As of this update, both
jobs are **queued, not confirmed training**, and no scientific result exists.

Verified remote result prefixes from `amlt show`, under storage account
`csnewsandfeeds4150361735`, container `amulet`:

- A: `projects/v-yuejiali/amlt-results/7211333900.27038-f9e25890-92ca/execution_A`
- B: `projects/v-yuejiali/amlt-results/7211333900.26940-f41d8bbc-0130/execution_B`

`amlt show` also resolves target subscription to WPP and workspace subscription
to the original primary subscription, confirming the intended cross-subscription
route. The job UAI resolves to the project's `rankfun_aml` resource.
