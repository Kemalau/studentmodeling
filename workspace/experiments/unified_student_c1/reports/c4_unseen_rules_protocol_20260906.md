# C4 unseen-rule pilot: Feeds-LLM execution authorized

## 1. Question and non-goals

Can the frozen final A (matched candidate-free SFT) and B (state-exchange)
checkpoints infer and transport newly constructed rules from three observations?
No further training, recurrent state, unseen-pretraining claim, or invented-rule
claim. This test was designed after the previous six-rule results, and is a new
exploratory follow-up, not a retroactive confirmatory test.

## 2. Sample construction

Restrict to common-denominator fraction multiplication, where the prior two
checkpoints had the clearest effect. Keep the prior operand range (denominators
4–40, positive proper written fractions) and three text templates. New rules:

| Category | Two rules on `(a/d) × (c/d)` |
|---|---|
| Fixed offset | `a*c + 3`; `a*c + 7` |
| Composition | `(a+1)*(c+1)/d`; `(a+2)*(c+2)/d` |
| Conditional routing | even d: `a*c`, odd d: `a*c/d`; reversed mapping |
| Seen-rule control | `a*c`; `a*c/d` |

Four independent unordered history pairs per category, two directions each,
three test queries per direction: 16 pairs, 32 directions, 96 directional
queries (48 unique questions). Nine conditions give 864 generations/checkpoint,
1,728 total. Every pair has four distinct histories: two rule identities and
one independently sampled same-rule history for each. Each history has three
observations. Both denominator parities appear in every history and query triplet.

Generator seed 2026090603. Exclude unordered canonical operand/rendered-number
signatures from CF1 train/valid (6,000/600), MalRule source train/valid/tests
(9,000/900/900/900), all previous execution-pilot splits, and the original
81-query numerical panel. Exclude duplicate combinations across new pairs and
within each pair's histories/queries; opposite directions intentionally share
the same pair. Targets differ from the other rule, correct answer, visible-history
numbers, and the other query targets. No model-output-dependent filtering.

The early CF1 checkpoint saw 22 labelled categories including NONE, not only
the recent six repair rules. Their inventory and source hashes are in the
manifest. New rules are absent **as labelled mechanisms in those audited
fine-tuning sources**. Label audit is not a proof of semantic/pretraining absence:
conditional rules deliberately compose two seen primitives, and incidental
individual examples can agree with new rules. This scope must remain explicit.

Finite-set identifiability: each history and three-query answer signature
uniquely distinguish its rule among the eight declared rules. This is not
identifiability among arbitrary functions or every possible old misconception.

Concrete constructed item (expected target, not observed output): the offset-3
student history includes `36/40 × 39/40 → 1407`, `16/37 × 12/37 → 195`,
`6/16 × 15/16 → 93`. Held-out queries expect `16/20 × 13/20 → 211`,
`27/39 × 19/39 → 516`, and `5/16 × 3/16 → 18`.

## 3. Coordinate and objective

Reuse the prior full 64-block residual trajectory at the vision-pad token.
Evaluation has no loss minimization, optimizer, adapter update, or rule search.
Primary per-category specificity:
`D = mean_i[1(y_swap,i = target_donor,i) - 1(y_random,i = target_donor,i)]`.
Compare `D_B - D_A` on identical pairs. Report categories separately, and never
pool known controls into an unseen-rule success rate.

## 4. Arithmetic sign example

Illustration only: 12/24 directed hits and 3/24 wrong-rule hits give
`D = (12-3)/24 = 0.375`. This is not a measured model result.

## 5. Fixed checkpoints

A SHA256 `7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6`;
B SHA256 `87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32`.
Both are final step 240, training seed 42. Paths copied from the completed
recovery receipt into the manifest; remote accessibility must be rechecked
before submission. No checkpoint selection based on this new test.

## 6. Selection funnel

Six constructed rules + two fixed known controls → four pairs/category →
outcome-free number/collision/finite-rule identifiability filters → all 32
directional groups retained. Current builder: 331 seen-number rejections,
3 copy/repeated-answer rejections, 1,920 independent answer checks across eight
rules on 240 generated problems. Reference uses expanded integer formulas and
gcd reduction rather than the primary Fraction-based implementation.

## 7. Held-out controls

- `native`: recipient history through the bottleneck.
- `donor_native`: donor history through the bottleneck.
- `noop`: recipient history with independently captured own-state replacement.
- `swap`: recipient history with donor trajectory.
- `random`: a distinct history of the recipient rule, not Gaussian noise.
- `same_donor`: a distinct history of the donor rule.
- `oracle`: explicit donor rule, empty history, no answer/candidates.
- `open_history`: donor history with ordinary causal attention, no bottleneck mask.
- `empty_history`: no rule and no history, bottleneck decoder otherwise unchanged.

Open-history changes only the attention path relative to donor-native text;
it diagnoses history interpretation without forcing it through state. If oracle
fails, execution/instruction following is a limiting factor. Oracle passes but
open-history fails suggests induction/identification difficulty. Open-history
passes but donor-native fails suggests the bottleneck; donor-native passes but
swap fails suggests transport. These are diagnostic patterns, not unique causal
proofs, especially with only four independent pairs/category.

## 8. Intervention

`h_recipient^l ← s_donor^l` after every decoder block l=0..63 at the same
state-token address, dose 1, prefill only. Capture once per history, reuse for
three questions. Fresh KV for each generation. Use the previous corrected
late-fusion mask and answer anchor for every condition except open-history.
No claim of a history-free bridge or last-layer comparison in this test.

## 9. Decoding and scoring

Same NF4/bfloat16 base and kbit preparation as the prior evaluation, eager
attention, rank-8 adapter, dropout off, eval mode. Greedy, 48 output tokens,
EOS/closing-tag stopping, no truncation of input (768-token bound including
output allowance). Strict single numeric output/tag, exact rational equality.
Invalid/multiple/truncated outputs are misses. No LLM judge. No current target,
correct answer, rule description, or menu in primary prompts. Oracle is separately
labelled. The output metadata is never supplied to the generation renderer.

## 10. Reporting

Report all conditions per category, all-three donor agreement, all no-op
token/stop discrepancies, invalid counts, and paired B-minus-A specificity.
Bootstrap 10,000 times over unordered pair IDs, seed 2026090604. Four independent
pairs per category, not 24 independent queries; intervals are exploratory,
unadjusted, and weak evidence for broad population claims. Report both raw
counts and intervals regardless of sign. For conditional rules, matching one
branch is not enough: require the same captured state to match all three queries,
covering both branches. Do not rename a partial branch success as a new rule.

## 11. Failures and corruption

No inference results yet. Preserve all groups, raw text/tokens, parsed output,
stop reason, prompt hashes, state hashes, and checkpoint/source lineage.
Collector rejects duplicate/missing records, wrong prompts/targets, invalid
fingerprints, group/merged disagreement, and token/text disagreement.
Keep no-op mismatches in metrics and report them; previous A/B had 1/96 each.

## 12. Scope and falsifiers

Success would support transferring a newly specified rule within this fraction
task, not a general program interpreter or spontaneous rule invention. Failure
is an interpretable boundary and must be reported, not followed by selecting
easier rules on the same test. Any redesign becomes a new exploratory version.
The baseline old checkpoint is not part of this two-checkpoint comparison;
this test cannot by itself attribute new-rule ability to the repair training.

## 13. Reproduction and current status

Builder: `scripts/c4_unseen_rules.py`; manifest:
`manifests/c4_unseen_rules_v1_20260906.json`; evaluator:
`scripts/evaluate_c4_unseen_rules.py`; collector:
`scripts/collect_c4_unseen_rules.py`; tests:
`tests/test_c4_unseen_rules.py`. All paths are relative to this experiment root.

Initially prepared locally only. On 2026-09-06 the user explicitly requested
local execution; the local waiting queue below supersedes the initial
Feeds-only execution suggestion. No new remote GPU job was submitted and no
checkpoint was modified.
Five CPU tests passed in 10.35 seconds: reference formula examples; source/new
number isolation and counts; finite-set history/query-rule identifiability;
prompt separation and tokenizer length bounds; synthetic collector corruption
rejection and paired contrasts; and open-history versus bottleneck mask behavior
(several checks share one test). Python compilation also passed. These are
implementation checks, not model-performance evidence.
### Local execution amendment (2026-09-06)

The host has one A100 PCIe 80 GB, but the existing unrelated process uses about
64 GB, leaving about 17–18 GB. The data disk has roughly 695 GB free; disk
capacity is not the limiting factor. The 62 GB base model is already local at
`/home/v-yuejiali/studentmodeling/models/Qwen3-32B` (a data-disk symlink).
Only the two final adapters (513 MB total) were fetched from the completed
training jobs; both SHA256 values match section 5. The first download pattern
was rejected by azcopy for a nontrailing wildcard; the two exact per-arm
paths succeeded. No training intermediate or optimizer state was downloaded.

Local queue: `scripts/run_c4_unseen_rules_local.py`, using
`/mnt/codex-tmp/c4-followup-runtime/bin/python`. It waits for two consecutive
GPU-idle checks and at least 45,000 MiB free, never kills another process, and
runs one frozen group to measure memory/time before resuming all A groups and
then B. No outcome-dependent stop or rule filtering. Per-arm group records are
resumable; the existing collector audits all 1,728 records before marking
completion. A 24-hour resource wait or a four-hour child execution timeout
marks `needs_attention` and preserves completed group files; neither means a
scientific failure. GPU availability is checked again before each subprocess.

This execution amendment adds only bounded preflight/runtime logging and a
resource guard to the evaluator. It does not change the manifest, prompts,
decoding, weights, quantization, conditions, or scoring. Seven CPU tests passed
after the changes; compilation passed. Runtime: torch 2.8.0+cu126,
transformers 4.57.6, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.14.0.
The evaluator records runtime/package metadata and hashes of its code.

All local run artifacts, queue status and logs are under
`/datadisk/v-yuejiali/studentmodeling/c4_unseen_rules_20260906/`;
`status.json` is the live execution receipt. `A_preflight.runtime.json` will
contain actual throughput after GPU access; `summary.json` is created only
after both complete outputs pass the collector. No new-rule performance
result or wall-time estimate is available while waiting for GPU access.

### Feeds-LLM routing amendment (2026-09-06)

The user subsequently requested Feeds-LLM. The local supervisor PID 209733
was stopped while still waiting (no model inference had started). Local
checkpoint copies are preserved; the unrelated GPU job was not stopped.
The frozen scientific protocol above is unchanged.

Derive `amlt/feedsllm_c4_unseen_rules_v1_20260906.yaml` from the accepted
`feeds_c4_execution_training_v1_recovery1_20260906.yaml`, preserving workspace
`CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity`, target Feeds-LLM,
image `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`,
UAI `rankfun_aml`, and the existing `csnewsandfeeds4150361735/amulet` mount.
Two independent Standard `80G1-A100` jobs, one per checkpoint, run
`scripts/run_c4_unseen_rules_feeds.sh A` / `B`. Each has a four-hour runtime
limit and the inherited two-attempt infrastructure retry setting. No
outcome-driven retraining or automatic new experiment submission.

Base input is `/mnt/amulet/studentmodeling/models/Qwen3-32B`, copied to the
compute node's `/tmp/Qwen3-32B` only after a 120 GiB free-disk check. Final
adapter input paths remain exactly those in the manifest. Only allowlisted
source, config and the small manifest are uploaded. The runner verifies source,
prompt, formula and adapter hashes, and all outputs remain under each job's
`AMLT_OUTPUT_DIR/unseen_A` or `unseen_B`, with runtime/runner logs beside them.

Experiment name: `c4-unseen-rules-v1-seed42-feedsllm-20260906`; job names
`c4-unseen-rules-arm-a-seed42` and `c4-unseen-rules-arm-b-seed42`. Native
terminal-only monitoring and exact paired collection are implemented by
`scripts/watch_c4_unseen_rules_feeds.py`; its authoritative status receipt is
`results/c4_unseen_rules_v1_feedsllm_20260906.status.json`. Submission receipt
and current quota are stored separately; the protocol file itself is frozen
before submission. A queued job is not a completed experiment.
