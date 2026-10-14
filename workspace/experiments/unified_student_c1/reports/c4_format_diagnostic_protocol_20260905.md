# C4 prompt-support and readout diagnosis (frozen before new inference)

## 1. Question and non-goals

Diagnose the null open-generation result on the fixed 81-query seed-42 panel.
Keep checkpoint, histories, query numbers and donor identities fixed. Manipulate
only current-query candidates and correct-answer feedback, then compare scoring
and free generation. No training, seed search, recurrent upgrade or new gate.
This is an exploratory follow-up informed by the observed free-generation null,
not a preregistered independent replication of the original C4 claim.

## 2. Sample construction and concrete example

Reuse every query from `c4_freegen_novel_seed42_20260905.json`: 27 donor groups,
three queries each, 11 rules (54 subtraction, 21 fractions, six decimal
comparisons). No new exclusion or outcome-based selection. Parent hashes and
all original fields are retained in the new manifest. See the parent protocol
for source split counts, numerical novelty filters and the two excluded donors.

Each query has exactly three evaluation candidates: its distinct A-rule answer,
B-rule answer and mathematical answer. Candidate order is independently shuffled
with the fixed string seed `20260905:group_idx:query_idx`, once per query, and
then held constant across state conditions and formats. Candidate strings use
canonical rational values inside `<predicted_response>` tags. No rule identity
labels are shown. These are new three-answer lists on the novel panel, not the
identical historical candidate sets; comparisons to the old 150-pair experiment
are contextual, not a controlled estimate of numerical novelty alone.

Example (actual fixed panel group 2, query 1): `Calculate: 6029 - 1341`;
A=4678, B=4388, correct=4688. The previous native, directed and random generations
were all 4688. The candidate list contains the three tagged numbers in its
frozen shuffled order. The correct-answer line, when present, says 4688.

## 3. Coordinate, objective and formula

Capture all 64 decoder-block output residuals at the single `<|vision_pad|>`
address (64 x 5120), from instruction/history prefix only, with no current or
future query. Capture once per source identity on each shard; reuse the same
tensors across all questions and formats and log hashes. No discovery, gradient
objective, coordinate ranking or top-K selection occurs.

For each format f and readout m,
`D(f,m) = mean_q(1[pred_swap(f,m,q)=A(q)] - 1[pred_random(f,m,q)=A(q)])`.
Also report native B agreement, mathematical correctness, all five condition
rates, per-family rates, native-B subsets, applicable-random-rule subsets,
unseen-answer subsets, no-op disagreements and three-of-three donor success.

## 4. Arithmetic sign example

A swap-only A hit contributes +1/81 to D; a random-only A hit contributes -1/81.
The contrast `D(both,m)-D(correct_only,m)` measures the change associated with
adding candidates while correct feedback stays present. It does not establish
that the state implements arithmetic independently of the supplied answers.

## 5. Real prior observation motivating diagnosis

The previous full panel had directed=random=3/81, native B=5/81 and native
mathematical correctness=40/81. On the concrete 6029-1341 query, own-state no-op
changed 4688 to 4708. These are prior observations, not results of this follow-up.

## 6. Selection funnel

All 27 prior donors -> all 81 prior queries -> four prompt formats -> five state
conditions -> two readouts = 3,240 endpoint records. No generated output is
removed; invalid free generations count as failures. Three group-disjoint shards
use group_idx modulo 3; each produces 1,080 records (540 generations and 540
three-candidate scoring evaluations). There are no outcome-selected groups.

## 7. Causal validation / format matrix

| Format | Candidates visible | Current correct answer visible |
|---|---|---|
| both | yes | yes |
| candidates_only | yes | no |
| correct_only | no | yes |
| neither | no | no |

All formats retain original instruction, history and historical feedback.
The neither prompt is byte-for-byte identical to the previous experiment.
Each format uses native, own-state no-op, directed A, real-random and
same-recipient-rule controls. Random donors are unchanged; their rule applies
to only 33 of 81 queries, so that subset is reported separately.

## 8. Intervention and numerical checks

Full replacement at each block output: `h_B[layer,state] <- h_source[layer,state]`.
Reuse original late-fusion mask. During cached generation patches are applied
at prefill; downstream-layer state K/V carries the effect to generated tokens.
Each condition/query starts a fresh KV cache; original histories remain masked
but physically present. This is not the history-free experiment.

Separately audit the two previously observed no-op mismatches, (2,1) and (22,0):
compare per-layer prefix/full-prompt state values, native cached/full recompute,
prefix-noop cached/full recompute, and full-prompt-capture no-op cached decoding.
These six extra generations are explicitly post-hoc technical diagnostics and
are excluded from the factorial endpoints. Tiny-model cached/full recomputation
equivalence tests run locally and at remote startup; they do not by themselves
guarantee exact equality in the 32B quantized model.

## 9. Scoring, generation and rubric

Qwen3-32B with the frozen CF1 vision-pad seed-42 adapter, NF4 4-bit/bfloat16,
eager attention, original helpers. No retraining or adapter selection.
Scoring uses sum log probability of each complete tagged candidate, as in the
old C4 evaluator; select the maximum. Save scores and candidate-normalized
probabilities. Even when candidates are hidden from the prompt, this scoring
mode evaluates the same three strings via teacher forcing: it is not free
generation or full-vocabulary search.

Generation is greedy, 48-token maximum, no answer prefill, no constrained
vocabulary, stop at EOS or closing tag. Reject overlength prompts rather than
truncate. Retain raw text/tokens/stop reason. Frozen strict parser accepts a
single numeric tag or bare scalar; exact rational equality. No LLM judge.
Presence of the candidate list can enable copying; any gain there is evidence
of support dependence, not independent rule execution.

## 10. Aggregate and paired contrasts

Recompute all eight cells on the same 81 queries. Bootstrap 10,000 times over
27 donor clusters, preserving the three within-donor queries and all conditions.
Report native B and D differences for both-minus-candidates_only,
both-minus-correct_only, candidates_only-minus-neither, correct_only-minus-neither,
and score-minus-freegen within each format. Intervals are marginal, exploratory,
not multiplicity-adjusted; do not retrofit a pass gate from these comparisons.
Audit all 3,240 unique keys, raw parsing, scores, target flags, prompt hashes,
group-to-shard allocation and shared model/code fingerprints before summarizing.

## 11. Successes, failures and corruptions

Pending. Show matched examples across all four formats and readouts, including
negative or invalid outputs and no-op failures. Do not select only successes.

## 12. Interpretation and falsifiers

Recovery after restoring a prompt field supports dependence on that field in
this panel. Positive scoring but weak generation with identical prompts locates
a scoring/generation gap. Continued failure with both fields restored leaves
novel-number generalization, changed candidate construction, panel composition
and technical transport differences as alternatives; it does not isolate any
one of them. Null results do not prove exact population zero or answer memory.
All results are single-checkpoint diagnostics, not cross-seed generalization.

## 13. Reproduction and execution

- Manifest: `manifests/c4_format_diagnostic_seed42_20260905.json`.
- Protocol/scorer: `scripts/c4_format_diagnostic.py`.
- Evaluator: `scripts/evaluate_c4_format_diagnostic.py`.
- Tests: `tests/test_c4_format_diagnostic.py`, `tests/test_c4_freegen_decode.py`.
- Config derives from accepted `amlt/feeds_c4_freegen_novel_seed42_20260905.yaml`;
  generator: `scripts/prepare_c4_format_diagnostic.py`.
- New config: `amlt/feeds_c4_format_diagnostic_seed42_20260905.yaml`.
- Target Feeds / CS-NewsAndFeeds-Singularity workspace, 3 x Standard 80G1-A100;
  UAI `rankfun_aml`; image `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`.
- Base model: `/mnt/amulet/studentmodeling/models/Qwen3-32B`.
- Adapter: `/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter`.
- Output: each job's `AMLT_OUTPUT_DIR/format_diagnostic`; atomic donor-group
  checkpoints and lineage-checked resume. Upload only explicit small-file allowlist.
- Experiment intended name: `c4-format-diagnostic-seed42-v1-20260905`.
- Submission, job IDs, startup tests and completion will be recorded after observed.

## Submission receipt (2026-09-05 12:15 UTC)

Submission command completed with exit code 0. Immediate backend status: all
three jobs queued (accepted, not completed). Job IDs:

- shard0: `jolly_kiwi_62swz71jrg`
- shard1: `jovial_fig_42jtwnw4df`
- shard2: `loving_chin_qvwn83j8w9`

Last pre-submit A100 quota: Standard 58/97; Premium 0/0; Basic 0/0;
user_max limit 32 (live per-user usage not exposed by that quota endpoint).
Five currently active experiments examined separately requested 23 GPUs,
including a 4-GPU job and nine queued one-GPU jobs; scheduling remains dynamic.
No existing jobs were stopped or altered. Existing project browser-login cache
was reused without copying credentials or initiating device authentication.

Preflight: 13 tests passed; full 324-prompt token audit 213–380 tokens with
48-token generation budget; no truncation; all 12 frozen file hashes passed;
AMLT parsed config passed; exact 13-file upload allowlist checked (~537 KB).
Remote code package:
`projects/v-yuejiali/amlt-code/884d2fad26077c971fd6cded5d97fef0678c4c32bad41e513d1758524cbcf905`.

Background read-only monitor started at 12:15:54 UTC (PID 573455), invoking
native `amlt watch` every 60 seconds with terminal-only notifications. It
downloads after completion and runs full-record/lineage audit plus paired
summary; it never resubmits. Receipts:
`results/c4_format_diagnostic_seed42_20260905.monitor.json`, `.watch.json`,
`.monitor.log`; final summary destination `.summary.json`.
No new experiment outcomes were available when this receipt was recorded.
