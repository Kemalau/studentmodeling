# C4 history-length diagnosis (exploratory, inference only)

## 1. Question and non-goals

Does unseen-rule failure improve with more unique observations? Locate patterns
consistent with induction, bottleneck or transport failure without asserting
unique causal localization. Frozen final A/B, one training seed 42, no updates,
checkpoint search, recurrent state, or spontaneous-rule invention claim.

## 2. Sample construction and concrete sample

Reuse all 16 unordered history pairs, both directions, three queries/direction
from the audited September 6 unseen-rule panel: four categories, four pairs
each, six constructed rules and two known controls. Keep original questions,
targets, first three histories, templates, and checkpoint hashes unchanged.
These are 96 directional queries (48 unique questions) per history cohort.
This is a post-hoc diagnostic remeasurement, NOT fresh held-out replication.

Four cohorts: original 3 examples; nested 6; nested 12; original 3 repeated four
times. Seven conditions/cohort, 672 generations per shard, four shards/model,
5,376 outputs across A/B. Repeat is an approximate context-length/repetition
control, not exact token matching. All 12-example histories include both
denominator parities. Additional trials are letter-labelled from D onward to
avoid exposing small target answers in trial numbers; first three labels stay
byte-identical. Prompts retain the original 768-token bound including 48 output
tokens, with no truncation. The initial h12 tokenizer preflight peaks at 630
input tokens; the model context and evaluator length guard are unchanged.

Generator seed 2026090701. Add nine examples/history from the same denominator
range 4–40 and three templates. Exclude old CF1/MalRule/train/validation/test
number signatures, prior execution panels and every prior unseen-panel
history/query; added examples are globally distinct. Exclude all current query
targets from added history numeric strings, including correct historical answers.
Reverse directions reuse the same histories. Nested examples are NOT newly
sampled for each length. No model-output filtering. Rule identifiability remains
only within the eight declared functions, not arbitrary functions/pretraining.

Concrete existing offset-3 donor: `36/40 × 39/40 → 1407`,
`16/37 × 12/37 → 195`, `6/16 × 15/16 → 93`. Fixed queries require 211,
516 and 18. The 6/12 cohorts extend this history; they do not change targets.

## 3. Coordinate, objective and formula

Capture all 64 decoder-block output residuals at the vision-pad token, source
prefix without current query/target. No optimization or coordinate ranking.
For category c and history cohort k,
`D(c,k) = mean_i [1(y_swap,i = donor_i) - 1(y_random,i = donor_i)]`.
Primary diagnostic contrasts are paired `h12-h3` and `h12-repeat3x4` on open
history, donor-native and D, separately per category and checkpoint. h6 is
dose-response. Accuracy means exact rational agreement with synthetic rule
targets, not fidelity to real students.

## 4. Arithmetic sign example

Illustrative: 12/24 swap matches and 3/24 random matches give D=+0.375.
If D is 0.125 with three histories, the length contrast is +0.250. These
numbers illustrate scoring and are not measured follow-up results.

## 5. Fixed checkpoints and measured motivating example

A final step240: `7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6`.
B final step240: `87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32`.
There is no ranked neural candidate. The original offset-3 query
`16/20 × 13/20` expected 211; both A/B swapped states generated 208,
wrong-rule states generated 208, explicit-rule oracle generated 211.
That observation motivates this diagnostic; it does not select test examples.

## 6. Selection funnel

All four original categories → all four pairs/category → all eight directions
→ same three queries → four history cohorts. Source, numeric and independent
reference-answer filters run before inference. No dropped difficult category,
seed, output or pair. All eight original labelled functions remain in diagnostics.

## 7. Held-out causal controls

`native`: recipient history; `donor_native`: donor history behind bottleneck;
`noop`: independently captured own state; `swap`: donor state; `random`:
independent recipient-rule history state, not Gaussian noise; `oracle`:
explicit donor rule with empty history; `open_history`: ordinary causal
attention to donor history. All except oracle omit current rule, answer and menu.
Oracle prompt is identical across cohorts, providing a runtime invariance check.
Same-donor-history control is omitted in this cheaper diagnosis; original data
retains it. Original h3 conditions rerun on the same runtime as longer histories.

## 8. Intervention and dose

`h_recipient^l ← s_donor^l`, l=0..63, full replacement, dose 1, prefill only.
One captured source trajectory per history is reused across three questions.
Fresh KV per generation. Corrected late-fusion attention and answer anchor
are unchanged. Longer history also moves the state-token position and exceeds
the three-example repair-training format, so a negative length effect cannot
uniquely prove insufficient state capacity.

## 9. Decoding and evaluation

Same NF4/bfloat16 kbit-preparation, eager attention, frozen rank8, dropout off,
greedy 48-token generation. Strict single tagged/bare number, exact rational
equality; invalid, multi-answer and capped outputs count as misses. No LLM judge.
Whole-triplet rule-signature diagnostics distinguish fallback to learned
`a*c` or `a*c/d` from a genuinely transported conditional rule.

## 10. Aggregate and paired reporting

Per category/checkpoint/cohort: every condition hit count and denominator,
D, all-three success, invalids and every no-op mismatch. Pair-cluster
bootstrap 10,000 resamples; same original unordered pair keeps both directions,
three queries and compared cohorts together. Intervals are exploratory and
unadjusted, with only four independent pairs/category. No new replication gate.
Bootstrap helper uses its existing fixed RNG seed 2026090604. Zero-width
empirical intervals do not establish zero population uncertainty.

## 11. Successes, failures and corruption

Record every raw text/token sequence and prompt hash. Display first original
group per category across all cohorts regardless of success. Report old-rule
whole-triplet matches and inconsistent outputs separately. Preserve no-op errors.

## 12. Mechanism boundary and falsifiers

Unique history improvement beyond repetition supports information-dose limits
on this fixed panel. Oracle success + persistent open-history failure is
consistent with induction limits; poor oracle leaves execution/instruction
following unresolved. Open-history success + donor-native failure suggests a
bottleneck/attention-format limit. Donor-native success + swap failure suggests
transport limits. These are diagnostic patterns, not unique causal proofs.
Known controls cannot count toward unseen-rule success. No universal executor,
recurrent state or educational-human fidelity claim follows.

## 13. Reproduction and execution

Builder `scripts/c4_history_diagnostic.py`; frozen manifest
`manifests/c4_history_diagnostic_v1_20260907.json`; evaluator
`scripts/evaluate_c4_history_diagnostic.py`; merge
`scripts/collect_c4_history_diagnostic.py`. Each checkpoint/cohort is one
resumable Feeds-LLM Standard 80G1-A100 job using the accepted fixed project
image and explicit rankfun_aml UAI. Existing remote base/final adapters only;
all outputs go to AMLT_OUTPUT_DIR on the existing mounted project store.
Local work is generation, hashing, token-length validation and tests only.
Runtime/manifest/adapter/code hashes must agree before a complete result.
