# C4 prompt-support diagnosis: audited results

## 1. Question and non-goals

Which prompt support restores the original seed-42 checkpoint's donor-state
effect on the same new-number panel? Candidate visibility restores a large
effect; merely switching free generation to teacher-forced scoring does not.
This is exploratory diagnosis, not an independent replication or a test of
recurrent state updating.

## 2. Sample construction and concrete sample

The parent frozen panel has 27 donor histories, 81 queries (three per donor),
54 subtraction, 21 fraction, six decimal questions. Novelty was checked against
CF1 train/valid 6,000/600 rows and original MalRule train/valid/same-test/cross-test
9,000/900/900/900 rows; these file counts are not unique disjoint counts.
Numeric combinations and target answers in the visible histories were excluded
as specified in the parent manifest. Pretraining novelty is unknown.

Keep all histories, queries, source states and checkpoint fixed. Toggle visible
candidate list and current correct-answer feedback independently, giving four
formats. Every cell evaluates the same three A/B/correct target strings; scoring
still teacher-forces these targets when they are hidden from the prompt.

Example: `6029 - 1341`, A=4678, B=4388, correct=4688. Only the format determines
whether a shuffled list of these numbers and/or a line giving 4688 is visible.

## 3. Coordinate and objective

Capture all 64 decoder-block output residuals at the one vision-pad address
from the instruction/history prefix only, shape 64 x 5120. No query or future
answer enters this capture. No optimization or layer selection is performed.

`D(f,m) = mean_q [I(y_swap(f,m,q)=A(q)) - I(y_random(f,m,q)=A(q))]`.

Here f is prompt format, m is readout, A(q) is the synthetic donor-rule target.
This measures directed donor agreement, not arbitrary output flips.

## 4. Arithmetic sign example

One swap-only A match contributes +1/81, random-only contributes -1/81. For
candidates-only generation, (61-29)/81=0.39506. This is not the historical
clean-native-B 0.411 flip rate, which used another denominator/panel/control.

## 5. Real measured observation

Candidates-only free generation produces A on 61/81 directed queries, 29/81
random, and 16/81 native queries. Neither-support generation produces A on
3/81 directed and 3/81 random. Both comparisons use the same 81 questions.

## 6. Selection funnel

All 27 parent donors -> 81 queries -> four formats -> five state conditions ->
two readouts = 3,240 records. Three donor-disjoint shards, 1,080 records each.
No outcome-based removal. Invalid outputs remain failures. Selection is not
independent of the prior exploratory C4 pipeline.

## 7. Held-out causal validation

Five conditions: native, own-state no-op, donor A, real random, and another
history with recipient B's rule. The inherited random pool is family matched;
its rule is executable on only 33/81 queries, so applicability is a limitation
of this prior panel, not assumed from family matching. The next fixed-menu
experiment uses semantically checked random rules instead.

## 8. Steering definition

At prefill, overwrite `h_B[layer,state] <- h_source[layer,state]` at every block.
Full replacement, no dose search. Late-fusion masking hides history from later
positions and hides the state from query positions; generated answer positions
can read the state. Fresh KV per condition/query. Histories remain physically
present: this is not the strict history-free bridge.

## 9. Evaluation tasks and rubric

Qwen3-32B, frozen CF1 vision-pad seed-42 adapter, NF4/bfloat16, eager attention.
Greedy generation, 48-token maximum, no answer prefill or constrained vocabulary.
Stop at EOS or closing response tag. Strict single-number/tag parser with exact
rational equality. Scoring sums complete tagged-answer token log probabilities
and chooses the maximum over three candidates. No LLM judge; no prompt truncation.

## 10. Aggregate and paired results

All counts below have denominator 81. Intervals are 10,000 paired bootstrap
draws over the 27 donor groups, preserving all within-donor questions/conditions.
Marginal exploratory intervals are not multiplicity adjusted.

| Visible support | Readout | Directed A | Random A | D | 95% donor-cluster CI |
|---|---|---:|---:|---:|---|
| Candidates + correct | Score | 67 | 34 | 0.4074 | [0.2469, 0.5679] |
| Candidates + correct | Generate | 67 | 34 | 0.4074 | [0.2469, 0.5679] |
| Candidates only | Score | 60 | 29 | 0.3827 | [0.1975, 0.5556] |
| Candidates only | Generate | 61 | 29 | 0.3951 | [0.2222, 0.5556] |
| Correct only | Score | 26 | 13 | 0.1605 | [0.0741, 0.2716] |
| Correct only | Generate | 11 | 6 | 0.0617 | [0.0123, 0.1235] |
| Neither | Score | 7 | 4 | 0.0370 | [-0.0123, 0.0864] |
| Neither | Generate | 3 | 3 | 0.0000 | [-0.0370, 0.0370] |

Adding candidates without current correct feedback changes generation D by
+0.3951, paired CI [0.2346, 0.5556]. Adding correct feedback when candidates
already exist changes D by +0.0123, CI [-0.1481, 0.1852]; lack of a detectable
increment is not an equivalence proof.

The subsequent semantic audit flags four parent donor groups (12 queries):
six use a numerator-comparison rule on subtraction, six decimal recipient
targets disagree with the independent textual algorithm. Keep the original
table unchanged. **Post-hoc** restriction to the remaining 23 donors/69 queries:

| Visible support | Generated directed A | Random A | D | 95% donor-cluster CI |
|---|---:|---:|---:|---|
| Candidates + correct | 55/69 | 27/69 | 0.4058 | [0.2174, 0.5797] |
| Candidates only | 49/69 | 22/69 | 0.3913 | [0.2029, 0.5652] |
| Correct only | 6/69 | 3/69 | 0.0435 | [0.0000, 0.0870] |
| Neither | 1/69 | 1/69 | 0.0000 | [-0.0435, 0.0435] |

This check preserves the qualitative support dependence; it is not a replacement
preregistered endpoint or a fresh confirmation set.

## 11. Failures and numerical diagnostics

With neither support, native B agreement is only 5/81, whereas native
mathematical correctness is 40/81. The model is not reliably simulating its
recipient's error pattern in this unsupported format. Directed output has
13/81 strict parse failures, random 10/81. The parent mixed-number sensitivity
still gives no directed-minus-random gain; see its preserved report.

No-op is not universally text-identical in these quantized prefix-capture
experiments. On two prior discrepancies, full-prompt capture restores native
cached output. Prefix vs full-prompt state differences are already present
early and reach max absolute differences 11.5 and 16 at layer 63. On one query,
prefix-noop full recomputation still differs; on the other it restores native.
These observations implicate capture/decode numerical consistency, but do not
isolate one kernel as the cause. Do not import the old no-op=0 claim as exact
text identity for this protocol.

## 12. Mechanism conclusion and limitations

On this checkpoint/panel, causal transfer is strongly supported when answer
options are visible and is not established in unsupported free generation.
Visible candidates may support copying, rule-conditioned verification, or
other question-dependent choice mechanisms. The factorial isolates support
visibility, not an internal algorithm, and does not prove stored answers either.

The ongoing explicit-rule and fixed-menu two-query follow-ups test execution
elicitation and constant-answer/position explanations, respectively. Their
results must be reported separately. A positive fixed-menu result would still
be candidate-supported behavior, not an autonomous executable rule.

## 13. Reproduction and lineage

Frozen protocol: `reports/c4_format_diagnostic_protocol_20260905.md`.
Original raw outputs: `results/downloads/c4-format-diagnostic-seed42-v1-20260905`.
Original summary: `results/c4_format_diagnostic_seed42_20260905.summary.json`.
New read-only audit: `scripts/audit_c4_format_semantic_subset.py`; output
`results/c4_format_diagnostic_semantic_valid_20260905.audit.json`.
All 3,240 unique records, prompt hashes, per-donor/shard agreement, saved flags,
score argmax, generated token-to-text decoding, code and tokenizer hashes were
checked. Reaggregating reproduces the original summary exactly (excluding only
the added experiment-name field).

```bash
cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
/home/v-yuejiali/DeepGraph/.venv/bin/python scripts/audit_c4_format_semantic_subset.py \
  --output results/c4_format_diagnostic_semantic_valid_20260905.audit.json
```

Original experiment: `c4-format-diagnostic-seed42-v1-20260905`, Feeds, three
Standard 80G1-A100 jobs, UAI `rankfun_aml`, image
`amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`.
Remote model `/mnt/amulet/studentmodeling/models/Qwen3-32B`, adapter
`/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter`.
These completed outputs are distinct from the currently running v2 follow-up.
