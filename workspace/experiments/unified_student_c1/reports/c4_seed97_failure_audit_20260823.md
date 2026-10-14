# C4 CF1 + vision-pad seed-97 failure audit

Date: 2026-08-23

## 1. Question and non-goals

This audit asks why the final CF1 + `<|vision_pad|>` checkpoint labelled seed
97 has `D_specific = 0.015`, whereas the corresponding seed-13 and seed-42
checkpoints have `0.188` and `0.221`.  The operational outcome is directional
response control under a frozen state-swap intervention.  This audit does not
establish a recurrent/Markov student state, a compact final-layer vector, or a
real-student causal effect.

## 2. Sample construction

All three model checkpoints were trained on the same 6,000-row CF1 Stage-A
dataset: 3,000 original Malrule response rows plus one same-family
counterfactually recombined row per original.  The train and validation SHA256
digests were respectively
`929487329576607572aa49b8e2daba53605b3c733749e7e631baa376a9df65f9`
and `4bf7117db760bcaba9790b8f7901ae89428911359f2b483f08aef39dc1fe6a36`.

Crucially, seeds 13, 42, and 97 were evaluated on exactly the same frozen
150-pair manifest, despite the different model-training seed names.  The
manifest is `c4_stage_a_gate_pairs_seed42_n150_v1.json`, SHA256
`edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8`.
Direct comparison of the downloaded raw swap records confirms that all 150
donor, recipient, same-rule, label, target, and family fields are identical
across the three evaluations.  Therefore the seed-97 collapse cannot be
attributed to drawing a different or unusually colliding pair set.

For each pair, the recipient prompt supplies the query.  Its native state is
the unpatched condition; the directed intervention replaces the residual at
the single state-token address with the donor trajectory across all 64 decoder
blocks.  Random-state, same-malrule, and no-op interventions are matched
controls.  One concrete pair has donor misconception
`fractions.ignore_denominators`, recipient misconception
`fractions.keep_common_denominator_for_multiplication`, donor response `7`,
recipient response `7/10`, and correct response `7/100`.

## 3. Coordinate, score, and exact contrast

The intervened coordinate is the residual stream at the `<|vision_pad|>` token
position after each of decoder blocks 0--63.  For recipient pair `i`, define

```text
m_i(c) = score_i(answer_A | c) - score_i(answer_B | c)
delta_i(c) = m_i(c) - m_i(no intervention)
```

where `c` is directed swap, random-state, same-rule, or no-op.  The primary
pre-registered discrete contrast is

```text
D_specific = mean_i 1[pred_i(swap) = answer_A]
           - mean_i 1[pred_i(random) = answer_A]
```

on the clean subset whose unpatched prediction is `answer_B`.  A continuous
diagnostic is `mean_i(delta_i(swap) - delta_i(random))`.

## 4. Arithmetic sign example

Illustratively, if a clean recipient has native margin `-3`, directed swap
moves it to `+2`, and random state moves it to `-1`, then directed and random
margin shifts are `+5` and `+2`; the specificity shift is `+3`.  A positive
value means the matched donor state moves the answer toward the donor more than
an arbitrary state.

## 5. Real pair example

In seed 97, pair 83 is one of only three clean directed flips.  Its recipient
answer is `1/14`, donor answer is `equal`, and correct answer is `1/10`.
The native donor-minus-recipient margin is `-0.151`; directed swap moves it to
`+0.063` (a `+0.215` shift), but no-op and same-rule also flip this borderline
example.  It therefore does not provide clean evidence of a large, stable
donor-specific mechanism by itself.

## 6. Selection funnel

No neurons or layers were selected from this evaluation.  The full fixed
trajectory (64 blocks) and all 150 preregistered pairs were evaluated.  The
clean filter is outcome-defined independently per checkpoint:

```text
150 frozen pairs -> unpatched prediction equals recipient answer -> clean set
```

This yields 101, 95, and 66 clean pairs for seeds 13, 42, and 97.  Because this
changing denominator could confound comparisons, this audit also recomputes
all results on the 53 pairs that are clean under every checkpoint.

## 7. Held-out causal validation

| training seed | final validation loss | cross-template response accuracy | best state probe | clean n | swap flip | random flip | same-rule flip | no-op flip | D_specific |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.0147 | 0.920 | 0.913 | 101 | 0.455 | 0.267 | 0.059 | 0.030 | 0.188 |
| 42 | 0.0158 | 0.907 | 0.933 | 95 | 0.411 | 0.189 | 0.042 | 0.000 | 0.221 |
| 97 | 0.0710 | 0.643 | 0.692 | 66 | 0.045 | 0.030 | 0.015 | 0.045 | 0.015 |

On the common 53-pair clean set, the discrete `D_specific` values are 0.302,
0.340, and 0.019 for seeds 13, 42, and 97.  Paired 20,000-replicate bootstrap
95% intervals are `[0.170, 0.434]`, `[0.208, 0.472]`, and
`[-0.038, 0.094]`.  Thus the seed-97 failure remains when the pair identities
and clean denominator are held fixed.

The same pattern is stronger in the continuous margins.  On each checkpoint's
own clean subset, mean directed/random margin shifts are `5.076/2.594`,
`4.723/2.264`, and `0.009/0.021`; hence continuous specificity is 2.482,
2.460, and -0.012.  On the common clean subset it is 3.394, 3.479, and -0.013.
Pairwise directed margin effects correlate at `r = 0.926` between seeds 13 and
42 but only `r = -0.023/-0.019` between seed 97 and seeds 13/42.

## 8. Intervention definition and dose semantics

The directed treatment copies the donor state-token residual into the
recipient at all 64 decoder blocks.  Random replaces it using the frozen
random donor specified by the evaluator; same-rule uses another state with the
recipient malrule; no-op patches a native-equivalent state.  This audit studies
the fixed full-replacement dose only; it does not infer a dose-response curve.

## 9. Evaluation task and rubric

The task is closed-candidate `malrule_next_response`.  Every condition is
scored against the same candidate responses.  `flip_to_A` means the top-ranked
candidate after intervention is the donor response; `stay_B` means it remains
the recipient response; all other top candidates are `other`.  The clean set
requires the unpatched model to select `answer_B`.

## 10. Aggregate and paired results

Seed 97 differs before intervention, not merely after it: unpatched recipient
accuracy on the 150 pairs is 44% (`clean_n=66`) versus 67.3% and 63.3% for
seeds 13 and 42.  Its intervention conditions are nearly indistinguishable:
mean absolute changes in the two target scores are 0.094 (swap), 0.093
(random), 0.086 (same-rule), and 0.085 (no-op).  For seeds 13/42, directed swap
changes those scores by 5.112/4.755 on average, versus no-op changes of only
0.081/0.089.  Seed 97 therefore learned a weak or inactive state route rather
than a route that fails only on a special family or subset.

The training logs corroborate this.  Seeds 13 and 42 undergo a sharp loss drop
near the middle of the epoch and finish at validation loss 0.0147/0.0158.
Seed 97 plateaus near 0.06--0.08 and finishes at 0.0710.  This is a general
optimization basin failure: response accuracy, state decodability, and causal
control fail together.

## 11. Representative success and failure

- Seed-97 pair 83 is a rare borderline success (`1/14 -> equal`) but also
  flips under no-op/same-rule, so it is not strong donor-specific evidence.
- On pair 1 (`7/10` recipient versus `7` donor), directed swap moves the
  seed-97 margin in the wrong direction (`-1.061 -> -1.186`), while the
  successful checkpoints show very large positive shifts on the same frozen
  item.
- Across subtraction, seed 97 has identical directed and random flip rates
  (`0.034` each) on its clean rows.  The collapse is not concentrated in one
  identified malrule.

## 12. Mechanism conclusion and limitations

The initial hypothesis that seed 97 sampled unusually difficult or colliding
malrule pairs is falsified: all three evaluations used identical pairs, and a
common-clean reanalysis preserves the failure.  The supported diagnosis is a
training-trajectory failure in which the model does not learn to transmit
student-specific information through the state address.

There is also a reproducibility defect in the training script.  The model and
LoRA modules were instantiated before Hugging Face `Trainer.__init__`, while
the script supplied the seed only through `TrainingArguments`.  In
Transformers 4.57.6, `Trainer` calls `set_seed` inside its constructor, so the
LoRA initialization occurred before the requested seed was applied.  Thus the
historical labels 13/42/97 reliably changed the subsequent training sampler but
did not fully control adapter initialization.  This does not invalidate the
three observed checkpoints, but it means they are not cleanly reproducible
training-seed replications.  The code has now been corrected to call
`transformers.set_seed(args.seed)` before model/LoRA construction.

The next confirmatory experiment should rerun at least three fresh seeds after
this fix, with the pair manifest and data hashes unchanged, and checkpoint at
250-step intervals.  Predeclare selection by validation loss plus held-out C4,
and report every seed.  This would falsify the current optimization-basin
account if fixed-seed reruns converge equally well yet seed 97-like causal
failures remain.

## 13. Reproduction files and commands

- Training configuration: `amlt/feeds_state_alias_factorial_20260729.yaml`
- Trainer: `scripts/train_c1_lora.py`
- Swap evaluator: `scripts/state_swap_misconception_smoke.py`
- Frozen pairs: `manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json`
- Aggregate historical results:
  `results/state_alias_factorial_monitor_20260729/summary.json`
- Raw seed-level swap outputs were fetched with:

```bash
AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-studentmodeling-feeds-v2 \
  amlt results download state-alias-factorial-7job-20260729 \
  :state-cf1-visionpad-seed97 -I 'state_swap_masked.json' -o <output-dir>
```
