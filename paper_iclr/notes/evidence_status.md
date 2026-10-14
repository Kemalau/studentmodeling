# Evidence and Claim Status

This file is a guardrail for drafting. It should be updated whenever the formal
matrix changes.

## New method status

- The earlier promoted recurrent trajectory is diagnostic evidence, not the
  current final method. Under the stricter State8 curriculum gate, every base
  checkpoint failed at least one task-retention or C4 requirement. The active
  path is therefore causal-teacher, early-label, and interleaved-retention
  repair selection; no replacement candidate has yet passed the global gate.
  Exact-epoch final fit and full final evaluation remain downstream and have
  not started from a qualified replacement. Do not report candidate or
  intermediate-checkpoint metrics as final-model results until the exact epoch,
  full frozen 21,676-row evaluation, final C4, and 1,494-row external T3
  judgment all pass.
- On the selected round-4250 recurrent candidate, independent C4 pair seeds
  13/42/97 give `D_specific=0.195/0.165/0.165` (mean `0.175`, minimum `0.165`),
  and the two-of-three replication gate passes.  This establishes pair-sample
  stability for that checkpoint, not training-seed stability and not the final
  exact-epoch result.
- A diagnostic-only 50-pair audit of the promoted exact-resume round-1,000
  checkpoint gives swap/random/same-rule/no-op
  `0.556/0.296/0.222/0.000`, hence `D_specific=0.259` on `clean_n=27`.
  Directional specificity is positive, but the run fails the full gate because
  clean support is below 50 and the same-rule control exceeds 0.10. A frozen
  150-pair replication over pair seeds 13/42/97 is running; neither result is
  a substitute for final-checkpoint C4.

- The recurrent Causally Grounded Generative Student Model (`CGSM`) has now
  been implemented and exercised in small M2/M3 diagnostics.  The completed
  scalar recurrent variants fail the frozen C4 causal gate, so implementation
  is not evidence that the proposed state is causally closed.
- Existing full-data Qwen results belong to the unrestricted mixed QLoRA model.
- Existing probe and swap results belong to the one-shot late-fusion Malrule
  model, not to recurrent CGSM.
- The completed M2/M3 jobs are evidence that the recurrent interface executes,
  not that it preserves the one-shot causal variable.  In the frozen C4 audit,
  M2 CF1 initialization and M2 full-task decoding each give
  `D_specific=0.000`, while M3 generation+KD gives `-0.039`.  The completed CF1
  positive result remains a one-shot counterfactual-recombination result; it is
  not evidence that a recurrent counterfactual objective works.
- The abstract intentionally describes the method and evaluation plan without
  numerical performance claims. Add numbers only after CGSM evaluation.

## Ready to state

- Under task taxonomy v2, one shared conditional-generation schema covers
  headline T1--T5: correctness, response, misconception diagnosis,
  prospective behavior, and strict error persistence. T6 is the legacy T4
  future-KC-performance proxy and remains appendix-only.
- The appendix T6 future-KC-performance baseline has now been rebuilt on 2,939
  student-disjoint DBE+XES test rows. A class-balanced ordinal PFA reaches
  macro-F1 0.439, balanced accuracy 0.487, and quadratic weighted kappa 0.260;
  the future outcomes used to construct the label are not model features. This
  is an appendix baseline and is not part of the 21,676-row headline matrix.
  A 10,000-replicate student-cluster bootstrap gives 95% intervals
  [0.420, 0.458], [0.463, 0.511], and [0.226, 0.294], respectively; the paired
  balanced-accuracy gain over source majority is +0.153 [0.129, 0.178].
- The current full training mixture contains 259,845 examples after merging the
  available validation rows and filtering to 1,536 tokens.
- The formal test manifest contains 21,676 examples across six sources.
- The formal evaluation uses open generation for FoundationalASSIST response
  tasks and a full-taxonomy or open protocol for Eedi misconception diagnosis.
- In the fixed 150-pair CF1 + `<|vision_pad|>` seed-42 gate, response accuracy
  is 0.890 same-template and 0.907 cross-template, and the state-position probe
  reaches 0.933 cross-template accuracy at the best layer.
- In that fixed gate, the clean directed swap reaches 0.411 `flip_to_A`, versus
  0.189 random, 0.042 same-rule, and 0.000 no-op. Thus the primary
  causal-specificity contrast is 0.221 in this single seed.
- The matched newly added, mean-initialized row-sparse token is a negative
  anchor: directed swap, random, same-rule, and no-op each reach only 0.018 on
  its clean subset. Decodability without controlled causal use is not treated
  as a successful state.
- In the completed factorial, CF1 + vision-pad passes the preregistered causal
  gate in two of three seeds (`D_specific=0.188/0.221/0.015` for seeds
  13/42/97). Original-only + vision-pad fails in all three
  (`-0.018/-0.017/0.000`), while seed-42 image-pad and video-pad also fail
  (`-0.024/0.014`). This establishes an observed curriculum--initialization--seed
  interaction, not token invariance.
- In a separate matched seed-42 checkpoint-curve run, neither CF1 nor
  original-only passes at any of steps 250/500/750/1000/1250/1500.  CF1 gives
  `D_specific=-0.016/-0.061/-0.111/0.020/0.000/0.023`; original-only gives
  `-0.054/-0.054/-0.042/-0.017/-0.056/0.018`.  This is negative evidence
  against a simple early-formation/late-decay story and shows recipe/trajectory
  sensitivity that must be resolved before making a robust formation claim.
- A zero-training clone from the vision-pad input/output rows to a newly
  registered `<STUDENT_STATE>` gives exactly zero hidden-state and logit error
  on three checked examples before and after artifact reload. Thus the tested
  behavior is portable to a dedicated token address and does not depend on the
  surface token name.
- A zero-training, strict history-free layerwise bridge preserves nearly the
  full one-shot C4 effect while using only the controlled state address and no
  persistent history KV cache.  On `clean_n=97`, swap/random/same-rule/no-op are
  `0.402/0.196/0.072/0.000`, so `D_specific=0.206`.  In contrast, transporting
  only the final layer-63 vector gives no changes at all on `clean_n=61`.  The
  causal representation is therefore localized to the token's depth-indexed
  residual trajectory, not its final-layer vector.  Because this state has
  shape `64 x 5120`, it is one address across depth but not yet a compact single
  vector or a recurrently closed world state.

## Preliminary only

- Unified Qwen3-32B currently improves over the best traditional specialist
  suite on the primary metric for T1--T5 in one formal run, but T3 is not an
  identical-score cross-paradigm comparison and the same-backbone task-only
  controls are still pending.
- The five one-epoch same-backbone task-only controls are exact target-task
  exposure matches to the unified 259,845-row final-fit epoch: for every task,
  the unique example-ID set and the `task`/`target`/`target_label` supervision
  signatures are identical. This is a verified design property, not a result;
  adapter completion, full frozen evaluation, common TRAPI-v2 judgment, and
  paired comparisons are still pending.
- The current margins are +0.015 T1 accuracy, +0.054 T2 mathematical
  equivalence, +0.063 T3 semantic accuracy, +0.006 T4 joint mathematical
  equivalence, and +0.002 T5 accuracy.
- T3 is not yet an apples-to-apples model comparison: LLM generations receive
  blinded semantic judgments, while taxonomy specialists receive exact class
  accuracy.
- The completed exposure-matched C3 experiment does not support the transfer
  hypothesis: the misconception arm does not improve the real-Eedi semantic
  endpoint over the best control, and no arm preserves every non-T3 gate. This
  is a one-seed negative result, not evidence of positive cross-task transfer.
- C4 results are strong controlled pilots, not yet a large-scale real-student
  intervention result.
- The older exploratory layer sweeps reported 0.494 masked and 0.301
  open-attention `flip_to_A`. They are retained as exploratory evidence; the
  fixed-pair, controlled 0.411/0.189/0.042/0.000 gate above is the primary
  estimate for replication.
- The old late-fusion pilot motivates CGSM, but comparison against it is a
  required ablation rather than evidence that CGSM works.

## Do not claim yet

- Do not claim state of the art.
- Do not claim that the unrestricted unified model beats same-backbone
  task-only Qwen controls until the five one-epoch adapters, frozen full-test
  evaluations, common T3-v2 rejudgment, and paired comparisons complete.
- Do not claim that all traditional models are theoretically unable to produce
  responses; say that standard KT models do not natively expose these targets
  without task-specific extensions.
- Do not call T6 latent mastery. Its label is a future-KC-performance band
  (legacy label T4).
- Do not call XES repeat incorrectness misconception persistence.
- Do not describe FoundationalASSIST T5 as within-session repeated attempts. It
  is same-question later-exposure exact-response recurrence.
- Do not report a six-task macro accuracy.
- Do not imply cold-item generalization from the current mostly warm-item test.
- Do not present 32-candidate Eedi ranking as open misconception diagnosis.

## Required for the current CGSM submission

If the recurrent gates fail, these requirements must be rewritten for a
narrower causal-state paper rather than presented as completed CGSM evidence.

- At least three seeds for the unified model and critical specialist baselines.
- Confidence intervals and paired significance tests where examples align.
- Exposure-matched C3 ablations on the full Eedi test set.
- T3 human audit and agreement statistics for the semantic judge.
- T5 results separated by prediction distance `delta = 1, 3, 5`.
- Natural-prevalence and cold-item T1 analyses, or an explicit limitation if
  they cannot be completed.
- More swap pairs and at least one additional seed for C4.
- Implement CGSM and compare it with full-context and one-shot late-fusion
  models under matched data, steps, and adapter capacity.
- Measure the state-compression gap, recurrent drift by trajectory length, and
  incremental-inference cost.
