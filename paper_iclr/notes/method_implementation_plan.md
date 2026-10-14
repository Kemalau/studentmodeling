# Student World Model / CGSM Experiment Plan

Updated: 2026-07-31

This is the execution plan for turning the current one-shot late-fusion pilot
into one checkpoint that supports C1--C4 and, if the transition experiments
pass, a recurrent student world model.  Every stage has an explicit exit gate.
Failing a gate triggers diagnosis or a narrower claim; it does not automatically
trigger a larger sweep.

## Executive decision

Run two nested claim tracks, with the first one protected from failure of the
second:

1. **Causal-state track:** establish that one fixed-capacity state address is
   sufficient, readable, and specifically causal under held-out
   recombinations.  This track requires M0, M1, the one-shot C1--C4 comparison,
   and multi-seed C4 replication; it does not require recurrent updates.
2. **World-state track:** establish that the state can be updated without raw
   history, remains predictive under long rollouts, and supports useful action
   selection.  This track begins at M2 and earns the stronger world-model claim
   only after transition closure and drift gates pass.

The central variable is the hidden state at a controlled address, not the
surface spelling of the token.  A pretrained multimodal token may be used as an
initialization anchor.  If M1 establishes zero-training equivalence, the paper
uses a dedicated `<STUDENT_STATE>` name and reports the copied initialization
as an implementation detail and ablation.

## Live execution board

Snapshot: 2026-07-31 19:36 UTC.

| Priority | Stage | State | Immediate action | Exit artifact |
|---:|---|---|---|---|
| P0 | M0 alias/curriculum factorial | Completed | Retain all seven valid rows and exclude the three superseded infrastructure failures | Seven-row summary with raw C4 controls and seed aggregation |
| P0 | M1 zero-training clone | Engineering equivalence passed | Package the cloned row/tokenizer with the selected adapter; no retraining | Exact zero-error hidden/logit equivalence before and after reload |
| P0 | Matched checkpoint curves | Completed; causal gate failed | Retain as a negative matched trajectory and audit why it differs from the successful CF1 anchor | Six-checkpoint original-only and CF1 curves |
| P1 | One-shot C1--C4 bridge | Layerwise C4 passed; task screen queued | Finish the strict history-free T1/T2/T3/T5/T6 screen and compare with the full-context checkpoint | Per-task compression gaps plus passed C4 bridge |
| P1 | M2 recurrent wrapper | Scalar failed; layerwise zero-training bridge passed | Finish the remaining layer/position controls, then train the layerwise update only if task sufficiency passes | One-address transition diagnosis with no raw-history route |
| P2 | M3 objective ablation | Initial generation+KD arm evaluated; C4 gate failed | Do not scale this arm; finish only the already-submitted CF4 specificity screens | Sequential objective diagnosis, not a grid |
| P2 | M4 unified recurrent checkpoint | Not started | Single seed first; three seeds only after the pilot gate | C1--C4 multi-seed table |
| P3 | M5/M6 paper completion | Not started | Run only for claims that survive M4 | Robustness, efficiency, capacity, and planning tables |

The M0 experiments `state-alias-factorial-7job-20260729` and
`state-alias-factorial-original-retry1-20260729` are complete.  CF1 +
vision-pad passes the causal gate for seeds 13 and 42 but fails for seed 97;
original-only + vision-pad fails for all three seeds, and the seed-42 image-pad
and video-pad controls fail.  The three original-only failures in the first
experiment occurred before model loading and are superseded by the successful
retry.  The matched dynamics experiment
`state-visionpad-checkpoint-causal-curve-20260729` is also complete: both
one-GPU jobs passed, but no retained checkpoint passes the causal gate.  At
steps 250/500/750/1000/1250/1500, CF1 has `D_specific =
-0.016/-0.061/-0.111/0.020/0.000/0.023`, while original-only has
`-0.054/-0.054/-0.042/-0.017/-0.056/0.018`.  This rules out a simple monotone
"causality forms early and is later forgotten" account for this matched
1500-step recipe and exposes strong trajectory sensitivity relative to the
successful CF1 anchor.

The scalar recurrent M2/M3 transition audit
`recurrent-m2-m3-c4-transition-seed42-20260731` is complete.  All three jobs
passed operationally but failed the scientific C4 gate: M2 initialized from
the CF1 adapter has `D_specific=0.000`, M2 decoded with the full-task adapter
has `0.000`, and M3 generation+KD has `-0.039`.  Therefore the current scalar
update/readout does not preserve the one-shot causal variable.  Zero-training
layerwise transport, one-hidden-vector layer sweeps, strict history-free formal
screens, and CF4 donor/specificity objectives are submitted as bounded
diagnostics; they are not yet positive recurrent evidence.

A newly completed zero-training compact layerwise bridge gives the missing
mechanistic localization.  It captures the CF1 state token at the same single
address after each of the 64 decoder blocks and injects that native-coordinate
trajectory into a strict history-free decoder.  On the frozen 150-pair C4 gate
(`clean_n=97`), it reaches swap/random/same-rule/no-op
`0.402/0.196/0.072/0.000`, hence `D_specific=0.206`, and passes every gate.
The otherwise matched final-layer-63 single-vector bridge gives all four rates
`0.000` on `clean_n=61`.  Thus the positive one-shot causal variable is not
preserved by collapsing the state address to its final-layer vector; it is
carried by the depth-indexed residual trajectory.  This is one address but 64
native 5120-dimensional vectors, so it must not be described as a single
5120-dimensional latent or as a completed recurrent world state.

## Hypothesis ladder

Each rung answers one question and is required before spending compute on the
next claim:

1. **Existence:** can a forced one-token route retain task performance?
2. **Causal use:** does a matched donor state change the response more than
   random, same-rule, and no-op interventions?
3. **Attribution:** is the effect due to curriculum, token initialization, or
   their interaction, and does it replicate across seeds?
4. **Symbol independence:** can the learned function be moved to a dedicated
   token by exact embedding-row cloning with no training?
5. **Transition closure:** can `z_t` plus one interaction produce `z_{t+1}`
   without raw history or a hidden KV-cache route?
6. **Long-horizon utility:** does the recurrent state remain calibrated and
   useful for prediction and action selection as updates accumulate?

Failure at rungs 1--4 narrows the result to a token-sensitive one-shot pilot.
Failure at rung 5 preserves a causal bottleneck result but rules out a recurrent
world-state claim.  Only rung 6 supports the planning extension.

## 0. Current evidence and claim boundary

### Completed evidence

- The unrestricted unified Qwen3-32B model covers T1, T2, T3, T5, and strict
  T6.  In the current single formal run it is competitive with the assembled
  specialist suite, but this is not yet a multi-seed result.
- The hard state-route model has a complete small formal evaluation.  The
  pasted row `.610 .457 .004 .717 .840` means T1/T2/T3/T5/T6 =
  `0.610/0.457/0.004/0.717/0.840`; its near-zero T3 result is an important
  state-compression failure, not a formatting failure.
- The one-shot late-fusion Malrule pilot demonstrates a readable and partially
  causal state.  The strongest completed CF1 + `<|vision_pad|>` seed-42 gate
  has response accuracy `0.890` same-template and `0.907` cross-template,
  state-position cross-template probe accuracy `0.933`, and clean directed
  swap `flip_to_A=0.411` versus random `0.189`, same-rule `0.042`, and no-op
  `0.000`.
- The otherwise matched newly added, mean-initialized row-sparse state token
  does not establish causal use: clean swap, random, same-rule, and no-op all
  have `flip_to_A=0.018`.  Decodability alone is therefore insufficient.
- In the completed M0 factorial, CF1 + vision-pad has `D_specific` values
  `0.188/0.221/0.015` for seeds `13/42/97`, passing the preregistered two-of-three
  replication rule.  Original-only + vision-pad has
  `-0.018/-0.017/0.000`; seed-42 image-pad and video-pad have `-0.024/0.014`.
- Copying the vision-pad input and output embedding rows to a newly registered
  `<STUDENT_STATE>` with no optimizer step produces exactly zero maximum hidden
  state and logit error on three checked examples, including after saving and
  reloading the tokenizer and row artifact.

### In progress

- The compact strict history-free layerwise-state C4 bridge has passed.  Its
  aligned-position, cross-adapter, recurrent-update, and remaining single-layer
  controls are running or queued.
- A zero-training T1/T2/T3/T5/T6 layerwise formal screen is queued with the CF1
  adapter as state encoder and the complete-task adapter as decoder.  The
  decoder receives no raw history and no persistent history KV cache.
- CF4 donor-directional and specificity-triplet screens are running or queued
  from the completed M2 checkpoint.  They are diagnostics of the objective,
  not authorization to claim recurrent causal closure.

### Claim boundary

- The completed results support a one-shot, fixed-capacity causal bottleneck
  pilot.
- Recurrent code and small diagnostics now exist, but the completed scalar M2
  and M3 C4 audits do **not** support a causally closed/Markov state.  They also
  do not support long-horizon rollout planning.
- A special token is a state address and initialization choice, not the claim.
  The claim depends on exclusivity, sufficiency, transition closure, and causal
  behavior under controlled interventions.

## 1. Frozen metrics and gates

All model-selection gates use validation students or the frozen Malrule gate.
The formal test remains untouched until the selected configuration is fixed.

### Predictive metrics

- T1: accuracy and AUROC, by source and natural/balanced prevalence.
- T2: mathematical equivalence, separated into closed-choice and open response.
- T3: full-taxonomy accuracy/MRR or blinded semantic accuracy; never merge the
  two protocols as if they were identical.
- T5: joint correctness + mathematical equivalence, separately for
  `delta=1/3/5`.
- Strict T6 remains a secondary exact-response recurrence analysis.

### State metrics

- `D_specific = flip_to_A(swap) - flip_to_A(random)` on the frozen clean subset.
- `D_damage = flip_to_A(random) - flip_to_A(noop)`.
- No-op and same-rule controls, changed top-1, probability/margin deltas, and
  clean-subset size are always reported with raw swap.
- Cross-template state-position probe accuracy is reported with the
  history-last control and chance level.

### Initial go/no-go thresholds

These thresholds are preregistered engineering gates, not final statistical
claims:

- response same-template and cross-template accuracy >= `0.80`;
- state-position cross-template probe accuracy >= `0.85`;
- `D_specific >= 0.15`;
- no-op `flip_to_A <= 0.05` and same-rule `flip_to_A <= 0.10`;
- for replication, the mean directed effect must be positive and at least two
  of three seeds must individually pass the causal gate; final reporting uses
  paired bootstrap confidence intervals rather than the threshold alone.

## M0. Close the current token/curriculum factorial

**Question.** Is the completed seed-42 causal effect driven by counterfactual
curriculum, seed luck, or a particular pretrained token embedding?

### Conditions

1. CF1 + vision-pad, seeds 13/42/97 (42 is the completed anchor).
2. Original-only + vision-pad, seeds 13/42/97.
3. CF1 + image-pad and video-pad, seed 42.
4. Completed mean-initialized dedicated-token seed 42 as a negative anchor.

All conditions use the same 100% late-fusion mask, effective example exposure,
step count, LoRA rank, pair manifest, and C4 gate.

### Decision table

- CF1 passes and original-only fails: counterfactual recombination is necessary.
- Both pass: late-fusion plus pretrained initialization is sufficient for this
  controlled task; CF1 is retained only if it improves specificity.
- All pretrained aliases pass: the effect is generic to a useful pretrained
  embedding anchor.
- Only vision-pad passes: report token-specific initialization sensitivity and
  use copied initialization in M1 rather than claiming token invariance.
- No replication passes: stop the recurrent expansion and diagnose pairing,
  checkpoint selection, and intervention implementation.

### Output

- One seven-condition summary with per-seed raw records and paired bootstrap
  intervals.
- A frozen choice of source embedding for the dedicated state token.
- A factorial attribution statement that distinguishes curriculum main effect,
  initialization main effect, and their observed interaction; do not infer
  either main effect from the seed-42 anchor alone.

## M1. Zero-training dedicated-token equivalence

**Question.** Can a dedicated `<STUDENT_STATE>` replace the chosen alias without
changing the learned function?

### Implementation

1. Register exactly one new tokenizer token.
2. Copy the chosen alias input-embedding row to the new token.
3. If input/output embeddings are not tied, copy the output row as well.
4. Preserve the existing adapter and late-fusion mask unchanged.
5. Save the new row with the adapter/tokenizer and verify reload equivalence.

### Tests

- tokenization produces exactly one ID;
- cloned and source embeddings are exactly equal before forward;
- identical prompts produce maximum hidden-state and logit error within the
  expected numerical tolerance;
- C4 response/probe/swap metrics match within paired sampling error;
- save/reload reproduces the same equivalence.

Run the equivalence check at three levels: tokenizer IDs, pre-forward embedding
rows, and end-to-end logits/hidden states.  It is not enough for generated
answers alone to match on a small sample.

### Gate

- No training is allowed in this stage.
- Any material behavior difference is an implementation bug or an unmodeled
  token-ID-specific path and must be resolved before M2.

### One-shot bridge after the gate

Before recurrent training, run the cloned checkpoint through a single frozen
manifest shared with the unrestricted checkpoint and report:

- T1/T2/T3/T5/T6 values separately, never as a single macro score;
- absolute state-compression gaps against the full-context model;
- C4 response, probe, directed swap, random, same-rule, and no-op results;
- exact attention mask and proof that the decoder cannot access raw history.

This experiment decides whether the one-shot causal-state track already stands
on its own and identifies which tasks need extra state capacity or distillation.
The existing hard-route row `0.610/0.457/0.004/0.717/0.840` is a diagnostic
baseline, not evidence that all one-token states must fail T3.

## M2. Minimal recurrent state transition

**Question.** Can the state be updated from the previous state plus one new
interaction without rereading the full history?

### Model interface

Implement a thin wrapper around `Qwen3ForCausalLM`:

```python
z_next = update_state(z_prev, interaction)
logits = decode_from_state(z_next, instruction, query, labels=None)
```

The update sequence is:

```text
<STUDENT_STATE_IN>  serialize(interaction)  <STUDENT_STATE_OUT>
```

- Inject `LN(z_prev)` through `inputs_embeds` at the input carrier.
- Read the causal hidden activation at the output carrier.
- Start with identity `state_in_proj/state_out_proj`; add trainable projections
  only if the identity diagnostic fails for a concrete reason.
- The decoder receives instruction, injected state, and query, but no raw
  history.
- Use one shared tokenizer ID for input carrier, output collector, and decoder
  state address.

### Engineering tests

- exact shape/dtype/device checks for cached states;
- gradients reach LoRA and state projections but not frozen 4-bit weights;
- batch-size-one update/decode matches a hand-constructed forward;
- cached-state detach works at a block boundary;
- checkpoint/save/resume includes projections, tokenizer row, and optimizer;
- no raw-history or KV-cache path reaches the decoder.

### Training diagnostic

- Use a small chronological Malrule trajectory set.
- Unroll 1 and then 4 interactions.
- Run only a diagnostic optimizer step locally; all usable training runs go to
  Feeds.

### Gate

- One-step recurrent response accuracy must be within `5` absolute points of
  the direct one-shot state on identical held-out examples.
- Recurrent/direct cosine similarity must not collapse to a constant state;
  report state variance and pairwise similarity as well as the mean cosine.

## M3. Learn a closed predictive state

**Question.** Which objectives preserve full-context behavior and prevent
recurrent drift?

### Training views

- `full_context`: frozen unrestricted teacher with raw history.
- `direct_state`: current one-shot late-fusion encoder of complete visible
  history.
- `recurrent_state`: sequential updates followed by state-only decoding.

### Losses

\[
L = L_{gen} + \lambda_{KD} L_{KD} + \lambda_{dyn} L_{dyn}
    + \lambda_{cf} L_{cf}.
\]

- `L_gen`: target-token generation for direct and recurrent state views.
- `L_KD`: teacher-to-state token-distribution KL only at supervised output
  positions.
- `L_dyn`: recurrent/direct state agreement at sampled checkpoints, always
  paired with behavioral supervision and with alternating stop-gradient.
- `L_cf`: executable donor-rule response under held-out state/query
  recombination; no auxiliary malrule classifier is attached to the state.

### Sequential ablations

Do not launch a full hyperparameter grid.

1. Recurrent generation only vs recurrent + KD, seed 42.
2. Starting from the better arm: add dynamics consistency, add CF, and add both.
3. Only the selected full configuration proceeds to multi-seed training.

Tune loss weights on validation students with at most two planned adjustments
per loss.  Record every attempted condition, including failures.

Use a capacity intervention only after an objective-level diagnosis.  If one
token fails because of measured compression, compare 1/4/8 tokens under a
matched trainable-parameter and context budget.  Do not increase state size to
hide a transition-closure or causal-specificity failure.

For promotion from the frozen round-750 source checkpoint, every field needed
by the passing diagnostic is part of the selected recipe. This includes the
loss profile, learning rate, train-only T3 view, and causal coefficient.
Current task-retention candidates use `causal_weight=2.0`; the full exact-epoch
continuation must preserve that value and may not silently fall back to the
earlier main-line value `1.0`. Optimizer moments and scheduler position remain
continuous across this single audited curriculum boundary, and all subsequent
recoveries are exact.

### Gates

- No task may improve solely by predicting the majority class; retain AUROC,
  calibration, and generation metrics.
- Recurrent performance degradation from 1 to 4 to 8 updates must be measured,
  not hidden by re-encoding full history.
- The selected state must retain causal specificity under the frozen C4 gate.

## M4. One checkpoint for C1--C4

**Question.** Can one recurrent state support the headline tasks while remaining
causally controllable?

### Curriculum

1. Initialize from the completed unrestricted unified adapter and cloned state
   token.
2. Stabilize direct/recurrent state decoding with generation + KD.
3. Enable trajectory consistency after state-only validation stabilizes.
4. Mix CF1 counterfactual batches with balanced real-data replay.
5. Finish with low-learning-rate real-task replay while retaining a fixed CF
   fraction; never reopen a raw-history decoder path.

### Data

- Chronological DBE-KT22, XES3G5M, and FoundationalASSIST trajectories.
- Eedi Response remains a non-temporal profile task; do not invent chronology.
- Eedi-Mining is one observed-wrong-response update followed by a T3 query.
- Malrule provides executable counterfactual grounding and held-out templates.

### Single-seed pilot gate

Relative to the full-context teacher on identical examples:

- T1 accuracy gap <= `3` points;
- T2 and T5 mathematical-equivalence gap <= `5` points;
- T3 semantic/full-taxonomy gap <= `5` points within its matched protocol;
- frozen C4 `D_specific >= 0.15` with control thresholds preserved;
- report cache size, update latency, and decode latency.

If the pilot fails, classify the failure as compression, transition drift,
task interference, or causal forgetting before changing the method.

### Replication

Only after the pilot passes, train seeds 13/42/97 as independent one-GPU jobs
with identical manifests, ordering rules, steps, and pair sets.

## M5. Paper-critical experiments

These are required for a serious submission after the unified checkpoint
passes M4.

1. **Matched model ablations**
   - unrestricted full context;
   - one-shot late fusion;
   - recurrent generation only;
   - recurrent without KD;
   - recurrent without dynamics consistency;
   - recurrent without counterfactual grounding;
   - full model.
2. **C3 exposure-matched transfer** on the full Eedi split, with equal target
   examples and optimizer exposure across conditions.
3. **T3 audit** with blinded human judgments and inter-annotator agreement for
   the semantic judge.
4. **T5 distance analysis** at `delta=1/3/5`.
5. **T1 robustness** at natural prevalence and on the strongest defensible
   cold-item split.
6. **State capacity** at 1/4/8 tokens under a matched parameter and context
   budget; the one-token model remains the headline unless it clearly fails.
7. **Trajectory length** at 1/4/8/16/32 updates with recurrent/direct drift,
   task performance, calibration, and state variance.
8. **Second-scale replication**, preferably a 7B/8B model, for the minimal
   unified + C4 configuration.
9. **Efficiency**: state cache bytes per student, update/decode latency, and
   full-history versus cached-state context cost.

## M6. World-model planning demonstration (high-value extension)

This stage is not required to establish C1--C4, but it materially strengthens a
world-model and spotlight/oral story.

- Treat problem/feedback selection as the action, student response as the
  observation, and recurrent state as the latent state.
- In the executable Malrule environment, compare random selection, uncertainty
  selection, a KT-style heuristic, model rollout selection, and an oracle.
- Evaluate counterfactual prediction and action ranking on held-out rules and
  templates.
- Do not claim real learning gains from a synthetic simulator.  A real-log
  analysis may establish predictive/off-policy utility only under its
  defensible assumptions.

## 2. Compute routing and submission discipline

### Local

- tokenizer/embedding clone and save/reload tests;
- unit tests, mask audits, state injection, one forward pass, and one diagnostic
  optimizer step;
- dataset materialization, manifests, bootstrap/statistical analysis, and
  report generation;
- forward-only 32B evaluation when the local A100 is available.

### Feeds

- every usable LoRA/QLoRA weight update;
- independent condition/seed jobs, normally one `80G1-A100` each;
- large 32B evaluation only when explicitly authorized or when it is bundled
  with the already-authorized remote training gate.

Each Feeds job must record the image, UAI, base-model revision/path, exact data
hashes, seed, condition, step count, output adapter, and fixed evaluation
manifest.  Adapters/checkpoints are written to remote storage; large base
weights are never downloaded to the submission host.

## 3. Artifact layout

Use one immutable directory per stage and condition:

```text
manifests/world_model_v1/
configs/world_model_v1/
results/world_model_v1/<stage>/<condition>/seed_<seed>/
reports/world_model_v1/
```

Every selected result must contain:

- `run_config.json` and input SHA256 manifest;
- training/evaluation logs and checkpoint receipt;
- task metrics by source/protocol;
- C4 raw pair records plus summarized controls;
- code/config hash and base-model identifier;
- a machine-readable pass/fail record for the stage gate.

## 4. Execution order

The critical path is:

```text
M0 factorial ----- choose source alias -----+
                                           +-> M1 zero-training clone
M1 implementation/tests (in parallel) -----+       |
                                                    +-> one-shot C1--C4 bridge
                                                    +-> M2 one-step/4-step recurrence
  -> M3 sequential loss ablations
  -> M4 unified single-seed pilot
  -> M4 three-seed replication
  -> M5 paper experiments
  -> M6 planning extension
```

Engineering for M1 and unit-test scaffolding for M2 proceed while M0 runs.  The
M1 numerical gate and the one-shot bridge are required before any usable
recurrent training.  No recurrent multi-seed run begins before the single-seed
M4 gate passes.
