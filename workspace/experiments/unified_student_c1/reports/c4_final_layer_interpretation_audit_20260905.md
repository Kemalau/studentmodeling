# C4 final-layer comparison: computational-path audit

## Question and non-goals

Does the null final-layer-63 bridge show that the student information cannot be
represented by one vector? **No: that comparison is structurally uninformative
about single-vector sufficiency in its implemented form.** This audit changes no
trained model, frozen endpoint, or original result. It is a code/architecture
diagnostic, not new 32B behavioral inference.

## Sample and intervention

The old `evaluate_single_layer_state_bridge.py` captures a decoder block OUTPUT
at the state position, then overwrites that same position at the same decoder
block OUTPUT in a history-free recipient prompt. `score_state` evaluates answer
tokens later in the sequence. The intervention is

`h_out[layer,state] <- donor_state[layer]`.

For layer 63 of 64, all self-attention operations have already happened. Qwen3's
remaining final normalization and vocabulary head operate independently at each
position. The overwrite therefore changes the logits at the state position, but
cannot change any later answer-position logits. The last layer's K/V were also
computed before this output overwrite; cached future decoding does not add a
path that was absent. This holds for this hook and placement, not for all ways
of using a last-layer representation.

## Verification

`tests/test_c4_final_output_path.py` constructs a tiny random three-block Qwen3
model on CPU, seed 46, hidden size 32, nine synthetic input tokens, state index 4,
late-fusion prompt end 7. A random replacement at final block output:

- changes the state-position vocabulary logits;
- leaves ALL later-position logits bitwise equal;
- leaves all layers' cached keys and values bitwise equal.

The positive path control replaces the preceding block's output instead: final
answer-position logits do change. The test passed both in the general local
environment and `/home/v-yuejiali/.venvs/studentmodeling`, whose Transformers
4.57.6 matches the frozen remote requirements. An initial test invocation
failed only because its cache comparison used an obsolete indexing API;
comparison via the installed cache's `.layers[].keys/values` fixed the test.
No experiment output or statistical criterion was changed.

## Consequence for paper language

Keep the measured layerwise bridge and its historical numbers, but do not write
that final-layer null establishes “the representation is a trajectory rather
than a vector.” The defensible wording is:

> The implemented depth-indexed transport interface retains the measured
> one-shot effect. Its comparison against a final-block-output-only overwrite
> does not establish that a single-vector carrier is insufficient, because that
> overwrite has no computational path to the later answer tokens.

An informative comparison must provide both carriers with a downstream readout
path: e.g. single earlier native-layer output versus a depth-matched layer block,
or an explicitly trained/validated read-in from a final vector. Such tests are
not silently claimed completed here. Single earlier-layer nulls also do not
prove that every possible learned single-vector interface must fail.

## Reproduction

```bash
cd /home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
/home/v-yuejiali/DeepGraph/.venv/bin/python -m pytest -q tests/test_c4_final_output_path.py
```

Implementation audited: `scripts/evaluate_single_layer_state_bridge.py`,
`scripts/state_swap_misconception_smoke.py`,
`scripts/evaluate_c4_freegen_novel.py`, and installed Qwen3 decoder, normalization,
and language-model head implementation. This is separate from the ongoing
`c4-rule-followup-seed42-v2-20260905` behavior experiment.
