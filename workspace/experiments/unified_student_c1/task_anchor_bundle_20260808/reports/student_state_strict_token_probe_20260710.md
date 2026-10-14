# Strict `<STUDENT_STATE>` Probe Smoke - 2026-07-10

## Goal

Rerun the Malrule student-state readability smoke test with `<STUDENT_STATE>` registered as a real single special token.

This addresses the main caveat from the previous probe: the marker was previously tokenized as five pieces (`<`, `ST`, `UDENT`, `_STATE`, `>`).

## Code Changes

- `experiments/unified_student_c1/scripts/train_c1_lora.py`
  - Added `--add-state-token`.
  - Registers `<STUDENT_STATE>` as an additional special token.
  - Asserts it tokenizes to exactly one id.
  - Resizes model embeddings.
  - Mean-initializes the new input/output embedding row.
  - Saves/trains `embed_tokens` via PEFT `modules_to_save`.
  - Added `--save-final-only` to avoid writing multiple large embedding checkpoints.

- `experiments/unified_student_c1/scripts/probe_student_state_smoke.py`
  - Loads tokenizer from adapter dir when available.
  - Resizes base model embeddings before loading the adapter.

- `experiments/unified_student_c1/scripts/evaluate_c1.py`
  - Same tokenizer-from-adapter and resize fix for strict-token adapters.

## Strict Adapter

Training command used Qwen3-32B, response-only Malrule data, QLoRA, 40 optimizer steps.

Output:

`experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_qlora_s40_len1536/final_adapter`

Tokenizer verification:

`<STUDENT_STATE>` -> `[151669]`

The adapter is large (`adapter_model.safetensors` about 12.7GB) because PEFT saves the resized `embed_tokens` module.

## Probe Setup

Probe output:

`experiments/unified_student_c1/results/probe_student_state_malrule_strict_state_token_qwen3_32b_s40_e10_layers9.json`

- Labels: 22 malrule/NONE classes
- Random level: 4.55%
- Rows: train 865, same-template 196, cross-template 208
- Layers: 0, 8, 16, 24, 32, 40, 48, 56, 63
- Positions:
  - `state`: the single `<STUDENT_STATE>` token
  - `history_last`: token immediately before `<STUDENT_STATE>`

## Strict-Token Results

| Layer | STATE same | STATE cross | History-last same | History-last cross |
|---:|---:|---:|---:|---:|
| 0 | 0.673 | 0.692 | 0.648 | 0.688 |
| 8 | 0.684 | 0.707 | 0.730 | 0.688 |
| 16 | 0.684 | 0.654 | 0.709 | 0.692 |
| 24 | 0.663 | 0.659 | 0.750 | 0.760 |
| 32 | 0.689 | 0.649 | 0.796 | 0.712 |
| 40 | 0.694 | 0.659 | 0.796 | 0.774 |
| 48 | 0.730 | 0.726 | 0.801 | 0.812 |
| 56 | 0.760 | 0.750 | 0.776 | 0.760 |
| 63 | 0.770 | 0.788 | 0.755 | 0.736 |

Best STATE cross-template accuracy: 0.788 at layer 63.

Best history-last cross-template accuracy: 0.812 at layer 48.

## Comparison With Previous Non-Strict Probe

Previous marker-as-five-tokens response-only adapter:

- Best STATE cross: 0.841
- Best history-last cross: 0.779

Strict single-token 40-step adapter:

- Best STATE cross: 0.788
- Best history-last cross: 0.812

## Interpretation

Readability still clearly holds under the stricter single-token marker setup: 78.8% cross-template accuracy is far above the 4.55% random baseline.

But marker specificity does not hold yet. The history-last control is at least as readable as the STATE token in this short marker-only training run. This means the model has learned a distributed representation of malrule behavior across the history prefix, rather than naturally compressing it uniquely into `<STUDENT_STATE>`.

This is not a failure of C4; it is exactly why the pilot design needs the bottleneck variant. Marker-only answers "does the information exist near the marker?" The answer is yes. Bottleneck must answer "can we force that information through the marker and then causally use it?"

## Next Decision

The next useful experiment is not another marker-only probe. It is the bottleneck variant:

- keep `<STUDENT_STATE>` as a single token,
- train response-only Malrule again,
- block question/target tokens from attending to history,
- rerun probe,
- then run noop/swap activation patching.

