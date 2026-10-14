# Student-State Probe Smoke - 2026-07-10

## Goal

Test whether the current Qwen3-32B Malrule response-only adapter contains a linearly readable malrule signal at the `<STUDENT_STATE>` prompt position.

This is a C4 smoke test, not the full C4 protocol. It checks readability only, not causal intervention.

## Model And Data

- Base model: `/data1/liyuejia/models/Qwen3-32B`
- Adapter: `experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_ocr_no_val_qlora_s120_len1536/final_adapter`
- Train activations: `experiments/unified_student_c1/data/processed/single_task_transfer_ocr_no_val/train_malrule_next_response.jsonl`
- Same-template eval: `experiments/unified_student_c1/data/processed/malrule_c1_clean/test_same_template.jsonl`
- Cross-template eval: `experiments/unified_student_c1/data/processed/malrule_c1_clean/test_cross_template.jsonl`
- Task filter: `malrule_next_response`
- Labels: 22 malrule/NONE classes
- Random level: 4.55%
- Sampled rows: train 865, same 196, cross 208

Important: this adapter was trained on response prediction only, not malrule identity. That makes the probe less circular than probing a model directly trained to output malrule labels.

## Method

For each prompt, the script finds `<STUDENT_STATE>` in the text, maps it to the final token of the marker span, and extracts selected block outputs. Because the current tokenizer does not treat `<STUDENT_STATE>` as a single special token, this is only an approximate marker-position probe.

Positions:

- `state`: final token of the `<STUDENT_STATE>` marker span
- `history_last`: token immediately before the marker, used as a position control

Layers scanned: 0, 8, 16, 24, 32, 40, 48, 56, 63.

Probe: standardized linear classifier trained on frozen activations.

Script:

`experiments/unified_student_c1/scripts/probe_student_state_smoke.py`

Output JSON:

`experiments/unified_student_c1/results/probe_student_state_malrule_single_response_qwen3_32b_s40_e10_layers9.json`

## Results

| Layer | STATE same | STATE cross | History-last same | History-last cross |
|---:|---:|---:|---:|---:|
| 0 | 0.679 | 0.707 | 0.663 | 0.697 |
| 8 | 0.679 | 0.697 | 0.719 | 0.721 |
| 16 | 0.704 | 0.736 | 0.745 | 0.716 |
| 24 | 0.724 | 0.697 | 0.770 | 0.760 |
| 32 | 0.724 | 0.688 | 0.755 | 0.721 |
| 40 | 0.704 | 0.716 | 0.776 | 0.779 |
| 48 | 0.816 | 0.755 | 0.786 | 0.755 |
| 56 | 0.862 | 0.841 | 0.750 | 0.736 |
| 63 | 0.811 | 0.841 | 0.679 | 0.697 |

Best STATE cross-template accuracy: 0.841 at layers 56 and 63.

Best history-last cross-template accuracy: 0.779 at layer 40.

## Interpretation

The readable signal is strong: cross-template malrule identity is far above random, even though the model was trained only to predict the next response.

This supports the weak C4-readability claim: the response-trained model internally encodes which malrule-like behavior generated the student's history.

However, this is not yet a strict "student state" result. The history-last position is also highly readable, so the signal is not uniquely concentrated at the marker. The current marker is also not a single special token, which weakens positional precision.

## Caveats Before Treating This As C4 Evidence

- `<STUDENT_STATE>` is not a single tokenizer special token in this run.
- No shuffled-label probe control yet.
- No unfine-tuned base-model control yet.
- No bottleneck-mask variant yet.
- No state-swap intervention yet.
- The probe uses sampled rows rather than the full split.

## Next Step

The next scientifically useful step is to rerun this with:

1. a strict single-token `<STUDENT_STATE>` tokenizer/model,
2. shuffled-label and base-model controls,
3. bottleneck training,
4. state-swap causal intervention.

