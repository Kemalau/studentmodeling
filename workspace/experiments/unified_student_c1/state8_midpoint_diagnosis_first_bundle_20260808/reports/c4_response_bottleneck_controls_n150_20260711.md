# C4 Response Bottleneck Swap Controls n150 — 2026-07-11

## Purpose

This run addresses the earlier underpowered C4 response-swap smoke. The previous 0/7 result was not interpretable because pair sampling used at most one donor per recipient and only 30 pairs were scored.

This run uses:

- fixed ordered-pair sampling;
- 150 legal response-level donor/recipient pairs;
- all-layer STATE replacement;
- five conditions: `none`, `noop`, `swap`, `random`, `samemal`;
- response-level classification via executable answer matching.

Model is still the response-only bottleneck S200 adapter:

`experiments/unified_student_c1/results/qwen3_32b_single_malrule_next_response_state_token_bottleneck_qlora_s200_len1536/final_adapter`

Important caveat: this is still below the full decision gate because response accuracy is only 0.5917, below the intended 0.7 threshold.

## Pair Generation Check

For `test_same_template` / `malrule_next_response`:

| quantity | value |
|---|---:|
| rows | 300 |
| malrule rows | 236 |
| NONE rows | 64 |
| legal ordered response pairs | 3566 |
| recipients with at least one legal partner | 113 / 236 |

After fixing the sampler:

| requested pairs | built pairs |
|---:|---:|
| 30 | 30 |
| 100 | 100 |
| 150 | 150 |
| 300 | 300 |
| 1000 | 1000 |

So the earlier `n=7` clean denominator was not a data sparsity result. It came from the old sampler plus weak baseline `none=stay_B` rate.

## Output

Result file:

`experiments/unified_student_c1/results/c4_response_bottleneck_s200_swap_response_all_layers_n150_controls.json`

## All-Pair Distribution

| condition | n | flip_to_A | stay_B | correct | other |
|---|---:|---:|---:|---:|---:|
| none | 150 | 33 | 34 | 47 | 36 |
| noop | 150 | 30 | 42 | 47 | 31 |
| samemal | 150 | 32 | 39 | 45 | 34 |
| random | 150 | 36 | 54 | 17 | 43 |
| swap | 150 | 31 | 45 | 43 | 31 |

Rates:

| condition | flip_to_A | stay_B | correct | other |
|---|---:|---:|---:|---:|
| none | 0.220 | 0.227 | 0.313 | 0.240 |
| noop | 0.200 | 0.280 | 0.313 | 0.207 |
| samemal | 0.213 | 0.260 | 0.300 | 0.227 |
| random | 0.240 | 0.360 | 0.113 | 0.287 |
| swap | 0.207 | 0.300 | 0.287 | 0.207 |

## Noop Check

Noop is not perfectly identical to none:

| subset | n | noop changed top-1 |
|---|---:|---:|
| all pairs | 150 | 17 |
| none = stay_B | 34 | 1 |

Interpretation:

- In the clean subset, noop is mostly stable: 1/34 changed.
- Across all pairs, noop changes 17/150, so the pipeline still has some numerical/context fragility.
- This is not catastrophic for smoke, but it is not clean enough for a final causal claim.

## Clean Subset: none = stay_B

This is the most important subset for directional flip.

| condition | n | flip_to_A | stay_B | correct | other |
|---|---:|---:|---:|---:|---:|
| none | 34 | 0 | 34 | 0 | 0 |
| noop | 34 | 1 | 33 | 0 | 0 |
| samemal | 34 | 0 | 34 | 0 | 0 |
| random | 34 | 7 | 24 | 0 | 3 |
| swap | 34 | 0 | 34 | 0 | 0 |

Delta metrics in the clean subset:

| condition | changed top-1 | stayB_to_flipA | delta p(A) | delta p(B) | delta margin A-B |
|---|---:|---:|---:|---:|---:|
| noop | 1 | 1 | +0.002106 | +0.002949 | +0.003202 |
| samemal | 0 | 0 | +0.002430 | +0.000487 | +0.013801 |
| random | 10 | 7 | +0.020081 | +0.004895 | +0.047023 |
| swap | 0 | 0 | +0.001984 | -0.000269 | +0.011421 |

Key observation:

- `random` has a real disruptive effect and can produce `flip_to_A` in 7/34 clean cases.
- `swap` produces 0/34 `stay_B -> flip_to_A`.
- `samemal` also produces 0/34, as desired.
- `swap` is not better than `samemal`; it is much weaker than random in terms of perturbation.

## Current Interpretation

This run is much better than the previous 0/7 smoke, but still not a final 4b adjudication.

What it says:

- The previous pair-generation bug is fixed.
- Five-condition infrastructure now works.
- Random vector patching is strong enough to perturb behavior, so the hook location is not completely inert.
- Donor STATE swap still does not directionally flip recipient answers in the clean subset.

What it does not yet prove:

- It does not prove 4b fails, because the response model is below the EM gate and the clean subset is only 34 pairs.
- It does not settle marker vs bottleneck at full spec.

## Next

To reach decision quality:

1. improve response-only bottleneck EM to at least 0.7;
2. rerun the same five-condition all-layer swap with enough sampled pairs to get at least 100 `none=stay_B` clean cases;
3. keep `noop` and `random` as hard self-checks;
4. only then interpret `swap` vs `samemal`.
