# C4 history-length diagnostic: audited results

Post-hoc exploratory, four independent pairs/category, one training seed. Same queries as prior test. No new replication gate or multiple-comparison correction. Zero-width empirical intervals are not zero population uncertainty.

All source design, interventions, sample construction and exclusions are in the frozen protocol. No generations are excluded.

| Arm | Category | History | Open history | Donor native | Swap | Random | D | All-three swap | No-op mismatches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | offset | h3 | 0/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | offset | h6 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | offset | h12 | 3/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | offset | repeat3x4 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | composition | h3 | 0/24 | 1/24 | 1/24 | 0/24 | 0.0417 | 0/8 | 0 |
| A | composition | h6 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | composition | h12 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | composition | repeat3x4 | 0/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| A | conditional | h3 | 14/24 | 14/24 | 14/24 | 13/24 | 0.0417 | 0/8 | 0 |
| A | conditional | h6 | 13/24 | 13/24 | 13/24 | 12/24 | 0.0417 | 0/8 | 0 |
| A | conditional | h12 | 13/24 | 13/24 | 12/24 | 11/24 | 0.0417 | 0/8 | 1 |
| A | conditional | repeat3x4 | 15/24 | 13/24 | 13/24 | 13/24 | 0.0000 | 0/8 | 0 |
| A | seen_control | h3 | 23/24 | 24/24 | 24/24 | 0/24 | 1.0000 | 8/8 | 0 |
| A | seen_control | h6 | 21/24 | 24/24 | 24/24 | 0/24 | 1.0000 | 8/8 | 0 |
| A | seen_control | h12 | 22/24 | 24/24 | 24/24 | 0/24 | 1.0000 | 8/8 | 0 |
| A | seen_control | repeat3x4 | 20/24 | 24/24 | 24/24 | 1/24 | 0.9583 | 8/8 | 0 |
| B | offset | h3 | 0/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 1 |
| B | offset | h6 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| B | offset | h12 | 3/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| B | offset | repeat3x4 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| B | composition | h3 | 0/24 | 1/24 | 1/24 | 0/24 | 0.0417 | 0/8 | 0 |
| B | composition | h6 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| B | composition | h12 | 1/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| B | composition | repeat3x4 | 0/24 | 0/24 | 0/24 | 0/24 | 0.0000 | 0/8 | 0 |
| B | conditional | h3 | 14/24 | 14/24 | 14/24 | 13/24 | 0.0417 | 0/8 | 0 |
| B | conditional | h6 | 13/24 | 13/24 | 13/24 | 12/24 | 0.0417 | 0/8 | 0 |
| B | conditional | h12 | 13/24 | 13/24 | 12/24 | 11/24 | 0.0417 | 0/8 | 0 |
| B | conditional | repeat3x4 | 15/24 | 13/24 | 13/24 | 13/24 | 0.0000 | 0/8 | 0 |
| B | seen_control | h3 | 23/24 | 24/24 | 24/24 | 0/24 | 1.0000 | 8/8 | 0 |
| B | seen_control | h6 | 21/24 | 24/24 | 24/24 | 0/24 | 1.0000 | 8/8 | 0 |
| B | seen_control | h12 | 22/24 | 24/24 | 24/24 | 0/24 | 1.0000 | 8/8 | 0 |
| B | seen_control | repeat3x4 | 20/24 | 24/24 | 24/24 | 1/24 | 0.9583 | 8/8 | 0 |

## Paired diagnostic contrasts

| Arm | Category | Contrast | Measure | Difference [95% pair-cluster CI] |
|---|---|---|---|---|
| A | offset | h12_minus_h3 | open_history | 0.1250 [0.0417, 0.1667] |
| A | offset | h12_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | offset | h12_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| A | offset | h12_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| A | offset | h12_minus_repeat3x4 | open_history | 0.0833 [0.0000, 0.1667] |
| A | offset | h12_minus_repeat3x4 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | offset | h12_minus_repeat3x4 | swap | 0.0000 [0.0000, 0.0000] |
| A | offset | h12_minus_repeat3x4 | specificity | 0.0000 [0.0000, 0.0000] |
| A | offset | h6_minus_h3 | open_history | 0.0417 [0.0000, 0.1250] |
| A | offset | h6_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | offset | h6_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| A | offset | h6_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| A | composition | h12_minus_h3 | open_history | 0.0417 [0.0000, 0.1250] |
| A | composition | h12_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| A | composition | h12_minus_h3 | swap | -0.0417 [-0.1250, 0.0000] |
| A | composition | h12_minus_h3 | specificity | -0.0417 [-0.1250, 0.0000] |
| A | composition | h12_minus_repeat3x4 | open_history | 0.0417 [0.0000, 0.1250] |
| A | composition | h12_minus_repeat3x4 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | composition | h12_minus_repeat3x4 | swap | 0.0000 [0.0000, 0.0000] |
| A | composition | h12_minus_repeat3x4 | specificity | 0.0000 [0.0000, 0.0000] |
| A | composition | h6_minus_h3 | open_history | 0.0417 [0.0000, 0.1250] |
| A | composition | h6_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| A | composition | h6_minus_h3 | swap | -0.0417 [-0.1250, 0.0000] |
| A | composition | h6_minus_h3 | specificity | -0.0417 [-0.1250, 0.0000] |
| A | conditional | h12_minus_h3 | open_history | -0.0417 [-0.2500, 0.1250] |
| A | conditional | h12_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| A | conditional | h12_minus_h3 | swap | -0.0833 [-0.1667, 0.0000] |
| A | conditional | h12_minus_h3 | specificity | 0.0000 [-0.1250, 0.1250] |
| A | conditional | h12_minus_repeat3x4 | open_history | -0.0833 [-0.2500, 0.0000] |
| A | conditional | h12_minus_repeat3x4 | donor_native | 0.0000 [-0.1250, 0.1250] |
| A | conditional | h12_minus_repeat3x4 | swap | -0.0417 [-0.1667, 0.0833] |
| A | conditional | h12_minus_repeat3x4 | specificity | 0.0417 [-0.1250, 0.2500] |
| A | conditional | h6_minus_h3 | open_history | -0.0417 [-0.2500, 0.1250] |
| A | conditional | h6_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| A | conditional | h6_minus_h3 | swap | -0.0417 [-0.1250, 0.0000] |
| A | conditional | h6_minus_h3 | specificity | 0.0000 [-0.1250, 0.1250] |
| A | seen_control | h12_minus_h3 | open_history | -0.0417 [-0.1250, 0.0000] |
| A | seen_control | h12_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h12_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h12_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h12_minus_repeat3x4 | open_history | 0.0833 [0.0000, 0.2500] |
| A | seen_control | h12_minus_repeat3x4 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h12_minus_repeat3x4 | swap | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h12_minus_repeat3x4 | specificity | 0.0417 [0.0000, 0.1250] |
| A | seen_control | h6_minus_h3 | open_history | -0.0833 [-0.1667, 0.0000] |
| A | seen_control | h6_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h6_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| A | seen_control | h6_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| B | offset | h12_minus_h3 | open_history | 0.1250 [0.0417, 0.1667] |
| B | offset | h12_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | offset | h12_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| B | offset | h12_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| B | offset | h12_minus_repeat3x4 | open_history | 0.0833 [0.0000, 0.1667] |
| B | offset | h12_minus_repeat3x4 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | offset | h12_minus_repeat3x4 | swap | 0.0000 [0.0000, 0.0000] |
| B | offset | h12_minus_repeat3x4 | specificity | 0.0000 [0.0000, 0.0000] |
| B | offset | h6_minus_h3 | open_history | 0.0417 [0.0000, 0.1250] |
| B | offset | h6_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | offset | h6_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| B | offset | h6_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| B | composition | h12_minus_h3 | open_history | 0.0417 [0.0000, 0.1250] |
| B | composition | h12_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| B | composition | h12_minus_h3 | swap | -0.0417 [-0.1250, 0.0000] |
| B | composition | h12_minus_h3 | specificity | -0.0417 [-0.1250, 0.0000] |
| B | composition | h12_minus_repeat3x4 | open_history | 0.0417 [0.0000, 0.1250] |
| B | composition | h12_minus_repeat3x4 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | composition | h12_minus_repeat3x4 | swap | 0.0000 [0.0000, 0.0000] |
| B | composition | h12_minus_repeat3x4 | specificity | 0.0000 [0.0000, 0.0000] |
| B | composition | h6_minus_h3 | open_history | 0.0417 [0.0000, 0.1250] |
| B | composition | h6_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| B | composition | h6_minus_h3 | swap | -0.0417 [-0.1250, 0.0000] |
| B | composition | h6_minus_h3 | specificity | -0.0417 [-0.1250, 0.0000] |
| B | conditional | h12_minus_h3 | open_history | -0.0417 [-0.2500, 0.1250] |
| B | conditional | h12_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| B | conditional | h12_minus_h3 | swap | -0.0833 [-0.1667, 0.0000] |
| B | conditional | h12_minus_h3 | specificity | 0.0000 [-0.1250, 0.1250] |
| B | conditional | h12_minus_repeat3x4 | open_history | -0.0833 [-0.2500, 0.0000] |
| B | conditional | h12_minus_repeat3x4 | donor_native | 0.0000 [-0.1250, 0.1250] |
| B | conditional | h12_minus_repeat3x4 | swap | -0.0417 [-0.1667, 0.0833] |
| B | conditional | h12_minus_repeat3x4 | specificity | 0.0417 [-0.1250, 0.2500] |
| B | conditional | h6_minus_h3 | open_history | -0.0417 [-0.2500, 0.1250] |
| B | conditional | h6_minus_h3 | donor_native | -0.0417 [-0.1250, 0.0000] |
| B | conditional | h6_minus_h3 | swap | -0.0417 [-0.1250, 0.0000] |
| B | conditional | h6_minus_h3 | specificity | 0.0000 [-0.1250, 0.1250] |
| B | seen_control | h12_minus_h3 | open_history | -0.0417 [-0.1250, 0.0000] |
| B | seen_control | h12_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h12_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h12_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h12_minus_repeat3x4 | open_history | 0.0833 [0.0000, 0.2500] |
| B | seen_control | h12_minus_repeat3x4 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h12_minus_repeat3x4 | swap | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h12_minus_repeat3x4 | specificity | 0.0417 [0.0000, 0.1250] |
| B | seen_control | h6_minus_h3 | open_history | -0.0833 [-0.1667, 0.0000] |
| B | seen_control | h6_minus_h3 | donor_native | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h6_minus_h3 | swap | 0.0000 [0.0000, 0.0000] |
| B | seen_control | h6_minus_h3 | specificity | 0.0000 [0.0000, 0.0000] |

## Interpretation boundary

More unique history helping beyond repeated history supports an information-dose explanation in this panel. Oracle success with history failure is consistent with an induction limitation, not a uniquely established cause. Donor-native success with swap failure implicates transport; oracle failure leaves arithmetic or instruction-following unresolved. Repetition is not an exact token-length control. The longer context also shifts the state position and is outside the three-example repair-training format. Known controls and all no-op mismatches remain visible.

Full per-condition counts, invalids, whole-triplet rule signatures, paired transitions, raw example text and provenance are retained in the summary JSON.
