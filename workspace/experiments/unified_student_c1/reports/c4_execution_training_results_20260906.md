# C4 candidate-free execution training: audited pilot results


## 1. Question and non-goals

Does candidate-free SFT improve one-shot student simulation, and does differentiable state-exchange supervision improve donor-specific free generation beyond matched SFT? One exploratory training seed (42), six known arithmetic rules, final checkpoint only. This is an explicitly trained causal interface, not evidence that the old model naturally contained a general executable rule engine. No recurrent or human-student claim.


## 2. Sample construction and concrete sample

Programmatically generated subtraction and common-denominator multiplication. Train: 160 independent unordered history pairs, two directions, three queries, two instruction modes = 1,920 rows. Validation: 8 pairs, 96 rows. Test: 16 pairs, 32 directions, three queries = 96 directional queries; ten conditions per query = 960 generations per model, 2,880 across baseline/A/B. Each history has three observations. Test/validation history rules are identifiable within the six-rule panel. Two auxiliary training histories have documented ambiguity and are not donor/native sources.

Generator seed 2026090601. Number-multiset/operand signatures are excluded against old CF1 6,000/600, MalRule 9,000/900/900/900, the prior 81-query numerical panel, and across new train/valid/test. Each target differs from the other rule and correct answer, visible-history numeric tokens, and the other two query answers. 5,520 operand-based answer checks passed. This does not prove pretraining novelty or unseen-rule/template generalization.

Illustrative construction (not a claimed measured success): a donor following ignore-denominators should answer `2/7 × 3/7` with `6`, not `6/49`. The source contains its history and state token only, without the current query or target.


## 3. Coordinate, loss and exact formula

At the `vision_pad` address after each of 64 decoder blocks, B replaces recipient residual `h_recipient^l ← s_donor^l`. The donor and recipient are rows in one differentiable forward pass. Non-reentrant gradient checkpointing preserves source gradients. A uses donor history with ordinary answer-token SFT; both arms execute two-row forwards. Only target-row answer/EOS tokens contribute to `L = −(1/|T|) Σ[t∈T] log pθ(y_t | prefix, y_<t)`. Donor-source/prompt labels are −100. No probe or neuron ranking is optimized.


## 4. Illustrative loss sign

Raising a target-token probability from 0.2 to 0.5 lowers its negative log likelihood from 1.609 to 0.693. These numbers illustrate the objective, not observed per-neuron attribution or a measured intervention effect.


## 5. Fixed candidate and lineage

There is no neural-coordinate candidate search or ranking. The sole endpoint is the preselected final step 240 for each arm, warm-started from the historical CF1 `vision_pad` rank-8 adapter. Initial adapter SHA256: `c9b463ed12f11ef84d346c1399d2588b9b396ecb9e15c5f4ca85c7f1d7fe991f`.

| Model | Evaluated adapter SHA256 | Prediction SHA256 |
| --- | --- | --- |
| baseline | c9b463ed12f11ef84d346c1399d2588b9b396ecb9e15c5f4ca85c7f1d7fe991f | 732beae2ceaf35ed0452d5c9217bd4fd0caa0462dacd0dcf36024b8c068f73dd |
| A | 7d614cf19d847aa220d7b87c8d471ada95867476653425188d633747efccd1b6 | 142c896214558c71f541d01bd75acd17386acd4d5b89fe2438fedd637cf51d08 |
| B | 87a9bee0d0bde5be481ed95f2209d91807c974a190953aae3b1fa33a0eb24a32 | 1bb454a0017db601d6ebdd2485721bb6c0411eace34642a65c83081b3130389d |


## 6. Selection funnel and training budget

Frozen generated manifest → all rows retained → identical supervised-target order and 26,524 target/EOS tokens per arm → 240 updates → final checkpoint only. No test-based seed, prompt, layer, checkpoint, or example selection. First 120 updates use explicit rules; last 120 use history, with differentiable exchange in B. QLoRA NF4/bfloat16, rank 8, dropout zero, AdamW lr 2e-5, 12-step warmup then linear decay, accumulation 8, grad clip 1, weight decay .01, max sequence 768 with no truncation. Save every 40 updates. The original adapter is unchanged.


## 7. Held-out causal validation

`D_model = mean_i[1(output_swap,i = target_donor,i) − 1(output_random,i = target_donor,i)]`. Primary contrast: `D_B − D_A`. The random donor is a distinct history with recipient rule, applicable to the query and with a target distinct from the donor. It is not an arbitrary random vector distribution. Same-donor-rule and own-state no-op controls are separate.

Bootstrap: 10,000 resamples of the 16 unordered history-pair IDs, keeping both directions and all three queries together. Intervals are exploratory 95% percentile intervals, unadjusted; no retrospective replication gate.


## 8. Generation-time intervention

Full residual replacement at all 64 state-token addresses, dose 1, on the prefill pass only. A donor trajectory is captured once per history and reused for different queries. Fresh KV per generation. Post-state positions cannot read history; query tokens cannot read state; answer anchor/generated tokens can read state. Baseline is freshly evaluated with this same corrected mask. No history-free or final-layer-versus-trajectory comparison is part of this pilot.


## 9. Evaluation protocol

Greedy decoding, 48-token limit, stop at EOS or closing predicted-response tag. Strict single-number/tag parsing, exact rational equality. Invalid, multiple-answer or truncated outputs count as misses. No LLM judge. Primary open conditions provide no current correct answer, menu or rule description. The explicit-rule control and candidate-visible controls are separately labelled; menu controls must not be interpreted as unsupported free generation.


## 10. Aggregate and paired results

| Model | Directed donor hit /96 | Wrong-rule donor hit /96 | D [95% pair-cluster CI] |
| --- | --- | --- | --- |
| baseline | 4/96 (4.2%) | 3/96 (3.1%) | 0.0104 [-0.0312, 0.0521] |
| A | 56/96 (58.3%) | 6/96 (6.2%) | 0.5208 [0.3021, 0.7396] |
| B | 56/96 (58.3%) | 8/96 (8.3%) | 0.5000 [0.2604, 0.7396] |


Primary **B − A specificity**: -0.0208 [-0.0521, 0.0000].


Secondary paired specificity changes against the freshly evaluated baseline: A − baseline = 0.5104 [0.2917, 0.7292]; B − baseline = 0.4896 [0.2604, 0.7188]. These are unadjusted exploratory secondary intervals, not replication gates.


All conditions follow (native accuracy means recipient hits, all other accuracy columns below are explicitly donor or recipient hits as labelled).

| Model | Condition | Donor hits | Recipient hits | Invalid |
| --- | --- | --- | --- | --- |
| baseline | native | 1/96 (1.0%) | 4/96 (4.2%) | 3 |
| baseline | donor_native | 4/96 (4.2%) | 1/96 (1.0%) | 3 |
| baseline | noop | 1/96 (1.0%) | 4/96 (4.2%) | 3 |
| baseline | swap | 4/96 (4.2%) | 2/96 (2.1%) | 3 |
| baseline | random | 3/96 (3.1%) | 3/96 (3.1%) | 3 |
| baseline | same_donor | 4/96 (4.2%) | 2/96 (2.1%) | 2 |
| baseline | oracle | 35/96 (36.5%) | 2/96 (2.1%) | 0 |
| baseline | menu_native | 12/96 (12.5%) | 80/96 (83.3%) | 0 |
| baseline | menu_swap | 74/96 (77.1%) | 17/96 (17.7%) | 0 |
| baseline | menu_random | 12/96 (12.5%) | 81/96 (84.4%) | 0 |
| A | native | 7/96 (7.3%) | 57/96 (59.4%) | 0 |
| A | donor_native | 57/96 (59.4%) | 7/96 (7.3%) | 0 |
| A | noop | 7/96 (7.3%) | 57/96 (59.4%) | 0 |
| A | swap | 56/96 (58.3%) | 8/96 (8.3%) | 0 |
| A | random | 6/96 (6.2%) | 54/96 (56.2%) | 1 |
| A | same_donor | 54/96 (56.2%) | 6/96 (6.2%) | 1 |
| A | oracle | 60/96 (62.5%) | 3/96 (3.1%) | 0 |
| A | menu_native | 12/96 (12.5%) | 80/96 (83.3%) | 0 |
| A | menu_swap | 77/96 (80.2%) | 15/96 (15.6%) | 0 |
| A | menu_random | 14/96 (14.6%) | 80/96 (83.3%) | 0 |
| B | native | 8/96 (8.3%) | 56/96 (58.3%) | 0 |
| B | donor_native | 56/96 (58.3%) | 8/96 (8.3%) | 0 |
| B | noop | 8/96 (8.3%) | 56/96 (58.3%) | 0 |
| B | swap | 56/96 (58.3%) | 8/96 (8.3%) | 0 |
| B | random | 8/96 (8.3%) | 53/96 (55.2%) | 1 |
| B | same_donor | 53/96 (55.2%) | 7/96 (7.3%) | 1 |
| B | oracle | 60/96 (62.5%) | 3/96 (3.1%) | 0 |
| B | menu_native | 12/96 (12.5%) | 80/96 (83.3%) | 0 |
| B | menu_swap | 77/96 (80.2%) | 15/96 (15.6%) | 0 |
| B | menu_random | 13/96 (13.5%) | 81/96 (84.4%) | 0 |


Three-query agreement (32 directions, but only 16 independent pairs):

| Model | Condition | All-three donor matches /32 |
| --- | --- | --- |
| baseline | native | 0/32 (0.0%) |
| baseline | donor_native | 0/32 (0.0%) |
| baseline | noop | 0/32 (0.0%) |
| baseline | swap | 0/32 (0.0%) |
| baseline | random | 0/32 (0.0%) |
| baseline | same_donor | 0/32 (0.0%) |
| baseline | oracle | 4/32 (12.5%) |
| baseline | menu_native | 1/32 (3.1%) |
| baseline | menu_swap | 21/32 (65.6%) |
| baseline | menu_random | 2/32 (6.2%) |
| A | native | 0/32 (0.0%) |
| A | donor_native | 14/32 (43.8%) |
| A | noop | 0/32 (0.0%) |
| A | swap | 14/32 (43.8%) |
| A | random | 0/32 (0.0%) |
| A | same_donor | 14/32 (43.8%) |
| A | oracle | 15/32 (46.9%) |
| A | menu_native | 1/32 (3.1%) |
| A | menu_swap | 20/32 (62.5%) |
| A | menu_random | 2/32 (6.2%) |
| B | native | 0/32 (0.0%) |
| B | donor_native | 14/32 (43.8%) |
| B | noop | 0/32 (0.0%) |
| B | swap | 14/32 (43.8%) |
| B | random | 0/32 (0.0%) |
| B | same_donor | 14/32 (43.8%) |
| B | oracle | 15/32 (46.9%) |
| B | menu_native | 1/32 (3.1%) |
| B | menu_swap | 20/32 (62.5%) |
| B | menu_random | 2/32 (6.2%) |


Validation losses (teacher-forced; not generation accuracy):

| Arm | Checkpoint | Explicit-rule | History |
| --- | --- | --- | --- |
| A | 120 | 0.117552 | 0.267486 |
| A | 240 | 0.132085 | 0.141415 |
| B | 120 | 0.117552 | 0.268075 |
| B | 240 | 0.132082 | 0.142224 |


## 11. Failures, corruption and representative examples

| Model | No-op token/stop mismatches /96 |
| --- | --- |
| baseline | 0 |
| A | 1 |
| B | 1 |

No-op mismatches, invalid generations and per-rule failures remain in the data. These examples are deterministic first qualifying groups in manifest order, not an estimate of prevalence or an example-filtered metric. An empty success list means no group had all three directed hits and zero random hits.

```json
{
  "baseline": {
    "first_success": [],
    "first_failure": [
      {
        "group_id": "test:subtraction:000:0",
        "pair_id": "test:subtraction:000",
        "donor_rule": "subtraction.stops_borrow_at_zero",
        "donor_history": "<student_history>\nTrial 1:\nProblem: Subtract 8647 from 8804.\nStudent answer: 163\nCorrect answer: 157\nOutcome: incorrect\n\nTrial 2:\nProblem: What is 2082 minus 1485?\nStudent answer: 617\nCorrect answer: 597\nOutcome: incorrect\n\nTrial 3:\nProblem: Subtract 1814 from 3601.\nStudent answer: 1793\nCorrect answer: 1787\nOutcome: incorrect\n</student_history>\n",
        "recipient_history": "<student_history>\nTrial 1:\nProblem: What is 3015 minus 2575?\nStudent answer: 540\nCorrect answer: 440\nOutcome: incorrect\n\nTrial 2:\nProblem: Calculate: 4027 - 889\nStudent answer: 3838\nCorrect answer: 3138\nOutcome: incorrect\n\nTrial 3:\nProblem: What is 1062 minus 478?\nStudent answer: 484\nCorrect answer: 584\nOutcome: incorrect\n</student_history>\n",
        "queries": [
          {
            "problem": "Calculate: 1904 - 337",
            "targets": {
              "donor": "1573",
              "recipient": "1537",
              "same_donor": "1573",
              "random": "1537"
            },
            "swap": "<predicted_response>1567</predicted_response>",
            "random": "<predicted_response>1567</predicted_response>"
          },
          {
            "problem": "Calculate: 9025 - 8147",
            "targets": {
              "donor": "938",
              "recipient": "178",
              "same_donor": "938",
              "random": "178"
            },
            "swap": "<predicted_response>878</predicted_response>",
            "random": "<predicted_response>878</predicted_response>"
          },
          {
            "problem": "What is 8002 minus 3374?",
            "targets": {
              "donor": "4772",
              "recipient": "4378",
              "same_donor": "4772",
              "random": "4378"
            },
            "swap": "<predicted_response>4628</predicted_response>",
            "random": "<predicted_response>4628</predicted_response>"
          }
        ]
      }
    ]
  },
  "A": {
    "first_success": [
      {
        "group_id": "test:fractions:000:0",
        "pair_id": "test:fractions:000",
        "donor_rule": "fractions.ignore_denominators",
        "donor_history": "<student_history>\nTrial 1:\nProblem: Calculate (3/25) * (18/25).\nStudent answer: 54\nCorrect answer: 54/625\nOutcome: incorrect\n\nTrial 2:\nProblem: Multiply: 15/28 × 26/28\nStudent answer: 390\nCorrect answer: 195/392\nOutcome: incorrect\n\nTrial 3:\nProblem: Calculate (16/19) * (15/19).\nStudent answer: 240\nCorrect answer: 240/361\nOutcome: incorrect\n</student_history>\n",
        "recipient_history": "<student_history>\nTrial 1:\nProblem: Calculate (18/19) * (15/19).\nStudent answer: 270/19\nCorrect answer: 270/361\nOutcome: incorrect\n\nTrial 2:\nProblem: What is 3/36 times 25/36?\nStudent answer: 25/12\nCorrect answer: 25/432\nOutcome: incorrect\n\nTrial 3:\nProblem: Calculate (14/21) * (4/21).\nStudent answer: 8/3\nCorrect answer: 8/63\nOutcome: incorrect\n</student_history>\n",
        "queries": [
          {
            "problem": "Calculate (11/29) * (14/29).",
            "targets": {
              "donor": "154",
              "recipient": "154/29",
              "same_donor": "154",
              "random": "154/29"
            },
            "swap": "<predicted_response>154</predicted_response>",
            "random": "<predicted_response>154/29</predicted_response>"
          },
          {
            "problem": "What is 18/30 times 13/30?",
            "targets": {
              "donor": "234",
              "recipient": "39/5",
              "same_donor": "234",
              "random": "39/5"
            },
            "swap": "<predicted_response>234</predicted_response>",
            "random": "<predicted_response>19.5/15</predicted_response>"
          },
          {
            "problem": "What is 5/24 times 9/24?",
            "targets": {
              "donor": "45",
              "recipient": "15/8",
              "same_donor": "45",
              "random": "15/8"
            },
            "swap": "<predicted_response>45</predicted_response>",
            "random": "<predicted_response>15/8</predicted_response>"
          }
        ]
      }
    ],
    "first_failure": [
      {
        "group_id": "test:subtraction:000:0",
        "pair_id": "test:subtraction:000",
        "donor_rule": "subtraction.stops_borrow_at_zero",
        "donor_history": "<student_history>\nTrial 1:\nProblem: Subtract 8647 from 8804.\nStudent answer: 163\nCorrect answer: 157\nOutcome: incorrect\n\nTrial 2:\nProblem: What is 2082 minus 1485?\nStudent answer: 617\nCorrect answer: 597\nOutcome: incorrect\n\nTrial 3:\nProblem: Subtract 1814 from 3601.\nStudent answer: 1793\nCorrect answer: 1787\nOutcome: incorrect\n</student_history>\n",
        "recipient_history": "<student_history>\nTrial 1:\nProblem: What is 3015 minus 2575?\nStudent answer: 540\nCorrect answer: 440\nOutcome: incorrect\n\nTrial 2:\nProblem: Calculate: 4027 - 889\nStudent answer: 3838\nCorrect answer: 3138\nOutcome: incorrect\n\nTrial 3:\nProblem: What is 1062 minus 478?\nStudent answer: 484\nCorrect answer: 584\nOutcome: incorrect\n</student_history>\n",
        "queries": [
          {
            "problem": "Calculate: 1904 - 337",
            "targets": {
              "donor": "1573",
              "recipient": "1537",
              "same_donor": "1573",
              "random": "1537"
            },
            "swap": "<predicted_response>1573</predicted_response>",
            "random": "<predicted_response>1573</predicted_response>"
          },
          {
            "problem": "Calculate: 9025 - 8147",
            "targets": {
              "donor": "938",
              "recipient": "178",
              "same_donor": "938",
              "random": "178"
            },
            "swap": "<predicted_response>888</predicted_response>",
            "random": "<predicted_response>828</predicted_response>"
          },
          {
            "problem": "What is 8002 minus 3374?",
            "targets": {
              "donor": "4772",
              "recipient": "4378",
              "same_donor": "4772",
              "random": "4378"
            },
            "swap": "<predicted_response>4632</predicted_response>",
            "random": "<predicted_response>4732</predicted_response>"
          }
        ]
      }
    ]
  },
  "B": {
    "first_success": [
      {
        "group_id": "test:fractions:000:0",
        "pair_id": "test:fractions:000",
        "donor_rule": "fractions.ignore_denominators",
        "donor_history": "<student_history>\nTrial 1:\nProblem: Calculate (3/25) * (18/25).\nStudent answer: 54\nCorrect answer: 54/625\nOutcome: incorrect\n\nTrial 2:\nProblem: Multiply: 15/28 × 26/28\nStudent answer: 390\nCorrect answer: 195/392\nOutcome: incorrect\n\nTrial 3:\nProblem: Calculate (16/19) * (15/19).\nStudent answer: 240\nCorrect answer: 240/361\nOutcome: incorrect\n</student_history>\n",
        "recipient_history": "<student_history>\nTrial 1:\nProblem: Calculate (18/19) * (15/19).\nStudent answer: 270/19\nCorrect answer: 270/361\nOutcome: incorrect\n\nTrial 2:\nProblem: What is 3/36 times 25/36?\nStudent answer: 25/12\nCorrect answer: 25/432\nOutcome: incorrect\n\nTrial 3:\nProblem: Calculate (14/21) * (4/21).\nStudent answer: 8/3\nCorrect answer: 8/63\nOutcome: incorrect\n</student_history>\n",
        "queries": [
          {
            "problem": "Calculate (11/29) * (14/29).",
            "targets": {
              "donor": "154",
              "recipient": "154/29",
              "same_donor": "154",
              "random": "154/29"
            },
            "swap": "<predicted_response>154</predicted_response>",
            "random": "<predicted_response>154/29</predicted_response>"
          },
          {
            "problem": "What is 18/30 times 13/30?",
            "targets": {
              "donor": "234",
              "recipient": "39/5",
              "same_donor": "234",
              "random": "39/5"
            },
            "swap": "<predicted_response>234</predicted_response>",
            "random": "<predicted_response>19.5/15</predicted_response>"
          },
          {
            "problem": "What is 5/24 times 9/24?",
            "targets": {
              "donor": "45",
              "recipient": "15/8",
              "same_donor": "45",
              "random": "15/8"
            },
            "swap": "<predicted_response>45</predicted_response>",
            "random": "<predicted_response>15/8</predicted_response>"
          }
        ]
      }
    ],
    "first_failure": [
      {
        "group_id": "test:subtraction:000:0",
        "pair_id": "test:subtraction:000",
        "donor_rule": "subtraction.stops_borrow_at_zero",
        "donor_history": "<student_history>\nTrial 1:\nProblem: Subtract 8647 from 8804.\nStudent answer: 163\nCorrect answer: 157\nOutcome: incorrect\n\nTrial 2:\nProblem: What is 2082 minus 1485?\nStudent answer: 617\nCorrect answer: 597\nOutcome: incorrect\n\nTrial 3:\nProblem: Subtract 1814 from 3601.\nStudent answer: 1793\nCorrect answer: 1787\nOutcome: incorrect\n</student_history>\n",
        "recipient_history": "<student_history>\nTrial 1:\nProblem: What is 3015 minus 2575?\nStudent answer: 540\nCorrect answer: 440\nOutcome: incorrect\n\nTrial 2:\nProblem: Calculate: 4027 - 889\nStudent answer: 3838\nCorrect answer: 3138\nOutcome: incorrect\n\nTrial 3:\nProblem: What is 1062 minus 478?\nStudent answer: 484\nCorrect answer: 584\nOutcome: incorrect\n</student_history>\n",
        "queries": [
          {
            "problem": "Calculate: 1904 - 337",
            "targets": {
              "donor": "1573",
              "recipient": "1537",
              "same_donor": "1573",
              "random": "1537"
            },
            "swap": "<predicted_response>1573</predicted_response>",
            "random": "<predicted_response>1573</predicted_response>"
          },
          {
            "problem": "Calculate: 9025 - 8147",
            "targets": {
              "donor": "938",
              "recipient": "178",
              "same_donor": "938",
              "random": "178"
            },
            "swap": "<predicted_response>888</predicted_response>",
            "random": "<predicted_response>828</predicted_response>"
          },
          {
            "problem": "What is 8002 minus 3374?",
            "targets": {
              "donor": "4772",
              "recipient": "4378",
              "same_donor": "4772",
              "random": "4378"
            },
            "swap": "<predicted_response>4632</predicted_response>",
            "random": "<predicted_response>4732</predicted_response>"
          }
        ]
      }
    ]
  }
}
```


Per-donor-rule results:

| Model | Donor rule | Condition | Donor hits | Invalid |
| --- | --- | --- | --- | --- |
| baseline | subtraction.stops_borrow_at_zero | donor_native | 1/9 (11.1%) | 0 |
| baseline | subtraction.stops_borrow_at_zero | swap | 1/9 (11.1%) | 0 |
| baseline | subtraction.stops_borrow_at_zero | random | 1/9 (11.1%) | 0 |
| baseline | subtraction.stops_borrow_at_zero | oracle | 0/9 (0.0%) | 0 |
| baseline | subtraction.diff_0_n_equals_n | donor_native | 0/12 (0.0%) | 0 |
| baseline | subtraction.diff_0_n_equals_n | swap | 0/12 (0.0%) | 0 |
| baseline | subtraction.diff_0_n_equals_n | random | 1/12 (8.3%) | 0 |
| baseline | subtraction.diff_0_n_equals_n | oracle | 0/12 (0.0%) | 0 |
| baseline | subtraction.smaller_from_larger | donor_native | 0/18 (0.0%) | 0 |
| baseline | subtraction.smaller_from_larger | swap | 0/18 (0.0%) | 0 |
| baseline | subtraction.smaller_from_larger | random | 0/18 (0.0%) | 0 |
| baseline | subtraction.smaller_from_larger | oracle | 1/18 (5.6%) | 0 |
| baseline | subtraction.borrow_no_decrement | donor_native | 0/9 (0.0%) | 0 |
| baseline | subtraction.borrow_no_decrement | swap | 0/9 (0.0%) | 0 |
| baseline | subtraction.borrow_no_decrement | random | 0/9 (0.0%) | 0 |
| baseline | subtraction.borrow_no_decrement | oracle | 1/9 (11.1%) | 0 |
| baseline | fractions.ignore_denominators | donor_native | 1/24 (4.2%) | 0 |
| baseline | fractions.ignore_denominators | swap | 1/24 (4.2%) | 0 |
| baseline | fractions.ignore_denominators | random | 0/24 (0.0%) | 3 |
| baseline | fractions.ignore_denominators | oracle | 14/24 (58.3%) | 0 |
| baseline | fractions.keep_common_denominator_for_multiplication | donor_native | 2/24 (8.3%) | 3 |
| baseline | fractions.keep_common_denominator_for_multiplication | swap | 2/24 (8.3%) | 3 |
| baseline | fractions.keep_common_denominator_for_multiplication | random | 1/24 (4.2%) | 0 |
| baseline | fractions.keep_common_denominator_for_multiplication | oracle | 19/24 (79.2%) | 0 |
| A | subtraction.stops_borrow_at_zero | donor_native | 4/9 (44.4%) | 0 |
| A | subtraction.stops_borrow_at_zero | swap | 4/9 (44.4%) | 0 |
| A | subtraction.stops_borrow_at_zero | random | 1/9 (11.1%) | 0 |
| A | subtraction.stops_borrow_at_zero | oracle | 1/9 (11.1%) | 0 |
| A | subtraction.diff_0_n_equals_n | donor_native | 3/12 (25.0%) | 0 |
| A | subtraction.diff_0_n_equals_n | swap | 3/12 (25.0%) | 0 |
| A | subtraction.diff_0_n_equals_n | random | 2/12 (16.7%) | 0 |
| A | subtraction.diff_0_n_equals_n | oracle | 2/12 (16.7%) | 0 |
| A | subtraction.smaller_from_larger | donor_native | 2/18 (11.1%) | 0 |
| A | subtraction.smaller_from_larger | swap | 2/18 (11.1%) | 0 |
| A | subtraction.smaller_from_larger | random | 1/18 (5.6%) | 0 |
| A | subtraction.smaller_from_larger | oracle | 6/18 (33.3%) | 0 |
| A | subtraction.borrow_no_decrement | donor_native | 2/9 (22.2%) | 0 |
| A | subtraction.borrow_no_decrement | swap | 1/9 (11.1%) | 0 |
| A | subtraction.borrow_no_decrement | random | 2/9 (22.2%) | 0 |
| A | subtraction.borrow_no_decrement | oracle | 6/9 (66.7%) | 0 |
| A | fractions.ignore_denominators | donor_native | 24/24 (100.0%) | 0 |
| A | fractions.ignore_denominators | swap | 24/24 (100.0%) | 0 |
| A | fractions.ignore_denominators | random | 0/24 (0.0%) | 1 |
| A | fractions.ignore_denominators | oracle | 24/24 (100.0%) | 0 |
| A | fractions.keep_common_denominator_for_multiplication | donor_native | 22/24 (91.7%) | 0 |
| A | fractions.keep_common_denominator_for_multiplication | swap | 22/24 (91.7%) | 0 |
| A | fractions.keep_common_denominator_for_multiplication | random | 0/24 (0.0%) | 0 |
| A | fractions.keep_common_denominator_for_multiplication | oracle | 21/24 (87.5%) | 0 |
| B | subtraction.stops_borrow_at_zero | donor_native | 4/9 (44.4%) | 0 |
| B | subtraction.stops_borrow_at_zero | swap | 4/9 (44.4%) | 0 |
| B | subtraction.stops_borrow_at_zero | random | 1/9 (11.1%) | 0 |
| B | subtraction.stops_borrow_at_zero | oracle | 1/9 (11.1%) | 0 |
| B | subtraction.diff_0_n_equals_n | donor_native | 3/12 (25.0%) | 0 |
| B | subtraction.diff_0_n_equals_n | swap | 3/12 (25.0%) | 0 |
| B | subtraction.diff_0_n_equals_n | random | 3/12 (25.0%) | 0 |
| B | subtraction.diff_0_n_equals_n | oracle | 2/12 (16.7%) | 0 |
| B | subtraction.smaller_from_larger | donor_native | 2/18 (11.1%) | 0 |
| B | subtraction.smaller_from_larger | swap | 2/18 (11.1%) | 0 |
| B | subtraction.smaller_from_larger | random | 2/18 (11.1%) | 0 |
| B | subtraction.smaller_from_larger | oracle | 6/18 (33.3%) | 0 |
| B | subtraction.borrow_no_decrement | donor_native | 1/9 (11.1%) | 0 |
| B | subtraction.borrow_no_decrement | swap | 1/9 (11.1%) | 0 |
| B | subtraction.borrow_no_decrement | random | 2/9 (22.2%) | 0 |
| B | subtraction.borrow_no_decrement | oracle | 6/9 (66.7%) | 0 |
| B | fractions.ignore_denominators | donor_native | 24/24 (100.0%) | 0 |
| B | fractions.ignore_denominators | swap | 24/24 (100.0%) | 0 |
| B | fractions.ignore_denominators | random | 0/24 (0.0%) | 1 |
| B | fractions.ignore_denominators | oracle | 24/24 (100.0%) | 0 |
| B | fractions.keep_common_denominator_for_multiplication | donor_native | 22/24 (91.7%) | 0 |
| B | fractions.keep_common_denominator_for_multiplication | swap | 22/24 (91.7%) | 0 |
| B | fractions.keep_common_denominator_for_multiplication | random | 0/24 (0.0%) | 0 |
| B | fractions.keep_common_denominator_for_multiplication | oracle | 21/24 (87.5%) | 0 |


Training integrity and replay counts:

| Arm | Unique steps | Logged updates | Replayed steps |
| --- | --- | --- | --- |
| A | 240 | 274 | {"41": 2, "42": 2, "43": 2, "44": 2, "45": 2, "46": 2, "47": 2, "48": 2, "49": 2, "51": 2, "52": 2, "53": 2, "54": 2, "55": 2, "56": 2, "57": 2, "58": 2, "59": 2, "60": 2, "61": 2, "62": 2, "63": 2, "64": 2, "65": 2, "66": 2, "67": 2, "68": 2, "69": 2, "70": 2, "71": 2, "72": 2, "73": 2, "74": 2, "75": 2} |
| B | 240 | 240 | {} |


## 12. Interpretation and limits

**Causal-control caveat:** own-state replacement differs from native output on 2 model-query comparisons in total. The pipeline is not exactly no-op invariant. Do not claim zero intervention artifact; quantization/numerical sensitivity versus implementation mismatch remains to be distinguished. All such cases are retained in primary metrics.

The exploratory interval for B − A includes zero. This pilot does not establish an incremental specificity benefit of exchange supervision over matched SFT.

Even a positive D is limited to the known rule panel, fresh numbers and histories, this checkpoint and decoder. Native/oracle/joint-three performance and no-op validity must be considered alongside it. This does not establish unseen-rule compositional execution, recurrent self-updating, or human fidelity. A positive result is repair-training evidence, not retroactive proof about the old model.


## 13. Reproduction and provenance

Experiment `c4-execution-training-v1-seed42-recovery1-20260906`; A job `stoic_queen_l2zrtkcpcq`, B `tidy_card_gdqp59xzfl`; Feeds-LLM Standard, 1×80G1-A100 each, UAI rankfun_aml, original primary AML workspace. Requested image alias and actual resolved image differ; see recovery receipt for both identities. Original startup failure and A platform cancellation/automatic retry are retained in `reports/c4_execution_training_recovery1_20260906.md`.

Inputs/protocol: `manifests/c4_execution_training_v1_20260906.json`, `reports/c4_execution_training_protocol_20260906.md`; frozen code hashes: `manifests/c4_execution_training_v1_recovery1_20260906.sha256`. Collection: `scripts/collect_c4_execution_training.py`; full audit: `scripts/audit_c4_execution_delivery.py --tokenizer /home/v-yuejiali/studentmodeling/models/Qwen3-32B`; report: `scripts/render_c4_execution_results.py`. Full audit requires all 2,880 unique records and raw token/text equality. Checkpoints remain in remote storage.
