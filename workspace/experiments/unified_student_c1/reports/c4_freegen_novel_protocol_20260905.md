# Frozen free-generation rule-transfer audit (2026-09-05)

## Question and non-goals

Does replacing the state trajectory cause open generation of the donor-rule
answer on new numerical combinations? This follow-up tests the historical
CF1 vision-pad seed-42 checkpoint; it is not a new training run or a test of
recurrent closure. A positive effect supports executable-rule transfer by the
state-plus-decoder system. It does not establish an autonomous algorithm inside
the state vector or eliminate unknown base-model pretraining exposure.

## Sample construction

Source: frozen 150-pair seed-42 C4 manifest. Take the first occurrence of each
distinct donor, up to three donors per malrule in manifest order, without using
any model predictions. This selects 29 donors. Generate new problems with the
original recipient query's generator, level 9, MEDIUM difficulty, RNG 20260905.
Retain the first three qualifying scalar-numeric queries per donor. Two donors
(original pair indices 83 and 98) have fewer than three eligible queries after
10,000 attempts; report their complete filtering counts. Final panel: 27 donors,
81 queries, 11 donor rules (18 subtraction, 7 fractions, 2 decimals donors).

Exact input row counts and SHA256s are in the manifest's source_lineage. They
cover the actual CF1 train and valid data plus original train, val, same-template
test, and cross-template test data. Their queries AND histories enter the
novelty index. Reject unordered canonical operand or rendered-number
combinations already seen there, including numeric-equivalent spellings.
The actual CF1 train SHA256 is checked against its historical training recipe.
The rule-library revision and Python-source hashes are frozen in the manifest.

Use original executable algorithms to compute A, B, and correct answers. Require
agreement among applicable correct algorithms, consistency with metadata, and
three different A/B/correct answers. Require A's three query answers to differ.
Exclude A and B target numbers appearing anywhere numerically in any of the
four histories (donor, recipient, same-rule, random). Thus donor-answer exclusion
is included, with a stricter numeric-equivalence check. Record whether each A
answer appeared as any fine-tuning target; 41 of 81 did not. This does not imply
that every individual digit, answer string, or problem template is unseen.

Concrete construction example, original pair 5: A divides by multiplying
without inverting. Its fixed history is reused with these new queries:

| Query | A-rule answer | B-rule answer | Correct answer |
|---|---:|---:|---:|
| 2 5/8 ÷ 1 1/2 | 63/16 | 7 | 7/4 |
| 2 7/10 ÷ 2 1/2 | 27/4 | 5 | 27/25 |
| 3 9/10 ÷ 2 1/2 | 39/4 | 7 | 39/25 |

These are constructed targets, not measured model successes.

## Coordinate, formula, and selection

No coordinate discovery or layer search is performed. Reuse all 64 decoder
block output residuals at the single state-token address (64 × 5120). Capture
each source once using only the instruction and history through that address;
the future query is absent. For every new query replace
`h_B[layer, state] <- h_A[layer, state]`. The same tensors are reused for all
three queries. The decoder retains the original late-fusion attention mask.
Cached greedy decoding and full recomputation are checked on a tiny Qwen3 model
with and without patches. Each condition/query starts a new KV cache.

Arithmetic illustration: 21/8 × 3/2 = 63/16 is the donor-rule target for the
first query, while ordinary division gives 7/4. No attribution gradient or
coordinate ranking is involved, so gradient-sign examples are inapplicable.

## Held-out causal validation and generation

Five conditions on exactly the same queries: native/no patch, own-state no-op,
directed A, real-random donor, same-rule donor. Real-random identities match the
previous frozen control and differ from A and B in malrule identity, within the
same family. They are not guaranteed applicable to the particular operation:
only 33 of 81 queries have an executable random-rule target. Therefore report
that applicability subset separately, including its native-B subset; the broad
family-matched contrast alone cannot eliminate an operation-mismatch explanation.

Full original histories are present but hidden from answer attention except
through the state route. Historical correct-answer feedback is retained.
The query contains only `Problem: ...`; it contains no candidate options or
correct answer. Keep the original output instruction. Greedy decoding, no
answer prefill, no constrained vocabulary, maximum 48 generated tokens, stop at
EOS or the closing predicted_response tag. No sampling or multiple attempts.
No truncation is allowed; overlength prompts fail visibly.

Parse exactly one numeric predicted_response tag or a bare numeric response.
Compare exact rational values (e.g. 56/30 equals 28/15); do not select a matching
number from a response containing multiple guesses. Malformed, nonnumeric,
reasoning-only and incomplete responses fail and stay in the denominator.
Store raw text, token IDs, parse status, exact targets and all matched categories.

## Aggregate and paired endpoints

Primary: free-generation donor-target agreement difference
`D_open = mean(I[swap=A] - I[random_real=A])` over all 81 fixed queries.
Secondary: the same difference on queries whose native generation equals B.
Report both denominators, all five rates, and 10,000-replicate paired bootstrap
95% CIs resampling donor groups, keeping each donor's three queries together.
Also report random-rule-applicable subsets, target-string-unseen subset,
per-family contrasts, native B accuracy, invalid rates, no-op disagreements,
and counts of donors with all three distinct A targets generated correctly.
Do not compare these rates directly to candidate-scoring 0.411.

No gate is retrofitted from the results. A positive lower CI is evidence against
zero mean effect on this constructed panel, not by itself proof of universally
executable rules. Null/negative contrasts, low applicability-subset effects,
native-generation collapse, or no-op mismatches constrain interpretation.
No new model results are available at protocol freeze. Report representative
successes, failures and malformed generations after the run completes.

## Reproduction and execution

- Builder/scorer: scripts/c4_freegen_protocol.py
- Generator: scripts/evaluate_c4_freegen_novel.py
- Frozen inputs: manifests/c4_freegen_novel_seed42_20260905.json
- Code integrity: manifests/c4_freegen_novel_20260905.sha256
- Launcher: scripts/run_c4_freegen_novel.sh
- AMLT: amlt/feeds_c4_freegen_novel_seed42_20260905.yaml
- Experiment: c4-freegen-novel-seed42-v1-20260905 (check live status for acceptance).
- Three disjoint group_idx modulo 3 shards, one 80G1-A100 Standard each, Feeds.
- UAI: rankfun_aml; approved image reused from the completed dose-response job:
  amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258.
- Model: /mnt/amulet/studentmodeling/models/Qwen3-32B.
- Adapter: /mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter;
  weights SHA256 c9b463ed12f11ef84d346c1399d2588b9b396ecb9e15c5f4ca85c7f1d7fe991f.
- Remote output: each job's AMLT_OUTPUT_DIR/freegen, with atomic per-group
  files and exact-lineage resume validation. Merge rejects gaps and duplicates.

After downloading all three shards, run c4_freegen_protocol.py summarize with
the manifest, all three records.jsonl paths, and --output for the summary JSON.

## Submission receipt

Submitted successfully on 2026-09-05 around 11:04 UTC. Last directly observed
snapshot during setup: shard1 running; shard0 and shard2 queued. Job IDs:

- shard0: bubbly_cumin_q5m9r5l9w2
- shard1: affable_dog_b84t7wcr3m
- shard2: shy_snake_3fzqzvqnh1

Remote code package: projects/v-yuejiali/amlt-code/f64a961fb2a697586dfdda7a89f8bb4bed7264784abe845fa6f1dced396fe518.
Its 10-file listing was checked; no weights or unrelated project files uploaded.
Local preflight: 4 protocol tests plus 1 tiny-Qwen cached/full-decode equivalence
test passed, 9 frozen file hashes passed, no truncation (213–325 prompt tokens,
48 generation budget), 81-query integrity check passed. Remote startup repeats
the tests before loading the model. Results were not yet available at this entry.

Read-only monitor: scripts/watch_c4_freegen_novel.py. It polls status every 60s,
downloads after all three jobs pass, rejects missing/duplicate records, and writes
results/c4_freegen_novel_seed42_20260905.summary.json. Status and monitor-state
JSONs use the same prefix. Failed jobs are marked needs_attention; the monitor
never silently resubmits, changes the frozen panel, or chooses another checkpoint.
