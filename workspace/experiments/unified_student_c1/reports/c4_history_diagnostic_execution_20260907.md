# C4 history diagnosis: Feeds-LLM execution receipt

Submission began at 03:52:41 UTC and was confirmed by 03:54 UTC on 2026-09-07. This records
execution, not a model-performance result. The user requested trying the
history-insufficiency diagnosis after authorizing Feeds-LLM execution.

Experiment: `c4-history-diagnostic-v1-seed42-feedsllm-20260907`.
Eight independent Standard `80G1-A100` jobs, one GPU each, four-hour limit,
inherited two-attempt infrastructure policy. No training or checkpoint change.
Initial authoritative receipt: seven preparing, one queued; all eight accepted.
At 03:53:54 UTC the native monitor observed all eight queued; no generation
result was available. Collector/native-watch PIDs were 1622346/1622347.

| Checkpoint | History cohort | Job ID |
|---|---|---|
| A | h3 | sharp_pot_jv4sl3xkr9 |
| A | h6 | busy_curtain_vn3wmsy2rx |
| A | h12 | keen_giraffe_tr5wkts5nd |
| A | repeat3x4 | plucky_skin_d599b0p4hr |
| B | h3 | stoic_pizza_2p55pnb4wf |
| B | h6 | tender_horse_ns1fdt528c |
| B | h12 | stoic_knot_837b650pbb |
| B | repeat3x4 | brave_pocket_sn3bt3kmf2 |

Selected target Feeds-LLM, verified via both AMLT quota and direct read-only
ARM GET in subscription 078ad34f-9e5b-4962-9d78-c9448646548e, resource group
feeds-llm-distillation (virtual-cluster API 2021-03-01-preview). Submission
workspace remains CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity
in primary subscription b6dc87f3-c479-49c8-8cb5-7896da3ff895; no subscription
or target change. The first direct ARM query used an unsupported API version;
the version named by the service returned HTTP 200.

Fresh quota at submission: Premium 140/144, Standard 20/80, Basic 0/80,
user_max null. This exposes no personal cap, not a guarantee of capacity.
Fresh project active-request audit found 18 A100 requests on Feeds-LLM before
this eight-GPU request. One unrelated zero-job experiment was resolved only
after refreshed metadata confirmed n_jobs=0 and all_terminated=true.

Approved inherited image:
`amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258`.
Explicit UAI rankfun_aml. Workspace identity lookup located it in AMLStudio,
not the workspace resource group; identity and Storage Blob Data
Contributor/Reader assignments on csnewsandfeeds4150361735 were verified.
Initial lookup at the wrong group returned AuthorizationFailed; no access
policy or identity was changed to resolve it.

Authentication recovered using the existing same-user cache
`/home/v-yuejiali/.azure-studentmodeling-crossmodel-submit-20260903`.
Identity, expected tenant/subscription, token refresh, workspace and storage
read access were verified. Canonical cache was empty and default caches had
a JSON parse failure. No cache was copied/merged/repaired, and no interactive
login or device-code flow was used.

Remote base: `/mnt/amulet/studentmodeling/models/Qwen3-32B`.
Remote A/B final adapter paths and SHA256 values are frozen in the manifest;
blob existence was checked, and locally preserved adapters match both hashes.
Outputs: each job's `AMLT_OUTPUT_DIR/diagnostic` on the existing project store,
with runner.log beside it. Only 16 source/manifest/protocol/hash files, about
2.2 MiB, were uploaded. No credentials, local model, or adapter was uploaded.

13 relevant CPU tests passed in 28.69 s. Checks include deterministic nested
histories, reverse-direction identity, old-number exclusion, no target leak,
all prompt lengths, both checkpoint hashes, missing/duplicate/corrupt-output
rejection, synthetic paired contrasts and fail-closed quota handling. Python
compilation, bash syntax, full parsed AMLT YAML and frozen checksums passed.
All conditions fit the original 768-token cap; longest h12 input is 630 tokens.

Native monitor/collector: `scripts/watch_c4_history_diagnostic.py`. It waits
for terminal job status via `amlt watch --notify terminal`, retrieves only
JSON/logs, and requires every one of 5,376 records plus matching lineage before
declaring complete. It does not train, resubmit or exclude failed outputs.
Failure/observation-error states are distinct from negative scientific results.

Submission log and current job IDs/quota:
`results/c4_history_diagnostic_v1_20260907.submission.log` and `.submission.json`.
Live execution status: `results/c4_history_diagnostic_v1_20260907.status.json`.
Expected audited summary: `results/c4_history_diagnostic_v1_20260907.summary.json`.
Expected report: `reports/c4_history_diagnostic_results_20260907.md`.
Frozen scientific protocol: `reports/c4_history_diagnostic_protocol_20260907.md`.

The neuron-discovery-reporting audit requirements shaped the protocol: explicit
sample construction, fixed interface and scoring formula, retained negative
controls, paired denominators and limits on causal interpretation. The hybrid
routing skills keep source/data checks local and independent inference shards
on the user-selected cluster.
