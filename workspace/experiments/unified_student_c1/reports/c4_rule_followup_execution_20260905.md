# C4 rule follow-up execution receipt

## Authoritative run

`c4-rule-followup-seed42-v2-20260905`, submitted with exit code 0 at approximately
2026-09-05 16:53 UTC. Three Standard `80G1-A100` jobs on Feeds through
CS-NewsAndFeeds-Singularity, UAI `rankfun_aml`, approved immutable image tag from
the completed prior C4 diagnostic. Last pre-submit A100 quota Standard 68/97,
Premium/Basic 0/0, user_max limit 32 (no live per-user usage exposed).

Code upload: exactly 16 allowlisted files, about 0.6 MB, remote package
`projects/v-yuejiali/amlt-code/6ad2f11f31d8b30ea343b25293344c476b3565e4e0daa4a3133583b20d91370a`.
Remote model and adapter paths are in the frozen protocol. No model weights
downloaded to the submission host. Outputs are `AMLT_OUTPUT_DIR/rule_followup`.

Preflight: manifest deterministic reconstruction; all 15 file hashes passed;
three protocol tests and one cached/full-decode test passed; all 1,071 prompts
are 78–412 tokens plus at most 48 generated tokens, without truncation; AMLT
config parse passed. All 23 menu donors pass history-answer exclusion including
new controls. Controls use identical response instructions and distinct histories.

Authentication used existing same-user cache
`/home/v-yuejiali/.azure-studentmodeling-borrowed-keyflip-20260903`, validated for
the expected tenant/subscription and live Feeds status. No device login or
credential copying was performed.

## Superseded pre-inference run

`c4-rule-followup-seed42-v1-20260905` had a control-pool implementation defect:
task-specific example IDs did not ensure distinct histories or matching response
instructions. No outputs were inspected. All jobs were queued when the defect
was discovered and all three were explicitly canceled; terminal status was
verified `killed`, durations reported 1 second. Job IDs were
`loyal_shirt_f5w3ybdp5f`, `polite_brake_n3vzthlqb7`, `frank_root_kqvs9c7c9b`.
The original immutable uploaded code package and v1 manifest remain retained;
v1 must never contribute scientific evidence. The separate v2 amendment records
the correction before new inference.

## Monitoring and completion

`scripts/watch_c4_rule_followup.py` invokes native `amlt watch` at 60-second
intervals with terminal-only notifications, then downloads and audits. It does
not submit or alter jobs. Local receipts use prefix
`results/c4_rule_followup_v2_seed42_20260905` (`.monitor.json`, `.monitor.log`,
`.watch.json`, `.summary.json`). The foreground assistant separately reviews
completion, raw records and interpretations before giving a conclusion.
