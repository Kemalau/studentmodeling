# Unified Student Benchmark Task Taxonomy v2

This is the authoritative public numbering from 2026-08-13 onward. Internal
JSON task keys are unchanged to preserve historical manifests, checkpoint
hashes, and running Feeds jobs.

| Public label | Stable internal task key | Capability | Evidence role | Legacy label |
|---|---|---|---|---|
| T1 | `predict_correctness` | Predict whether the student answers correctly | Headline | T1 |
| T2 | `predict_response` | Predict the concrete choice or mathematical response | Headline | T2 |
| T3 | `diagnose_misconception` | Diagnose the error mechanism | Headline | T3 |
| T4 | `predict_prospective_error` | Predict behavior on a specified future item | Headline | T5 |
| T5 | `predict_error_persistence` | Predict whether the observed error recurs | Headline secondary | T6 |
| T6 | `estimate_mastery` | Predict a future-KC-performance band | Appendix proxy | T4 |

The headline progression is therefore continuous:

`current correctness -> current response -> diagnosis -> future behavior -> persistence`.

T6 is not called latent mastery. It is an appendix-only, future-performance
proxy. Historical v1 results remain immutable and must be translated through
`legacy_to_public = {T1:T1, T2:T2, T3:T3, T4:T6, T5:T4, T6:T5}`.

