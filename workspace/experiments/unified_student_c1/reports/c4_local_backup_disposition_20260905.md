# Local backup disposition

The third remote shard remained queued for approximately 49 minutes. A local
backup preflight was prepared on a newly free A100, using already-present base
weights, a hash-verified 269 MB adapter download, and an isolated Python 3.10 /
PyTorch 2.8.0+cu126 environment under `/mnt/codex-tmp`. The main remote evaluator,
manifest and code hashes were not modified.

At 17:42 UTC the original remote shard transitioned to Running. The local
backup was still loading weight shards (3/17, no predictions or state capture
yet) and was explicitly stopped to avoid redundant evaluation. No local
backup output enters the scientific analysis. The temporary environment and
adapter are retained; no other project's process or files were modified.

Authoritative results remain the three original remote jobs in
`c4-rule-followup-seed42-v2-20260905`. No cancellation/resubmission of those jobs
was performed in this continuation. Local status is amended as stopped in its
receipt; the runtime setup is not reported as an experimental replication.
