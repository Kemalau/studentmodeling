#!/usr/bin/env bash
set -euo pipefail

# Reclaim only disposable local copies after their read-only probes finish.
# Remote atomic checkpoints remain authoritative and are never modified.
project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_strong1_widen8_bridge_20260808/single_milestone_c4"
t3="$project/results/state8_strong1_widen8_bridge_20260808/single_step250_t3"
remote_root=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4
account=csnewsandfeeds4150361735
container=amulet
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
steps=(250 1000 2000 3000 4000 5000)

mkdir -p "$state/local_cleanup"
cd "$project"
exec 9>"$state/local_cleanup/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/local_cleanup/controller.active.pid"
trap 'rm -f "$state/local_cleanup/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/local_cleanup/controller.heartbeat"; }

for step in "${steps[@]}"; do
  root="$state/step-$step"
  marker="$state/local_cleanup/step-$step.complete"
  while [[ ! -f "$marker" ]]; do
    heartbeat
    if [[ ! -s "$root/summary.json" || ! -s "$root/artifact_audit.json" ]]; then
      sleep 60; continue
    fi
    if [[ "$step" == 250 && ! -f "$t3/complete" ]]; then
      sleep 60; continue
    fi
    remote="$remote_root/checkpoint-$step"
    remote_receipt="$(az storage blob exists --auth-mode login \
      --account-name "$account" --container-name "$container" \
      --name "$remote/checkpoint_completion.json" --query exists -o tsv 2>/dev/null || true)"
    remote_state="$(az storage blob exists --auth-mode login \
      --account-name "$account" --container-name "$container" \
      --name "$remote/recurrent_state.pt" --query exists -o tsv 2>/dev/null || true)"
    [[ "$remote_receipt" == true && "$remote_state" == true ]] \
      || { sleep 60; continue; }
    "$python_bin" - "$root/artifact_audit.json" "$root/summary.json" "$step" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); step=int(sys.argv[3])
assert a['stage']=='state8_strong1_single_milestone_artifact_audit_v1',a
assert a['status']=='passed' and a['step']==step,a
assert a['sha256']==s['artifact_sha256'],(a,s)
assert s['stage']=='state8_strong1_single_milestone_c4_v1' and s['step']==step,s
PY
    before="$(du -sb "$root" | cut -f1)"
    rm -f "$root/checkpoint-$step/recurrent_state.pt" \
      "$root/checkpoint-$step/adapter/adapter_model.safetensors" \
      "$root/c4/progress/state_cache.pt" \
      "$root/c4/progress/state_cache.pt.partial"
    after="$(du -sb "$root" | cut -f1)"
    "$python_bin" - "$root/local_cleanup_receipt.json" "$step" "$before" "$after" \
      "$remote" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); before=int(sys.argv[3]); after=int(sys.argv[4])
out={'stage':'state8_local_milestone_cleanup_v1','status':'complete',
 'step':int(sys.argv[2]),'bytes_before':before,'bytes_after':after,
 'bytes_reclaimed':before-after,'remote_checkpoint':sys.argv[5],
 'remote_authoritative_artifacts_verified_present':True,
 'local_metrics_and_hash_receipts_preserved':True}
q=p.with_suffix('.json.partial'); q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); q.replace(p)
PY
    touch "$marker"
  done
done
touch "$state/local_cleanup/complete"; heartbeat
