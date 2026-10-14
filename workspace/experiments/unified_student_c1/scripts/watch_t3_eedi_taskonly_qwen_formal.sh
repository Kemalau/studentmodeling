#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/t3_eedi_taskonly_qwen_formal_20260813"
experiment=t3-eedi-taskonly-qwen-formal-recovery1-20260813
job=qwen3-32b-t3-eedi-taskonly-formal-open-seed42
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
mkdir -p "$state"; cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  if [[ "$STUDENTMODELING_FEEDS_AUTH_HEALTH" == unhealthy ]]; then
    printf '%s\tAUTH_OR_STATUS_RETRY\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state/latest_status.tsv"
    sleep 60; continue
  fi
  status="$(timeout 90 amlt --json-tables status "$experiment" ":$job" \
    --hide-urls --force-status-update 2>/dev/null)" || { sleep 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  job_status="$(STATUS="$status" JOB="$job" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); name=os.environ['JOB']
matches=[x for x in rows if str(x.get('job_name') or '').lstrip(':')==name]
print(str(matches[0].get('status') or '').casefold() if len(matches)==1 else 'unknown')
PY
)"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_status" > "$state/latest_status.tsv"
  case "$job_status" in
    pass) break ;;
    paused)
      timeout 90 amlt resume "$experiment" ":$job" >> "$state/controller.log" 2>&1 || true
      ;;
    preparing|pre-queued|queued|running|resuming) ;;
    failed|killed|cancelled|canceled|stopped)
      timeout 180 amlt logs view -n 1000 "$experiment" ":$job" \
        > "$state/terminal.log" 2>&1 || true
      fail "submitted T3-Eedi formal job ended status=$job_status" ;;
  esac
  sleep 60
done

rm -rf "$state/download" "$state/validated"
mkdir -p "$state/download" "$state/validated"
timeout --foreground 900 amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" </dev/null \
  >> "$state/controller.log" 2>&1 \
  || fail "T3-Eedi formal result download failed"
mapfile -t metrics < <(find "$state/download" -type f -path '*/eval/t3_eedi_taskonly/metrics.json')
mapfile -t predictions < <(find "$state/download" -type f -path '*/eval/t3_eedi_taskonly/predictions.jsonl')
[[ ${#metrics[@]} -eq 1 && ${#predictions[@]} -eq 1 ]] \
  || fail "T3-Eedi result cardinality drift metrics=${#metrics[@]} predictions=${#predictions[@]}"
"$python_bin" - "${metrics[0]}" "${predictions[0]}" "$state/validated" <<'PY' \
  || fail "T3-Eedi formal result validation failed"
import hashlib,json,pathlib,shutil,sys
metrics_path,predictions_path,out=map(pathlib.Path,sys.argv[1:])
x=json.load(open(metrics_path,encoding='utf-8'))
group=x['metrics_by_group']['diagnose_misconception/eedi_mining/open_taxonomy_generation']
assert int(x['generation_rows'])==894 and int(group['n'])==894,x
assert int(group['llm_judge_n'])==894,group
assert x['artifact_sha256']['adapter_model']=='329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2',x
rows=[json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(rows)==894 and len({str(row['example_id']) for row in rows})==894,len(rows)
out.mkdir(parents=True,exist_ok=True)
shutil.copy2(metrics_path,out/'metrics.json')
shutil.copy2(predictions_path,out/'predictions.jsonl')
receipt={'status':'complete','stage':'t3_eedi_taskonly_qwen_formal_v1','n':894,
         'adapter_sha256':x['artifact_sha256']['adapter_model'],
         'predictions_sha256':hashlib.sha256(predictions_path.read_bytes()).hexdigest()}
(out/'completion.json').write_text(json.dumps(receipt,sort_keys=True)+'\n',encoding='utf-8')
PY
touch "$state/complete"
rm -f "$state/needs_attention"
heartbeat
log "T3-Eedi task-only formal baseline validated n=894"
