#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/t3_eedi_taskonly_qwen_formal_20260813"
config="$project/amlt/feeds_t3_eedi_taskonly_qwen_formal_sharded_recovery_20260814.yaml"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/formal_evaluation_v1/test.jsonl"
base_experiment=t3-eedi-taskonly-qwen-sharded-recovery1-20260814
recovery_prefix=t3-eedi-taskonly-qwen-sharded-recovery
recovery_suffix=-20260814
max_recovery_waves=16
jobs=(); for index in 0 1 2 3; do
  printf -v suffix '%02d' "$index"
  jobs+=("qwen3-32b-t3-eedi-taskonly-formal-shard$suffix")
done

mkdir -p "$state/sharded_preserved_results" "$state/validated" "$state/t3_semantic"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
rm -f "$state/needs_attention" "$state/complete"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

preserve_passed(){
  local experiment="$1" status
  source scripts/use_studentmodeling_feeds_auth.sh
  status="$(timeout --foreground 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" || return 1
  mapfile -t passed < <(STATUS="$status" "$python_bin" - <<'PY'
import json,os
for row in json.loads(os.environ['STATUS']):
    if str(row.get('status') or '').casefold() == 'pass':
        print(str(row['job_name']).lstrip(':'))
PY
  )
  (( ${#passed[@]} > 0 )) || return 0
  timeout --foreground 1800 amlt results download "$experiment" --status pass \
    --include '*.json' --include '*.jsonl' --include '*.txt' \
    --output "$state/sharded_preserved_results" </dev/null \
    >> "$state/controller.log" 2>&1 || return 1
  for job in "${passed[@]}"; do touch "$state/passed.$job"; done
}

submit_jobs(){
  local experiment="$1"; shift
  local names=("$@") selectors=() dump="$state/$experiment.preflight.yaml" output
  source scripts/use_studentmodeling_feeds_auth.sh
  for name in "${names[@]}"; do selectors+=(":$name"); done
  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$dump" \
    2> "$state/$experiment.preflight.stderr" || return 1
  "$python_bin" scripts/validate_t3_eedi_taskonly_qwen_sharded_amlt.py \
    --config "$dump" --jobs "${names[@]}" >> "$state/controller.log" 2>&1 || return 1
  output="$(amlt run "$config" "${selectors[@]}" "$experiment" -t Feeds -y \
    --description "Four-way deterministic recovery of frozen Eedi-only T3 specialist generation; same adapter, panel, decoding, and seed" 2>&1)" || {
      printf '%s\n' "$output" > "$state/$experiment.submission.txt"; return 1; }
  printf '%s\n' "$output" > "$state/$experiment.submission.txt"
  printf '%s\n' "$experiment" > "$state/sharded.active_experiment.txt"
  printf '%s\n' "${names[@]}" > "$state/sharded.active_jobs.txt"
  log "submitted Eedi specialist recovery experiment=$experiment jobs=${#names[@]}"
}

wave=1
experiment="$base_experiment"
if [[ -s "$state/sharded.active_experiment.txt" ]]; then
  experiment="$(<"$state/sharded.active_experiment.txt")"
  [[ "$experiment" =~ recovery([0-9]+)-20260814$ ]] && wave="${BASH_REMATCH[1]}"
else
  until submit_jobs "$experiment" "${jobs[@]}"; do heartbeat; sleep 120; done
fi

while true; do
  heartbeat
  source scripts/use_studentmodeling_feeds_auth.sh
  status="$(timeout --foreground 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json.partial"
  mv "$state/latest_status.json.partial" "$state/latest_status.json"
  "$python_bin" - "$status" "$state/latest_status.tsv" <<'PY'
import collections,json,pathlib,sys
rows=json.loads(sys.argv[1]); c=collections.Counter(str(r.get('status') or '').casefold() for r in rows)
active=sum(c[x] for x in ('preparing','pre-queued','queued','running','resuming','paused'))
status='sharded_running' if active else 'sharded_terminal'
pathlib.Path(sys.argv[2]).write_text(f"{status}\t{dict(c)}\n")
PY
  if "$python_bin" - "$status" <<'PY'
import json,sys
active={'preparing','pre-queued','queued','running','resuming','paused'}
raise SystemExit(0 if any(str(r.get('status') or '').casefold() in active for r in json.loads(sys.argv[1])) else 1)
PY
  then sleep 60; continue; fi
  preserve_passed "$experiment" || { sleep 60; continue; }
  missing=(); for job in "${jobs[@]}"; do [[ -f "$state/passed.$job" ]] || missing+=("$job"); done
  (( ${#missing[@]} == 0 )) && break
  (( wave < max_recovery_waves )) || fail "exhausted Eedi shard recoveries missing=${missing[*]}"
  wave=$((wave+1)); experiment="${recovery_prefix}${wave}${recovery_suffix}"
  until submit_jobs "$experiment" "${missing[@]}"; do heartbeat; sleep 120; done
done

rm -rf "$state/sharded_download" "$state/sharded_merge" "$state/t3_semantic"
mkdir -p "$state/sharded_download" "$state/sharded_merge" "$state/t3_semantic" "$state/validated"
cp -a "$state/sharded_preserved_results/." "$state/sharded_download/"
readarray -t predictions < <(find "$state/sharded_download" -type f -path '*/eval/t3_eedi_taskonly_shard*/predictions.jsonl' | sort)
readarray -t metrics < <(find "$state/sharded_download" -type f -path '*/eval/t3_eedi_taskonly_shard*/metrics.json' | sort)
[[ ${#predictions[@]} -eq 4 && ${#metrics[@]} -eq 4 ]] \
  || fail "Eedi shard artifact cardinality drift predictions=${#predictions[@]} metrics=${#metrics[@]}"
"$python_bin" scripts/merge_t3_eedi_taskonly_shards.py \
  --predictions "${predictions[@]}" --metrics "${metrics[@]}" \
  --eval-jsonl "$eval_jsonl" --output-dir "$state/sharded_merge" \
  >> "$state/controller.log" 2>&1 || fail "Eedi specialist shard merge failed"

[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
exec 11>"$state/t3_semantic/judge.lock"; flock -x 11
(while true; do heartbeat; sleep 60; done) & heartbeat_pid=$!
bash scripts/wait_for_trapi_auth.sh
attempt=0
while [[ ! -s "$state/t3_semantic/metrics.json" ]]; do
  attempt=$((attempt+1))
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$state/sharded_merge/predictions.jsonl" \
    --eval-jsonl "$eval_jsonl" --output-dir "$state/t3_semantic" \
    --judge-backend trapi --trapi-deployment gpt-4o_2024-11-20 \
    --judge-model gpt-5.5 --judge-workers 12 --judge-max-retries 12 \
    --judge-timeout-seconds 60 --judge-cache "$state/t3_semantic/cache.jsonl" \
    --judge-audit-samples 100 --seed 42 >> "$state/t3_semantic/judge.log" 2>&1 && break
  (( attempt < 30 )) || fail "Eedi specialist semantic judge exhausted"
  sleep 120
done
kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; heartbeat_pid=""; flock -u 11

"$python_bin" - "$state/t3_semantic/metrics.json" "$state/sharded_merge/metrics.json" \
  "$state/sharded_merge/predictions.jsonl" "$state/validated" <<'PY' \
  || fail "Eedi specialist final validation failed"
import hashlib,json,pathlib,shutil,sys
judge_path,merge_path,pred_path,out=map(pathlib.Path,sys.argv[1:])
j=json.load(open(judge_path,encoding='utf-8')); m=json.load(open(merge_path,encoding='utf-8'))
key='diagnose_misconception/eedi_mining/open_taxonomy_generation'; group=j['metrics_by_group'][key]
assert j['judge_backend']=='trapi' and j['judge_errors']==0,j
assert j['generation_rows']==j['judged_rows']==int(group['llm_judge_n'])==894,j
assert m['sampled_rows']==m['generation_rows']==894 and m['evaluation_shards']==4,m
assert m['artifact_sha256']['adapter_model']=='329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2',m
j['artifact_sha256']=m['artifact_sha256']; j['evaluation_shards']=4
out.mkdir(parents=True,exist_ok=True)
(out/'metrics.json').write_text(json.dumps(j,indent=2,sort_keys=True)+'\n',encoding='utf-8')
shutil.copy2(pred_path,out/'predictions.jsonl')
shutil.copy2(judge_path.parent/'predictions_redacted.jsonl',out/'predictions_redacted.jsonl')
receipt={'status':'complete','stage':'t3_eedi_taskonly_qwen_formal_sharded_v1','n':894,
 'adapter_sha256':m['artifact_sha256']['adapter_model'],
 'predictions_sha256':hashlib.sha256(pred_path.read_bytes()).hexdigest(),
 'judge_backend':'trapi','judge_rubric':j['judge_rubric'],'evaluation_shards':4}
(out/'completion.json').write_text(json.dumps(receipt,sort_keys=True)+'\n',encoding='utf-8')
PY
touch "$state/complete"; rm -f "$state/needs_attention"
printf 'pass\tsharded_complete\n' > "$state/latest_status.tsv"
heartbeat; log "T3-Eedi task-only specialist validated n=894 shards=4"
