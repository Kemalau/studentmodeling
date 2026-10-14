#!/usr/bin/env bash
set -euo pipefail

: "${PROBE_VARIANT:?PROBE_VARIANT is required}"
case "$PROBE_VARIANT" in
  t3heavy)
    seed=52
    run_tag=taxonomy_curriculum_consolidation_t3heavy_v1
    job=state8-taxonomy-consolidation-t3heavy-r100-probe ;;
  t3heavy_diag)
    seed=54
    run_tag=taxonomy_curriculum_consolidation_t3heavy_diag_v1
    job=state8-taxonomy-consolidation-t3heavy-diag-r100-probe ;;
  *) exit 2 ;;
esac

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
config=amlt/feeds_state8_taxonomy_consolidation_milestone_probe_20260808.yaml
round=100
account=csnewsandfeeds4150361735
container=amulet
source_prefix="studentmodeling/checkpoints/state8_stateonly_repair_seed${seed}_trajectory01_${run_tag}/checkpoint-round-${round}"
snapshot_prefix="studentmodeling/diagnostics/state8_taxonomy_consolidation_milestones_20260808/${PROBE_VARIANT}/round-${round}"
state="$project/results/state8_taxonomy_unified_pipeline_20260807/milestone_r100_${PROBE_VARIANT}"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
base_experiment="state8-taxonomy-consolidation-${PROBE_VARIANT}-r100-probe-20260808"

mkdir -p "$state"
cd "$project"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
blob_exists(){
  az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$1" --query exists -o tsv 2>/dev/null
}

copy_blob_verified(){
  local file="$1" source="$source_prefix/$file" destination="$snapshot_prefix/$file"
  local source_json source_size source_etag destination_json destination_size copy_status
  source_json="$(az storage blob show --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$source" \
    --query '{size:properties.contentLength,etag:properties.etag}' -o json)" \
    || fail "source metadata unavailable: $file"
  mapfile -t fields < <("$python_bin" - "$source_json" <<'PY'
import json,sys
x=json.loads(sys.argv[1]); print(x.get('size','')); print(x.get('etag',''))
PY
)
  source_size="${fields[0]:-}"; source_etag="${fields[1]:-}"
  [[ "$source_size" =~ ^[1-9][0-9]*$ && -n "$source_etag" ]] \
    || fail "invalid source metadata: $file"
  destination_json="$(az storage blob show --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$destination" \
    --query '{size:properties.contentLength,status:properties.copy.status}' -o json 2>/dev/null || true)"
  mapfile -t fields < <("$python_bin" - "$destination_json" <<'PY'
import json,sys
try: x=json.loads(sys.argv[1])
except Exception: x={}
print(x.get('size','')); print(x.get('status') or '')
PY
)
  destination_size="${fields[0]:-}"; copy_status="${fields[1]:-}"
  if [[ "$destination_size" == "$source_size" \
        && ( -z "$copy_status" || "$copy_status" == None || "$copy_status" == success ) ]]; then
    return 0
  fi
  if [[ "$copy_status" != pending ]]; then
    az storage blob copy start --auth-mode login --account-name "$account" \
      --destination-container "$container" --destination-blob "$destination" \
      --source-account-name "$account" --source-container "$container" \
      --source-blob "$source" --source-if-match "$source_etag" -o none \
      >/dev/null 2>&1 || fail "snapshot copy start failed: $file"
  fi
  local deadline=$(( $(date +%s) + 3600 ))
  while true; do
    heartbeat
    destination_json="$(az storage blob show --auth-mode login --account-name "$account" \
      --container-name "$container" --name "$destination" \
      --query '{size:properties.contentLength,status:properties.copy.status}' -o json 2>/dev/null || true)"
    mapfile -t fields < <("$python_bin" - "$destination_json" <<'PY'
import json,sys
try: x=json.loads(sys.argv[1])
except Exception: x={}
print(x.get('size','')); print(x.get('status') or '')
PY
)
    destination_size="${fields[0]:-}"; copy_status="${fields[1]:-}"
    if [[ "$copy_status" == success ]]; then
      [[ "$destination_size" == "$source_size" ]] || fail "snapshot size mismatch: $file"
      return 0
    fi
    [[ "$copy_status" != failed && "$copy_status" != aborted ]] \
      || fail "snapshot copy status=$copy_status: $file"
    (( $(date +%s) < deadline )) || fail "snapshot copy timed out: $file"
    sleep 10
  done
}

files=(adapter/adapter_config.json adapter/adapter_model.safetensors recurrent_state.pt checkpoint_completion.json)
while [[ ! -f "$state/snapshot_complete" ]]; do
  heartbeat
  ready=true
  for file in "${files[@]}"; do
    [[ "$(blob_exists "$source_prefix/$file")" == true ]] || ready=false
  done
  if [[ "$ready" != true ]]; then sleep 30; continue; fi
  for file in "${files[@]}"; do copy_blob_verified "$file"; done
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$snapshot_prefix/checkpoint_completion.json" \
    --file "$state/checkpoint_completion.json" --overwrite -o none \
    || fail "snapshot receipt download failed"
  "$python_bin" - "$state/checkpoint_completion.json" "$round" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); r=int(sys.argv[2])
assert x['status']=='complete' and x['round']==r and x['world_size']==4,x
assert set(x['sha256'])=={'adapter_model','recurrent_state'},x
assert all(len(v)==64 for v in x['sha256'].values()),x
PY
  touch "$state/snapshot_complete"
  log "immutable snapshot verified source=$source_prefix destination=$snapshot_prefix"
done

"$python_bin" scripts/validate_state8_taxonomy_consolidation_milestone_probe_amlt.py --config "$config" \
  >> "$state/controller.log" 2>&1 || fail "source config preflight failed"
exec 8>"$project/results/feeds_submission_global.lock"; flock 8
amlt run "$config" ":$job" --dump -t Feeds > "$state/preflight.yaml" 2> "$state/preflight.stderr" \
  || { flock -u 8; fail "rendered preflight failed"; }
"$python_bin" scripts/validate_state8_taxonomy_consolidation_milestone_probe_amlt.py --config "$state/preflight.yaml" \
  >> "$state/controller.log" 2>&1 || { flock -u 8; fail "rendered config preflight failed"; }
amlt --json-tables ti sing -t Feeds > "$state/quota.json" 2>> "$state/controller.log" \
  || { flock -u 8; fail "quota check failed"; }
flock -u 8

passed=false
for wave in 0 1 2 3 4; do
  experiment="$base_experiment"
  (( wave == 0 )) || experiment="${base_experiment}-recovery${wave}"
  set +e
  status="$(timeout 90 amlt status --no-update-db "$experiment" 2>&1)"; rc=$?
  set -e
  class="$("$python_bin" - "$rc" "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import classify_experiment_status
print(classify_experiment_status(int(sys.argv[1]),sys.argv[2]))
PY
)" || fail "ambiguous pre-submit state experiment=$experiment"
  if [[ "$class" == missing ]]; then
    exec 8>"$project/results/feeds_submission_global.lock"; flock 8
    amlt run "$config" ":$job" "$experiment" -t Feeds -y \
      --description "Read-only strict-state-only T3+C4 milestone probe variant=$PROBE_VARIANT round=$round recovery=$wave" \
      > "$state/$experiment.submission.txt" 2>&1 \
      || { flock -u 8; fail "submission failed experiment=$experiment"; }
    flock -u 8
    log "submitted experiment=$experiment job=$job"
  elif [[ "$class" != exists ]]; then
    fail "ambiguous existing experiment=$experiment"
  fi
  while true; do
    heartbeat
    status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
      || { sleep 60; continue; }
    printf '%s\n' "$status" > "$state/latest_status.json"
    phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
x=json.loads(os.environ['STATUS']); print(x[0]['status'] if len(x)==1 else 'invalid')
PY
)"
    case "$phase" in
      preparing|queued|running|paused|resuming|pre-queued) sleep 60 ;;
      pass) passed=true; break ;;
      *)
        timeout 90 amlt show "$experiment" > "$state/$experiment.show.txt" 2>&1 || true
        timeout 180 amlt logs view -n 1600 "$experiment" ":$job" \
          > "$state/$experiment.failure.log" 2>&1 || true
        break ;;
    esac
  done
  [[ "$passed" == true ]] && break
done
[[ "$passed" == true ]] || fail "all probe recovery waves failed"

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 || fail "result download failed"
mapfile -t t3pred < <(find "$state/download" -type f -path "*/$job/eval/t3_screen/predictions.jsonl")
mapfile -t c4pred < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_10/judge_predictions.jsonl")
mapfile -t c4eval < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_10/judge_eval.jsonl")
mapfile -t c4metrics < <(find "$state/download" -type f -path "*/$job/eval/diagnosis_c4_10/metrics.json")
[[ ${#t3pred[@]} -eq 1 && ${#c4pred[@]} -eq 1 && ${#c4eval[@]} -eq 1 && ${#c4metrics[@]} -eq 1 ]] \
  || fail "artifact cardinality drift"
mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${t3pred[0]}" --eval-jsonl "$eval_jsonl" --output-dir "$state/t3_semantic" \
  --judge-backend trapi --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/t3_semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
  >> "$state/t3_semantic/judge.log" 2>&1 || fail "T3 judge failed"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${c4pred[0]}" --eval-jsonl "${c4eval[0]}" --output-dir "$state/c4_semantic" \
  --judge-backend trapi --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
  >> "$state/c4_semantic/judge.log" 2>&1 || fail "C4 judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "${c4metrics[0]}" --output "$state/c4_semantic/summary.json" \
  >> "$state/c4_semantic/summary.log" 2>&1 || fail "C4 summary failed"
"$python_bin" - "$state/t3_semantic/metrics.json" "$state/c4_semantic/summary.json" \
  "$state/checkpoint_completion.json" "$state/summary.json" "$PROBE_VARIANT" <<'PY'
import json,pathlib,sys
t3=json.load(open(sys.argv[1])); c4=json.load(open(sys.argv[2])); receipt=json.load(open(sys.argv[3]))
groups=t3['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
assert t3['judge_backend']=='trapi' and t3['judge_errors']==0 and support==75,t3
out={'stage':'state8_taxonomy_consolidation_milestone_probe_v1','variant':sys.argv[5],
 'checkpoint_round':receipt['round'],'artifact_sha256':receipt['sha256'],
 'T3_semantic_accuracy':accuracy,'T3_support':support,'C4_semantic':c4,
 'raw_history_visible_to_decoder':False,'state_shape':[64,8,5120],
 'semantic_judge_authoritative':True,'exact_match_authoritative':False}
p=pathlib.Path(sys.argv[4]); q=p.with_suffix('.json.partial')
q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); q.replace(p)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state/complete"; heartbeat
log "milestone probe complete variant=$PROBE_VARIANT"
