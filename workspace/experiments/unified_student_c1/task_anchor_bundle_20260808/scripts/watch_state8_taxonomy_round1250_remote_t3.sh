#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment="${STATE8_ROUND1250_T3_EXPERIMENT:-state8-taxonomy-round1250-t3-screen-20260807}"
job=state8-taxonomy-t3heavy-round1250-t3-screen
state="$project/results/state8_taxonomy_round1250_remote_t3_20260807"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"

mkdir -p "$state"
cd "$project"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure}"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); print(rows[0]['status'] if len(rows)==1 else 'invalid')
PY
)"
  case "$phase" in
    preparing|queued|running|paused|resuming|pre-queued) sleep 60 ;;
    pass) break ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 1600 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
      fail "remote round-1250 T3 screen ended in state=$phase"
      ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 || fail "result download failed"
mapfile -t predictions < <(find "$state/download" -type f -path "*/$job/eval/t3_screen/predictions.jsonl")
mapfile -t metrics < <(find "$state/download" -type f -path "*/$job/eval/t3_screen/metrics.json")
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] || fail "artifact cardinality drift"

"$python_bin" - "${metrics[0]}" <<'PY' || exit 1
import json,sys
x=json.load(open(sys.argv[1])); expected={
 'adapter_model':'bf75b17f641731c84ac05718d67563c4827cd80eda78d83a2c2b1c94cdde4260',
 'recurrent_state':'202216966726650d70707678c67d0573f9d0fa3486583fde2c3e0b08c62f8613'}
assert x['state_shape']==[64,8,5120] and x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False and x['sampled_rows']==75,x
assert x['artifact_sha256']==expected,(x,expected)
PY

mkdir -p "$state/semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/semantic" --judge-backend trapi --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
  >> "$state/semantic/judge.log" 2>&1 || fail "TRAPI semantic judge failed"

"$python_bin" - "${metrics[0]}" "$state/semantic/metrics.json" "$state/summary.json" <<'PY' || exit 1
import json,pathlib,sys
formal=json.load(open(sys.argv[1])); semantic=json.load(open(sys.argv[2]))
assert semantic['judge_backend']=='trapi' and semantic['judge_errors']==0,semantic
groups=semantic['metrics_by_group']; assert groups,semantic
assert all(int(row.get('llm_judge_exact_shortcuts',-1))==0 for row in groups.values()),groups
support=sum(int(row['llm_judge_n']) for row in groups.values())
accuracy=sum(float(row['llm_judge_accuracy'])*int(row['llm_judge_n']) for row in groups.values())/support
assert support==semantic['judged_rows']==semantic['generation_rows']==75,semantic
out={'stage':'state8_taxonomy_round1250_remote_t3_semantic_v1','checkpoint_round':1250,
 'raw_history_visible_to_decoder':False,'state_shape':[64,8,5120],'support':support,
 'semantic_accuracy':accuracy,'judge_backend':'trapi','judge_errors':0,
 'exact_match_authoritative':False,'artifact_sha256':formal['artifact_sha256']}
path=pathlib.Path(sys.argv[3]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state/complete"; heartbeat
log "round-1250 strict-state-only T3 semantic screen complete"
