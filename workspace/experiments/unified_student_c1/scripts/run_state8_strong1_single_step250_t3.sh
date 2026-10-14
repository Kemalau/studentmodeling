#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
checkpoint="$project/results/state8_strong1_widen8_bridge_20260808/single_milestone_c4/step-250/checkpoint-250"
audit="$project/results/state8_strong1_widen8_bridge_20260808/single_milestone_c4/step-250/artifact_audit.json"
state="$project/results/state8_strong1_widen8_bridge_20260808/single_step250_t3"
formal="$state/formal"
semantic="$state/semantic_v2"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
rubric=t3-semantic-equivalence-v2-substantive-schema

mkdir -p "$formal" "$semantic"
cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  rm -f "$state/controller.active.pid"
}
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
fail(){ printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; touch "$state/needs_attention"; exit 1; }
heartbeat

for path in "$checkpoint/checkpoint_completion.json" "$audit" "$eval_jsonl"; do
  test -f "$path" || fail "missing input=$path"
done

# The read-only milestone cleanup removes multi-GB model artifacts after
# predictions and their hash-bound metrics have been materialized.  Rebuilding
# or upgrading a semantic summary must therefore remain possible without
# downloading the checkpoint again.  Only generation needs the model files.
if [[ ! -s "$formal/metrics.json" ]]; then
  for path in "$checkpoint/adapter/adapter_model.safetensors" \
    "$checkpoint/recurrent_state.pt" "$model/model.safetensors.index.json"; do
    test -f "$path" || fail "missing generation input=$path"
  done
fi

"$python_bin" - "$checkpoint/checkpoint_completion.json" "$audit" <<'PY' \
  || fail "artifact provenance audit failed"
import json,sys
r=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2]))
assert r['status']=='complete' and r['step']==250,r
assert r['sha256']==a['sha256'],(r,a)
assert a['state_shape']==[64,8,5120] and a['raw_history_visible_to_decoder'] is False,a
PY

if [[ ! -s "$formal/metrics.json" ]]; then
  exec 10>"$project/results/local_gpu_readonly_preflight.lock"
  while ! flock -n 10; do heartbeat; sleep 30; done
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
    heartbeat; sleep 30
  done
  (while true; do heartbeat; sleep 30; done) &
  heartbeat_pid=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols_taxonomy.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$formal" --state-token-alias '<|vision_pad|>' \
      --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
      --tasks diagnose_misconception --sources eedi_mining malrule \
      --save-unredacted-predictions --t3-natural-schema \
      > "$formal/live.log" 2>&1 \
    || {
      kill "$heartbeat_pid" 2>/dev/null || true
      wait "$heartbeat_pid" 2>/dev/null || true
      heartbeat_pid=""; flock -u 10; fail "local T3 generation failed"
    }
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""; flock -u 10
fi

"$python_bin" - "$formal/metrics.json" "$audit" <<'PY' \
  || fail "strict T3 generation audit failed"
import json,sys
x=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2]))
assert x['state_architecture']=='layerwise' and x['state_shape']==[64,8,5120],x
assert x['raw_history_visible_to_decoder'] is False,x
assert x['persistent_history_kv_cache'] is False and x['sampled_rows']==75,x
assert x['artifact_sha256']==a['sha256'],(x,a)
PY

if [[ ! -s "$semantic/metrics.json" ]]; then
  (while true; do heartbeat; sleep 30; done) &
  heartbeat_pid=$!
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "$formal/predictions.jsonl" --eval-jsonl "$eval_jsonl" \
    --output-dir "$semantic" --judge-backend trapi --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 --max-records 75 \
    --judge-cache "$semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
    > "$semantic/judge.log" 2>&1 \
    || {
      kill "$heartbeat_pid" 2>/dev/null || true
      wait "$heartbeat_pid" 2>/dev/null || true
      heartbeat_pid=""; fail "TRAPI semantic judge failed"
    }
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
fi

"$python_bin" - "$formal/metrics.json" "$semantic/metrics.json" \
  "$state/summary.json" "$rubric" <<'PY' \
  || fail "T3 semantic summary failed"
import json,pathlib,sys
formal=json.load(open(sys.argv[1])); semantic=json.load(open(sys.argv[2]))
groups=semantic['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
raw_accuracy=sum(float(x['llm_judge_raw_semantic_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
schema_n=sum(int(x['llm_judge_schema_valid_n']) for x in groups.values())
rubric=sys.argv[4]
assert semantic['judge_backend']=='trapi' and semantic['judge_errors']==0,semantic
assert semantic['judge_rubric']==rubric,semantic
assert support==75 and all(int(x['llm_judge_exact_shortcuts'])==0 for x in groups.values()),groups
out={'stage':'state8_strong1_single_step250_t3_semantic_v2','checkpoint_step':250,
 'support':support,'semantic_accuracy':accuracy,'raw_semantic_accuracy':raw_accuracy,
 'substantive_schema_valid_rate':schema_n/support,'metrics_by_group':groups,
 'judge_rubric':rubric,
 'state_shape':[64,8,5120],'raw_history_visible_to_decoder':False,
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'artifact_sha256':formal['artifact_sha256'],'diagnostic_only':True}
p=pathlib.Path(sys.argv[3]); q=p.with_suffix('.json.partial')
q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); q.replace(p)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state/complete"; heartbeat
