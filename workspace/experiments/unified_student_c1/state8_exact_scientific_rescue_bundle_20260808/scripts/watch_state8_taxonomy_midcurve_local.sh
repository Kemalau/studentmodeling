#!/usr/bin/env bash
set -euo pipefail

: "${STATE8_MIDCURVE_VARIANT:?STATE8_MIDCURVE_VARIANT is required}"
case "$STATE8_MIDCURVE_VARIANT" in
  t3heavy)
    remote=studentmodeling/checkpoints/state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_taxonomy_semantic_v1/checkpoint-round-500 ;;
  balanced)
    remote=studentmodeling/checkpoints/state8_qualified_teacher_balanced_curriculum_seed42_exact5000_rounds1250_v3_20260805_taxonomy_semantic_v1/checkpoint-round-500 ;;
  *) exit 2 ;;
esac

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_taxonomy_midcurve_20260807/$STATE8_MIDCURVE_VARIANT"
checkpoint="$state/checkpoint-round-500"
formal="$state/formal_screen"
semantic="$state/t3_semantic"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
mkdir -p "$state" "$formal" "$semantic"
cd "$project"
exec 9>"$state/pipeline_controller.lock"
flock -n 9 || exit 0
echo $$ > "$state/pipeline_controller.active.pid"
heartbeat_pid=""
cleanup(){
  if [[ -n "$heartbeat_pid" ]]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  rm -f "$state/pipeline_controller.active.pid"
}
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/pipeline_controller.heartbeat"; }
heartbeat

export STATE_SLUG="state8-taxonomy-midcurve-$STATE8_MIDCURVE_VARIANT"
export LOCAL_C4_SUBDIR=round500
export LOCAL_CHECKPOINT_STATE_DIR="$state"
export LOCAL_C4_COMPLETION_MARKER="$state/c4_complete"
export CHECKPOINT_ROUND=500
export EXPECTED_WORLD_SIZE=4
export EXPECTED_PLANNED_OPTIMIZER_STEPS=1250
export ARTIFACT_LABEL="state8-taxonomy-$STATE8_MIDCURVE_VARIANT"
export REMOTE_PREFIX="$remote"
bash scripts/watch_layerwise_remote_checkpoint_local_c4.sh

(while true; do heartbeat; sleep 60; done) &
heartbeat_pid=$!

exec 10>"$project/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do sleep 60; done
if [[ ! -s "$formal/metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols_taxonomy.py \
      --model "$model" --adapter "$checkpoint/adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$formal" --state-token-alias '<|vision_pad|>' \
      --max-samples-per-group 25 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
      --save-unredacted-predictions --t3-natural-schema \
      > "$formal/live.log" 2>&1
fi
flock -u 10

if [[ ! -s "$semantic/metrics.json" ]]; then
  PYTHONPATH="$project/scripts:/home/v-yuejiali/fkft/src" "$judge_python" \
    scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "$formal/predictions.jsonl" --eval-jsonl "$eval_jsonl" \
      --output-dir "$semantic" --judge-backend trapi --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 --max-records 75 \
      --judge-cache "$semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
      > "$semantic/judge.log" 2>&1
fi

set +e
"$python_bin" scripts/summarize_layerwise_checkpoint250_formal_screen.py \
  --root "$formal" --output "$state/formal_summary.json" --checkpoint-round 500 \
  > "$state/formal_summary.log" 2>&1
formal_rc=$?
set -e
(( formal_rc == 0 || formal_rc == 2 ))

"$python_bin" - "$state/artifact_audit.json" "$state/summary.json" \
  "$state/formal_summary.json" "$semantic/metrics.json" "$state/midcurve_summary.json" <<'PY'
import json,pathlib,sys
artifact,c4,formal,semantic=map(lambda p:json.load(open(p)),sys.argv[1:5])
groups=semantic['metrics_by_group']; support=sum(int(x['llm_judge_n']) for x in groups.values())
accuracy=sum(float(x['llm_judge_accuracy'])*int(x['llm_judge_n']) for x in groups.values())/support
assert semantic['judge_backend']=='trapi' and semantic['judge_errors']==0,semantic
assert all(int(x['llm_judge_exact_shortcuts'])==0 for x in groups.values()),groups
value={'stage':'state8_taxonomy_midcurve_local_v1','diagnostic_only':True,
 'checkpoint_round':500,'artifact':artifact,'C4':c4,'formal':formal,
 'T3_semantic':{'support':support,'accuracy':accuracy,'judge_backend':'trapi',
                'exact_match_authoritative':False}}
path=pathlib.Path(sys.argv[5]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
PY
touch "$state/complete"
heartbeat
