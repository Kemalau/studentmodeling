#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-taxonomy-consolidation-round35-t3-probe-20260808
job=state8-taxonomy-consolidation-round35-t3-probe
state="$project/results/state8_taxonomy_unified_pipeline_20260807/round35_t3_probe"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"

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
      timeout 180 amlt logs view -n 1800 "$experiment" ":$job" > "$state/failure.log" 2>&1 || true
      fail "round-35 T3 probe ended in state=$phase"
      ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --status pass --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 || fail "probe download failed"
mapfile -t predictions < <(find "$state/download" -type f -path "*/$job/eval/t3_screen/predictions.jsonl")
mapfile -t metrics < <(find "$state/download" -type f -path "*/$job/eval/t3_screen/metrics.json")
[[ ${#predictions[@]} -eq 1 && ${#metrics[@]} -eq 1 ]] || fail "probe artifact cardinality drift"

mkdir -p "$state/semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/semantic" --judge-backend trapi --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/semantic/cache.jsonl" --judge-audit-samples 75 --seed 42 \
  >> "$state/semantic/judge.log" 2>&1 || fail "TRAPI judge failed"

"$python_bin" - "${predictions[0]}" "$state/semantic/metrics.json" "$state/summary.json" <<'PY'
import collections,json,pathlib,sys
pred=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
semantic=json.load(open(sys.argv[2])); groups=semantic['metrics_by_group']
assert semantic['judge_backend']=='trapi' and semantic['judge_errors']==0,semantic
assert semantic['generation_rows']==semantic['judged_rows']==75,semantic
support=sum(int(row['llm_judge_n']) for row in groups.values())
accuracy=sum(float(row['llm_judge_accuracy'])*int(row['llm_judge_n']) for row in groups.values())/support
eedi=[row for row in pred if row['source']=='eedi_mining']
outputs=[str(row.get('raw_output') or '') for row in eedi]
out={'stage':'state8_taxonomy_consolidation_round35_t3_probe_v1','checkpoint_round':35,
 'semantic_accuracy':accuracy,'support':support,'judge_backend':'trapi',
 'eedi_semantic_accuracy':groups['diagnose_misconception/eedi_mining/open_taxonomy_generation']['llm_judge_accuracy'],
 'malrule_cross_semantic_accuracy':groups['diagnose_misconception/malrule/synthetic_cross_template_open_generation']['llm_judge_accuracy'],
 'malrule_same_semantic_accuracy':groups['diagnose_misconception/malrule/synthetic_same_template_open_generation']['llm_judge_accuracy'],
 'eedi_unique_output_rate':len(set(outputs))/len(outputs),
 'eedi_most_common_output_rate':collections.Counter(outputs).most_common(1)[0][1]/len(outputs),
 'exact_match_authoritative':False}
path=pathlib.Path(sys.argv[3]); tmp=path.with_suffix('.json.partial')
tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); tmp.replace(path)
print(json.dumps(out,indent=2,sort_keys=True))
PY
touch "$state/complete"; heartbeat
log "round-35 T3 semantic probe complete"
