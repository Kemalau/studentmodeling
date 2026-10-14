#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/formal_specialist_v2/unrestricted_qwen_t3_v2"
raw="$project/results/formal_specialist_v1/t3_llm_judge_raw_amlt"
eval_jsonl="$project/data/processed/formal_evaluation_v1/test.jsonl"
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
mkdir -p "$state"; cd "$project"
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
fail(){ printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$state/controller.log"; touch "$state/needs_attention"; exit 1; }

mapfile -t adapter_predictions < <(find "$raw" -type f \( \
  -path '*/adapter_t3_eedi/predictions.jsonl' -o \
  -path '*/adapter_t3_malrule/predictions.jsonl' \) | sort)
mapfile -t base_predictions < <(find "$raw" -type f \( \
  -path '*/base_t3_eedi/predictions.jsonl' -o \
  -path '*/base_t3_malrule/predictions.jsonl' \) | sort)
[[ ${#adapter_predictions[@]} -eq 2 && ${#base_predictions[@]} -eq 2 ]] \
  || fail "raw prediction cardinality drift"
[[ -x "$judge_python" && -s "$eval_jsonl" ]] || fail "judge runtime or eval manifest missing"

judge(){
  local label="$1" destination="$state/$2"; shift 2
  local -a predictions=("$@")
  mkdir -p "$destination"
  [[ -s "$destination/metrics.json" ]] && return 0
  bash scripts/wait_for_trapi_auth.sh
  heartbeat
  PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
    --predictions-jsonl "${predictions[@]}" --eval-jsonl "$eval_jsonl" \
    --output-dir "$destination" --judge-backend trapi \
    --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
    --judge-max-retries 12 --judge-timeout-seconds 60 \
    --judge-cache "$destination/cache.jsonl" --judge-audit-samples 100 \
    --seed 42 >> "$destination/judge.log" 2>&1 \
    || fail "$label v2 semantic rejudge failed"
}

judge unified adapter_diagnose_misconception "${adapter_predictions[@]}"
judge base base_diagnose_misconception "${base_predictions[@]}"

/home/v-yuejiali/.venvs/studentmodeling/bin/python - \
  "$state/adapter_diagnose_misconception/metrics.json" \
  "$state/base_diagnose_misconception/metrics.json" \
  "$state/completion.json" <<'PY' || fail "v2 rejudge validation failed"
import json,pathlib,sys
adapter,base=(json.load(open(path,encoding='utf-8')) for path in sys.argv[1:3])
for value in (adapter,base):
    assert value['judge_backend']=='trapi',value
    assert value['judge_rubric']=='t3-semantic-equivalence-v2-substantive-schema',value
    assert int(value['generation_rows'])==1494 and int(value['judged_rows'])==1494,value
    assert int(value['judge_errors'])==0,value
    assert sum(int(group['llm_judge_n']) for group in value['metrics_by_group'].values())==1494,value
pathlib.Path(sys.argv[3]).write_text(json.dumps({
    'stage':'unrestricted_qwen_t3_v2_rejudge_v1','status':'complete',
    'judge_backend':'trapi','judge_rubric':'t3-semantic-equivalence-v2-substantive-schema',
    'base_n':1494,'unified_n':1494,
},sort_keys=True)+'\n',encoding='utf-8')
PY
rm -f "$state/needs_attention"
touch "$state/complete"; heartbeat
