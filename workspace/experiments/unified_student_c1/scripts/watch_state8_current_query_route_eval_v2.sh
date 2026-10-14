#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state="$project/results/state8_current_query_route_eval_v2_20260811"
experiment=state8-current-query-route-eval-v2-20260811
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
jobs=(state8-cp100-state-plus-current-item state8-cp100-current-item-only)
arms=(state_plus_current_item current_item_only)

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
cleanup(){ rm -f "$state/controller.active.pid"; }
trap cleanup EXIT
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
sleep_unlocked(){ sleep "$1" 9>&-; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while [[ ! -f "$state/remote_passed" ]]; do
  heartbeat
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls --force-status-update 2>/dev/null)" \
    || { sleep_unlocked 60; continue; }
  [[ -n "$status" && "$status" != '[]' ]] || { sleep_unlocked 30; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" "$python_bin" - <<'PY'
import json,os
rows=json.loads(os.environ['STATUS']); states={str(x.get('status') or '').casefold() for x in rows}
if len(rows)==2 and states=={'pass'}: print('pass')
elif states & {'preparing','pre-queued','queued','running','resuming'}: print('active')
elif 'paused' in states: print('paused')
else: print('terminal')
PY
)" || fail "status classification failed"
  case "$phase" in
    pass) touch "$state/remote_passed" ;;
    active) sleep_unlocked 60 ;;
    paused)
      timeout 90 amlt resume "$experiment" >> "$state/controller.log" 2>&1 || true
      sleep_unlocked 60 ;;
    *)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" > "$state/failure.log" 2>&1 || true
      fail "route counterfactual reached terminal non-pass state" ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download"
amlt results download "$experiment" --include '*.json' --include '*.jsonl' \
  --include '*.txt' --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"

summaries=()
for index in "${!jobs[@]}"; do
  job="${jobs[$index]}"; arm="${arms[$index]}"; out="$state/semantic/$arm"
  mapfile -t roots < <(find "$state/download" -type d -name "$job")
  [[ ${#roots[@]} -eq 1 ]] || fail "artifact root cardinality arm=$arm"
  root="${roots[0]}"
  mapfile -t predictions < <(find "$root" -type f -path '*/eval/formal_t3_75/predictions.jsonl')
  mapfile -t generation_metrics < <(find "$root" -type f -path '*/eval/formal_t3_75/metrics.json')
  mapfile -t receipts < <(find "$root" -type f -path '*/metrics/route_receipt.json')
  [[ ${#predictions[@]} -eq 1 && ${#generation_metrics[@]} -eq 1 && ${#receipts[@]} -eq 1 ]] \
    || fail "evaluation artifact cardinality arm=$arm"
  mkdir -p "$out"
  until PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
      --predictions-jsonl "${predictions[0]}" --eval-jsonl "$eval_jsonl" \
      --output-dir "$out" --judge-backend trapi \
      --trapi-deployment gpt-4o_2024-11-20 --judge-workers 12 \
      --judge-max-retries 12 --judge-timeout-seconds 60 \
      --judge-cache "$out/cache.jsonl" --judge-audit-samples 75 --seed 42 \
      >> "$out/judge.log" 2>&1; do heartbeat; sleep_unlocked 120; done
  "$python_bin" - "$arm" "${generation_metrics[0]}" "$out/metrics.json" \
      "${receipts[0]}" "$out/summary.json" <<'PY' || fail "summary failed arm=$arm"
import json,pathlib,sys
arm=sys.argv[1]
generation=json.load(open(sys.argv[2])); semantic=json.load(open(sys.argv[3])); receipt=json.load(open(sys.argv[4]))
groups=semantic['metrics_by_group']
def acc(fragment):
    matches=[v for k,v in groups.items() if fragment in k]
    assert len(matches)==1,(fragment,groups.keys())
    return float(matches[0]['llm_judge_accuracy'])
assert receipt['route_mode']==arm and receipt['raw_history_visible_to_decoder'] is False,receipt
assert receipt['student_response_visible_to_decoder'] is False,receipt
value={
 'stage':'state8_current_query_route_arm_summary_v1','status':'complete','diagnostic_only':True,
 'arm':arm,'T3_eedi_semantic':acc('eedi_mining/open_taxonomy_generation'),
 'T3_synthetic_same_semantic':acc('malrule/synthetic_same_template'),
 'T3_synthetic_cross_semantic':acc('malrule/synthetic_cross_template'),
 'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'raw_history_visible_to_decoder':False,'student_response_visible_to_decoder':False,
 'artifact_sha256':generation['artifact_sha256'],
}
pathlib.Path(sys.argv[5]).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
  summaries+=("$out/summary.json")
done

"$python_bin" - "${summaries[0]}" "${summaries[1]}" "$state/comparison.json" <<'PY' \
  || fail "comparison failed"
import json,pathlib,sys
state,item=map(lambda p:json.load(open(p)),sys.argv[1:3])
assert state['arm']=='state_plus_current_item' and item['arm']=='current_item_only',(state,item)
metrics=('T3_eedi_semantic','T3_synthetic_same_semantic','T3_synthetic_cross_semantic')
delta={key:state[key]-item[key] for key in metrics}
value={
 'stage':'state8_current_query_route_counterfactual_comparison_v1','status':'complete',
 'diagnostic_only':True,'semantic_judge_authoritative':True,'exact_match_authoritative':False,
 'state_plus_current_item':state,'current_item_only':item,'state_contribution_delta':delta,
 'routing_restores_eedi':state['T3_eedi_semantic']>0.0,
 'state_adds_eedi_information':delta['T3_eedi_semantic']>0.0,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY

touch "$state/complete"; heartbeat
log "current-query route counterfactual complete"

