#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-strong1-midpoint-eval-step2000-20260808
state="$project/results/state8_strong1_midpoint_eval_20260808"
c4_job=state8-strong1-midpoint-step2000-c4
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"

if [[ -z "${AZURE_CONFIG_DIR:-}" ]]; then
  export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure
fi
mkdir -p "$state"
cd "$project" || exit 1
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
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }
heartbeat
(while true; do heartbeat; sleep 30; done) & heartbeat_pid=$!

while true; do
  status="$(timeout 90 amlt --json-tables status "$experiment" --hide-urls 2>/dev/null)" \
    || { sleep 60; continue; }
  printf '%s\n' "$status" > "$state/latest_status.json"
  phase="$(STATUS="$status" C4_JOB="$c4_job" "$python_bin" - <<'PY'
import json,os
try: rows=json.loads(os.environ['STATUS'])
except Exception: print('invalid'); raise SystemExit
wanted={':' + os.environ['C4_JOB']}
selected=[row for row in rows if str(row.get('job_name') or '') in wanted]
if {str(row.get('job_name') or '') for row in selected} != wanted:
 print('active'); raise SystemExit
states={str(row.get('status') or '').casefold() for row in selected}
if states=={'pass'}: print('pass')
elif states & {'failed','cancelled','canceled','stopped','killed'}: print('failed')
elif states <= {'preparing','pre-queued','queued','paused','resuming','running'}: print('active')
else: print('invalid')
PY
)"
  case "$phase" in
    pass) touch "$state/remote_passed"; break ;;
    active) sleep 60 ;;
    failed)
      timeout 90 amlt show "$experiment" > "$state/show.txt" 2>&1 || true
      timeout 180 amlt logs view -n 3000 "$experiment" > "$state/failure.log" 2>&1 || true
      fail "remote midpoint evaluation reached a failed terminal state"
      ;;
    *) sleep 60 ;;
  esac
done

rm -rf "$state/download"; mkdir -p "$state/download" "$state/diagnosis_c4_semantic"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --include '*.txt' \
  --output "$state/download" >> "$state/controller.log" 2>&1 \
  || fail "result download failed"

mapfile -t answer_c4 < <(find "$state/download" -type f \
  -path '*/state8-strong1-midpoint-step2000-c4/eval/answer_c4_step2000_n50/recurrent_causal_gate.json')
mapfile -t diag_predictions < <(find "$state/download" -type f \
  -path '*/state8-strong1-midpoint-step2000-c4/eval/diagnosis_c4_step2000_n50/judge_predictions.jsonl')
mapfile -t diag_eval < <(find "$state/download" -type f \
  -path '*/state8-strong1-midpoint-step2000-c4/eval/diagnosis_c4_step2000_n50/judge_eval.jsonl')
mapfile -t diag_metrics < <(find "$state/download" -type f \
  -path '*/state8-strong1-midpoint-step2000-c4/eval/diagnosis_c4_step2000_n50/metrics.json')
[[ ${#answer_c4[@]} -eq 1 \
   && ${#diag_predictions[@]} -eq 1 && ${#diag_eval[@]} -eq 1 && ${#diag_metrics[@]} -eq 1 ]] \
  || fail "result artifact cardinality drift"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"

PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "${diag_predictions[0]}" --eval-jsonl "${diag_eval[0]}" \
  --output-dir "$state/diagnosis_c4_semantic" --judge-backend trapi --judge-workers 12 \
  --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/diagnosis_c4_semantic/cache.jsonl" --judge-audit-samples 100 --seed 42 \
  > "$state/diagnosis_c4_semantic/judge.log" 2>&1 || fail "diagnosis-C4 semantic judge failed"

"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/diagnosis_c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "${diag_metrics[0]}" --output "$state/diagnosis_c4_summary.json" \
  > "$state/diagnosis_c4_summarize.log" 2>&1 || fail "diagnosis-C4 summary failed"

"$python_bin" - "${answer_c4[0]}" "$state/diagnosis_c4_summary.json" "$state/summary.json" <<'PY' \
  || fail "midpoint C4 summary failed"
import json,pathlib,sys
answer,diagnosis=map(lambda p:json.load(open(p)),sys.argv[1:3])
s=answer['summary']; clean=s['clean']
out={
 'stage':'state8_strong1_midpoint_step2000_c4_eval_v1','status':'complete',
 'checkpoint_step':2000,'state_shape':[64,8,5120],
 'history_dropout_counts':{'0.0':1000,'0.25':1000,'0.5':0,'0.75':0,'1.0':0},
 'raw_history_visible_to_decoder':False,'semantic_judge_authoritative':True,
 'exact_match_authoritative':False,
 'answer_c4':{'clean_n':s['clean_n'],'swap':clean['swap']['flip_to_A_rate'],
              'random':clean['random']['flip_to_A_rate'],
              'same_rule':clean['samemal']['flip_to_A_rate'],
              'noop':clean['noop']['flip_to_A_rate'],'D_specific':s['D_specific'],
              'checks':s['checks'],'scientific_gate_pass':s['scientific_gate_pass']},
 'diagnosis_c4':diagnosis,
 'artifact_sha256':answer['artifact_sha256'],
}
assert answer['raw_history_visible_to_decoder'] is False,answer
p=pathlib.Path(sys.argv[3]); q=p.with_suffix('.json.partial')
q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); q.replace(p)
print(json.dumps(out,sort_keys=True))
PY

touch "$state/complete"; heartbeat
log "midpoint answer/diagnosis C4 evaluation complete"
