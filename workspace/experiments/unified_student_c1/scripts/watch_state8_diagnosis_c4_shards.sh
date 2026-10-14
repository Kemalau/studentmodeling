#!/usr/bin/env bash
set -uo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-diagnosis-c4-shards-c4g010-20260806
state="$project/results/state8_semantic_candidates_20260806/c4_g010/sharded_c4"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python
judge_python=/home/v-yuejiali/fkft/.venv/bin/python
judge_path="$project/scripts:/home/v-yuejiali/fkft/src"
t3_blob=projects/v-yuejiali/amlt-results/7213967912.49879-34c1b0c8-51a9/eval/formal_t3_300/predictions.jsonl
eval_jsonl="$project/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
export AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds

mkdir -p "$state"
cd "$project" || exit 1
exec 9>"$state/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state/controller.active.pid"
trap 'rm -f "$state/controller.active.pid"' EXIT
rm -f "$state/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state/needs_attention"; exit 1; }

while true; do
  heartbeat
  status="$(timeout 90 amlt status "$experiment" 2>&1)" || { sleep 120; continue; }
  printf '[%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$state/status.log"
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import all_expected_jobs_pass
raise SystemExit(0 if all_expected_jobs_pass(sys.argv[1],5) else 1)
PY
  then break; fi
  if "$python_bin" - "$status" <<'PY'
import sys
from scripts.recurrent_submit_guard import has_forward_progress_job
raise SystemExit(0 if has_forward_progress_job(sys.argv[1]) else 1)
PY
  then sleep 120; continue; fi
  printf '%s\n' "$status" > "$state/terminal_status.txt"
  fail "C4 shard experiment reached a terminal non-pass state"
done

rm -rf "$state/download" "$state/merged" "$state/t3_raw"
mkdir -p "$state/download" "$state/merged" "$state/t3_raw"
amlt results download "$experiment" --status pass \
  --include '*.json' --include '*.jsonl' --output "$state/download" \
  >> "$state/controller.log" 2>&1 || fail "C4 shard result download failed"
mapfile -t shard_metrics < <(find "$state/download" -type f -path '*/eval/diagnosis_c4_shard_*/metrics.json' | sort)
[[ ${#shard_metrics[@]} -eq 5 ]] || fail "Expected five C4 shard metric files"
merge_args=()
for metric in "${shard_metrics[@]}"; do merge_args+=(--shard "$(dirname "$metric")"); done
PYTHONPATH="$project/scripts" "$python_bin" scripts/merge_state8_diagnosis_c4_shards.py \
  "${merge_args[@]}" --expected-pairs 50 --output-dir "$state/merged" \
  >> "$state/controller.log" 2>&1 || fail "C4 shard merge failed"

az storage blob download --auth-mode login \
  --account-name csnewsandfeeds4150361735 --container-name amulet \
  --name "$t3_blob" --file "$state/t3_raw/predictions.jsonl" --overwrite \
  >> "$state/controller.log" 2>&1 || fail "T3 prediction download failed"
[[ -x "$judge_python" ]] || fail "TRAPI judge environment unavailable"
mkdir -p "$state/t3_semantic" "$state/c4_semantic"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$state/t3_raw/predictions.jsonl" --eval-jsonl "$eval_jsonl" \
  --output-dir "$state/t3_semantic" --judge-backend trapi \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/t3_semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 45 >> "$state/t3_semantic/judge.log" 2>&1 \
  || fail "T3 semantic judge failed"
PYTHONPATH="$judge_path" "$judge_python" scripts/judge_formal_t3_predictions.py \
  --predictions-jsonl "$state/merged/judge_predictions.jsonl" \
  --eval-jsonl "$state/merged/judge_eval.jsonl" \
  --output-dir "$state/c4_semantic" --judge-backend trapi \
  --judge-workers 12 --judge-max-retries 12 --judge-timeout-seconds 60 \
  --judge-cache "$state/c4_semantic/cache.jsonl" \
  --judge-audit-samples 100 --seed 42 >> "$state/c4_semantic/judge.log" 2>&1 \
  || fail "C4 semantic judge failed"
"$python_bin" scripts/summarize_state8_diagnosis_c4.py \
  --judge-redacted "$state/c4_semantic/predictions_redacted.jsonl" \
  --generation-metrics "$state/merged/metrics.json" \
  --output "$state/c4_semantic/summary.json" \
  >> "$state/controller.log" 2>&1 || fail "C4 semantic summary failed"
touch "$state/complete"; heartbeat
log "C4 shards merged and authoritative T3/C4 semantic judgments completed"
