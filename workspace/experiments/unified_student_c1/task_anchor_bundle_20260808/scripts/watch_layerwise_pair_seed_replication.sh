#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/pair_seed_replication"
artifact_root="$project_dir/results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/calibration_artifacts/layerwise-eedi-full-open-t3-lr-scan-recovery1-seed42-20260802/layerwise-eedi-full-open-t3-lr2e6-seed42"
model=/home/v-yuejiali/studentmodeling/models/Qwen3-32B
adapter="$artifact_root/model"
recurrent_state="$artifact_root/model/recurrent_state.pt"
jsonl="$project_dir/data/processed/malrule_c1_clean/test_same_template.jsonl"
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state_dir"
cd "$project_dir"
exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

rm -f "$state_dir/needs_attention" "$state_dir/complete"
test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B is incomplete"
test -f "$adapter/adapter_model.safetensors" || fail "adapter is missing"
test -f "$recurrent_state" || fail "recurrent state is missing"
test -f "$jsonl" || fail "C4 source is missing"

exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do heartbeat; sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 30
done

for seed in 13 97; do
  manifest="$state_dir/c4_pairs_seed${seed}_n150.json"
  output="$state_dir/c4_seed${seed}_n150.json"
  progress="$state_dir/progress_seed${seed}"
  test -f "$manifest" || fail "pair manifest missing for seed=$seed"
  if [[ ! -s "$output" ]]; then
    log "starting independent C4 pair-seed replication seed=$seed"
    (
      while true; do heartbeat; sleep 30; done
    ) &
    heartbeat_pid=$!
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
        --model "$model" --adapter "$adapter" \
        --recurrent-state "$recurrent_state" \
        --state-architecture layerwise --jsonl "$jsonl" \
        --pair-manifest "$manifest" --output-json "$output" \
        --state-token-alias '<|vision_pad|>' --max-length 1536 \
        --num-pairs 150 --seed "$seed" --progress-dir "$progress" \
        --progress-key "eedi-full-open-t3-state-path-lr2e6-round4250-pairseed${seed}-n150" \
        >> "$state_dir/seed${seed}.log" 2>&1 \
      || { kill "$heartbeat_pid" 2>/dev/null || true; wait "$heartbeat_pid" 2>/dev/null || true; fail "C4 replication failed seed=$seed"; }
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    log "completed independent C4 pair-seed replication seed=$seed"
  fi
done

"$python_bin" - "$state_dir" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for seed in (13, 42, 97):
    if seed == 42:
        path = pathlib.Path(
            "results/layerwise_unified_final_pipeline_20260731/eedi_full_open_t3_lr_scan/"
            "download/layerwise-eedi-full-open-t3-lr-scan-recovery1-seed42-20260802/"
            "layerwise-eedi-full-open-t3-lr2e6-seed42/eval/c4/recurrent_causal_gate.json"
        )
    else:
        path = root / f"c4_seed{seed}_n150.json"
    value = json.loads(path.read_text(encoding="utf-8"))["summary"]
    clean = value["clean"]
    rows.append(
        {
            "pair_seed": seed,
            "clean_n": value["clean_n"],
            "swap": clean["swap"]["flip_to_A_rate"],
            "random": clean["random"]["flip_to_A_rate"],
            "same_rule": clean["samemal"]["flip_to_A_rate"],
            "noop": clean["noop"]["flip_to_A_rate"],
            "D_specific": value["D_specific"],
            "D_damage": value["D_damage"],
            "scientific_gate_pass": value["scientific_gate_pass"],
            "checks": value["checks"],
        }
    )
payload = {
    "stage": "layerwise_c4_pair_seed_replication",
    "artifact": "eedi_full_open_t3__state_path__lr2e-6_round4250",
    "pair_seeds": rows,
    "mean_D_specific": statistics.fmean(r["D_specific"] for r in rows),
    "min_D_specific": min(r["D_specific"] for r in rows),
    "passing_seeds": sum(r["scientific_gate_pass"] for r in rows),
    "replication_gate_pass": (
        statistics.fmean(r["D_specific"] for r in rows) >= 0.15
        and sum(r["scientific_gate_pass"] for r in rows) >= 2
    ),
    "diagnostic_only": True,
    "final_exact_epoch_model_still_requires_frozen_evaluation": True,
}
target = root / "summary.json"
temporary = target.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

touch "$state_dir/complete"
heartbeat
log "independent C4 pair-seed replication complete"
