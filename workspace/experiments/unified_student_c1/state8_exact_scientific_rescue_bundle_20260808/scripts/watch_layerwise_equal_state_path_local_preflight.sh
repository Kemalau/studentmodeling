#!/usr/bin/env bash
set -uo pipefail

project_dir="/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1"
export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds}"
state_slug="${STATE_SLUG:-equal_state_path}"
loss_profile="${LOSS_PROFILE_EXPECTED:-state_path}"
artifact_label="${ARTIFACT_LABEL:-equal-state-path}"
checkpoint_round="${CHECKPOINT_ROUND:-3250}"
local_preflight_subdir="${LOCAL_PREFLIGHT_SUBDIR:-local_round${checkpoint_round}_preflight}"
state_dir="$project_dir/results/layerwise_unified_final_pipeline_20260731/$state_slug/$local_preflight_subdir"
checkpoint="$state_dir/checkpoint-round-$checkpoint_round"
adapter="$checkpoint/adapter"
model="/home/v-yuejiali/studentmodeling/models/Qwen3-32B"
eval_jsonl="$project_dir/data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl"
c4_jsonl="$project_dir/data/processed/malrule_c1_clean/test_same_template.jsonl"
pair_manifest="$project_dir/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
pair_manifest_n50="$state_dir/c4_pair_manifest_n50.json"
account=csnewsandfeeds4150361735
container=amulet
prefix="${REMOTE_PREFIX:-studentmodeling/diagnostics/layerwise_task_retention_service_recovery/equal_t3_t6/$loss_profile/lr5e-7_start12000_stop3250_32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440/checkpoint-round-$checkpoint_round}"
python_bin="/home/v-yuejiali/.venvs/studentmodeling/bin/python"
mkdir -p "$state_dir" "$adapter"
cd "$project_dir"

exec 9>"$state_dir/controller.lock"
flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"

heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

download_blob(){
  local name="$1" output="$2"
  az storage blob download --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$name" --file "$output" \
    --overwrite -o none >> "$state_dir/download.log" 2>&1
}

receipt="$checkpoint/checkpoint_completion.json"
while [[ ! -s "$receipt" ]]; do
  heartbeat
  exists="$(az storage blob exists --auth-mode login --account-name "$account" \
    --container-name "$container" --name "$prefix/checkpoint_completion.json" \
    --query exists -o tsv 2>/dev/null || true)"
  if [[ "$exists" == true ]]; then
    mkdir -p "$checkpoint"
    download_blob "$prefix/checkpoint_completion.json" "$receipt" \
      || { sleep 120; continue; }
  else
    sleep 120
  fi
done

"$python_bin" - "$receipt" "$checkpoint_round" <<'PY' || fail "checkpoint completion receipt invalid"
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("status") == "complete" and x.get("round") == int(sys.argv[2]), x
assert x.get("world_size") == 4, x
assert set(x.get("sha256") or {}) == {"adapter_model", "recurrent_state"}, x
PY

if [[ ! -s "$adapter/adapter_model.safetensors" ]]; then
  download_blob "$prefix/adapter/adapter_model.safetensors" \
    "$adapter/adapter_model.safetensors" || fail "adapter download failed"
fi
if [[ ! -s "$adapter/adapter_config.json" ]]; then
  download_blob "$prefix/adapter/adapter_config.json" \
    "$adapter/adapter_config.json" || fail "adapter config download failed"
fi
if [[ ! -s "$checkpoint/recurrent_state.pt" ]]; then
  download_blob "$prefix/recurrent_state.pt" "$checkpoint/recurrent_state.pt" \
    || fail "recurrent-state download failed"
fi

"$python_bin" - "$receipt" "$adapter/adapter_model.safetensors" \
  "$checkpoint/recurrent_state.pt" "$state_dir/artifact_audit.json" "$loss_profile" <<'PY' \
  || fail "downloaded round-3250 artifact hash audit failed"
import hashlib, json, pathlib, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
observed = {
    "adapter_model": digest(sys.argv[2]),
    "recurrent_state": digest(sys.argv[3]),
}
assert observed == receipt["sha256"], (observed, receipt)
out = {
    "stage": "layerwise_checkpoint_local_artifact_audit",
    "loss_profile": sys.argv[5],
    "status": "passed",
    "round": int(receipt["round"]),
    "world_size": 4,
    "sha256": observed,
}
path = pathlib.Path(sys.argv[4])
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
PY

test -f "$model/model.safetensors.index.json" || fail "local Qwen3-32B missing"
test -f "$eval_jsonl" || fail "frozen formal manifest missing"
test -f "$c4_jsonl" || fail "frozen C4 source missing"
test -f "$pair_manifest" || fail "frozen C4 pair manifest missing"

# All local checkpoint screens share one physical A100.  Hold a process-level
# lock across C4 and the formal screen so independently completed remote arms
# queue locally instead of racing each other into OOM.  Keep the heartbeat
# fresh while waiting; this lock changes scheduling only and never model state.
exec 10>"$project_dir/results/local_gpu_readonly_preflight.lock"
while ! flock -n 10; do
  heartbeat
  sleep 60
done
# The lock coordinates only this pipeline; other workspace owners can still
# use the shared A100.  Wait for an actually empty compute-process list before
# loading Qwen3-32B.  This is deliberately conservative because the local
# machine is for opportunistic inference, not a required training dependency.
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do
  heartbeat
  sleep 60
done

if [[ ! -s "$pair_manifest_n50" ]]; then
  "$python_bin" - "$pair_manifest" "$pair_manifest_n50" <<'PY' \
    || fail "local C4 n=50 pair-manifest derivation failed"
import hashlib, json, pathlib, sys
source = pathlib.Path(sys.argv[1])
payload = json.loads(source.read_text(encoding="utf-8"))
assert payload.get("num_pairs") == 150 and len(payload.get("pairs") or []) == 150, payload
derived = {
    **payload,
    "num_pairs": 50,
    "pairs": payload["pairs"][:50],
    "diagnostic_only": True,
    "selection": "ordered_prefix",
    "parent_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
path = pathlib.Path(sys.argv[2])
temporary = path.with_suffix(path.suffix + ".partial")
temporary.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
fi

if [[ ! -s "$state_dir/c4/recurrent_causal_gate.json" ]]; then
  mkdir -p "$state_dir/c4/progress"
  heartbeat
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_causal_gate.py \
      --model "$model" --adapter "$adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --jsonl "$c4_jsonl" \
      --pair-manifest "$pair_manifest_n50" \
      --output-json "$state_dir/c4/recurrent_causal_gate.json" \
      --state-token-alias '<|vision_pad|>' --max-length 1536 \
      --num-pairs 50 --seed 42 \
      --progress-dir "$state_dir/c4/progress" \
      --progress-key "$artifact_label-round${checkpoint_round}-local-n50" \
      >> "$state_dir/c4/live.log" 2>&1 \
    || fail "local round-3250 C4 n=50 preflight failed"
fi

if [[ ! -s "$state_dir/formal_n10/metrics.json" ]]; then
  mkdir -p "$state_dir/formal_n10"
  heartbeat
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -u scripts/evaluate_recurrent_formal_protocols.py \
      --model "$model" --adapter "$adapter" \
      --recurrent-state "$checkpoint/recurrent_state.pt" \
      --state-architecture layerwise --eval-jsonl "$eval_jsonl" \
      --output-dir "$state_dir/formal_n10" --state-token-alias '<|vision_pad|>' \
      --max-samples-per-group 10 --max-length 1536 --max-new-tokens 128 \
      --state-batch-size 2 --candidate-row-batch-size 2 --seed 42 \
      --save-unredacted-predictions >> "$state_dir/formal_n10/live.log" 2>&1 \
    || fail "local round-3250 formal n=10 preflight failed"
fi

"$python_bin" - "$state_dir/c4/recurrent_causal_gate.json" \
  "$state_dir/formal_n10/metrics.json" \
  "$state_dir/formal_n10/predictions.jsonl" "$state_dir/summary.json" \
  "$loss_profile" "$checkpoint_round" "$artifact_label" <<'PY' \
  || fail "local round-3250 preflight summary failed"
import collections, json, pathlib, sys
c4 = json.load(open(sys.argv[1], encoding="utf-8"))["summary"]
formal = json.load(open(sys.argv[2], encoding="utf-8"))
predictions = [json.loads(line) for line in open(sys.argv[3], encoding="utf-8") if line.strip()]
t3 = [row for row in predictions if row.get("task") == "diagnose_misconception"]
outputs = collections.Counter(str(row.get("raw_output", "")) for row in t3)
clean = c4["clean"]
out = {
    "stage": "layerwise_checkpoint_local_preflight",
    "artifact_label": sys.argv[7],
    "loss_profile": sys.argv[5],
    "diagnostic_only": True,
    "not_a_formal_gate": True,
    "checkpoint_round": int(sys.argv[6]),
    "formal_sampled_rows": formal.get("sampled_rows"),
    "metrics_by_group": formal.get("metrics_by_group"),
    "T3_output_diversity": {
        "support": len(t3),
        "unique_outputs": len(outputs),
        "top_output_share": (outputs.most_common(1)[0][1] / len(t3)) if t3 else None,
    },
    "C4_n50": {
        "clean_n": c4["clean_n"],
        "swap": clean["swap"]["flip_to_A_rate"],
        "random": clean["random"]["flip_to_A_rate"],
        "same_rule": clean["samemal"]["flip_to_A_rate"],
        "noop": clean["noop"]["flip_to_A_rate"],
        "D_specific": c4["D_specific"],
    },
}
path = pathlib.Path(sys.argv[4])
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY

touch "$state_dir/complete"
heartbeat
log "$artifact_label round-$checkpoint_round local read-only preflight complete"
