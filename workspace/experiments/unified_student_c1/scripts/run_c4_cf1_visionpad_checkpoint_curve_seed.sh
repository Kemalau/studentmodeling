#!/usr/bin/env bash
set -euo pipefail

export PATH="/tmp/studentmodeling-c4-checkpoint-curve-venv/bin:${PATH}"
export PYTHONPATH="${AMLT_CODE_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

: "${TRAIN_SEED:?TRAIN_SEED is required}"
: "${SOURCE_RUN_TAG:?SOURCE_RUN_TAG is required}"

LOCAL_MODEL=/tmp/Qwen3-32B
C4_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean
PAIR_MANIFEST="${AMLT_CODE_DIR}/manifests/c4_stage_a_gate_pairs_seed42_n150_v1.json"
RUN_DIR="/mnt/amulet/studentmodeling/checkpoints/c4_cf1_visionpad_strict_seed${TRAIN_SEED}_${SOURCE_RUN_TAG}"
CURVE_DIR="${AMLT_OUTPUT_DIR}/curve"
STEPS=(250 500 750 1000 1250 1500)

check_sha() {
  local expected="$1"
  local path="$2"
  test -f "$path"
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected"
}

check_sha 079f7634cc54dfd55dfc234503bb5925ab734b1ca74aa2f2a9a17cc4375a2c2f "$C4_DATA/test_same_template.jsonl"
check_sha 405446c371185d23f06b4a0530c8ef4da4d1ac9addb696180f83bdf092402779 "$C4_DATA/test_cross_template.jsonl"
check_sha edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8 "$PAIR_MANIFEST"
check_sha df2451a6e7658cdf4e6ff0e5abe1e3b0708e63685265d46ade0ee97b09846be6 "$AMLT_CODE_DIR/scripts/evaluate_c1.py"
check_sha dc1cda0b72d4bbe8fdbbf05952ac5237c9e62f1afc0e33dac77b1bdbded8c147 "$AMLT_CODE_DIR/scripts/probe_student_state_smoke.py"
check_sha 1b8d4db48f8b228218c735e8fbb900ded7600e81ee9a55ccc40a6d358230aadd "$AMLT_CODE_DIR/scripts/state_swap_misconception_smoke.py"
check_sha 797e8c3c73805fa320f0adb384320cd21fd038c1c331c274e0d2a82a20a33d95 "$AMLT_CODE_DIR/scripts/summarize_stage_a_gate.py"

mkdir -p "$HF_HUB_CACHE" "$CURVE_DIR" "$AMLT_OUTPUT_DIR/audit"
{
  echo 'protocol=cf1_visionpad_strict_checkpoint_development_v1'
  echo "train_seed=$TRAIN_SEED"
  echo "source_run_tag=$SOURCE_RUN_TAG"
  echo "source_run_dir=$RUN_DIR"
  echo 'steps=250,500,750,1000,1250,1500'
  echo 'state_token_alias=<|vision_pad|>'
  sha256sum "$C4_DATA/test_same_template.jsonl" "$C4_DATA/test_cross_template.jsonl" \
    "$PAIR_MANIFEST" "$AMLT_CODE_DIR/scripts/evaluate_c1.py" \
    "$AMLT_CODE_DIR/scripts/probe_student_state_smoke.py" \
    "$AMLT_CODE_DIR/scripts/state_swap_misconception_smoke.py"
} | tee "$AMLT_OUTPUT_DIR/audit/input_manifest.txt"

rm -rf "$LOCAL_MODEL"
mkdir -p "$LOCAL_MODEL"
python "$AMLT_CODE_DIR/scripts/copy_model_parallel.py" \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

wait_for_checkpoint() {
  local step="$1"
  local checkpoint="$RUN_DIR/checkpoint-$step"
  local deadline=$(( $(date +%s) + 129600 ))
  while true; do
    if [[ -f "$checkpoint/adapter_model.safetensors" \
          && -f "$checkpoint/adapter_config.json" \
          && -f "$checkpoint/trainer_state.json" \
          && -f "$checkpoint/optimizer.pt" \
          && -f "$checkpoint/scheduler.pt" ]]; then
      local first second
      first="$(sha256sum "$checkpoint/adapter_model.safetensors" | cut -d' ' -f1)"
      sleep 15
      second="$(sha256sum "$checkpoint/adapter_model.safetensors" | cut -d' ' -f1)"
      if [[ "$first" = "$second" ]]; then
        echo "$second"
        return 0
      fi
    fi
    if (( $(date +%s) >= deadline )); then
      echo "Timed out waiting for complete checkpoint-$step in $RUN_DIR" >&2
      return 1
    fi
    echo "[wait] seed=$TRAIN_SEED checkpoint=$step not complete" >&2
    sleep 60
  done
}

for step in "${STEPS[@]}"; do
  checkpoint="$RUN_DIR/checkpoint-$step"
  output_dir="$CURVE_DIR/step_$step"
  mkdir -p "$output_dir"
  adapter_sha="$(wait_for_checkpoint "$step")"
  printf '%s\n' "$adapter_sha" > "$output_dir/checkpoint_adapter.sha256"
  cp "$checkpoint/trainer_state.json" "$output_dir/trainer_state.json"

  if [[ ! -f "$output_dir/response_cross_masked.json" ]]; then
    python -u "$AMLT_CODE_DIR/scripts/evaluate_c1.py" \
      --model "$LOCAL_MODEL" --adapter "$checkpoint" \
      --train-jsonl "$C4_DATA/train.jsonl" \
      --eval-jsonl "$C4_DATA/test_cross_template.jsonl" \
      --output-json "$output_dir/response_cross_masked.json" \
      --tasks malrule_next_response --max-samples-per-task 300 --max-length 1536 \
      --batch-size 2 --bottleneck-mask --bottleneck-style late_fusion \
      --state-token-alias '<|vision_pad|>' --load-4bit --progress-every 50
  fi

  if [[ ! -f "$output_dir/state_probe_masked.json" ]]; then
    python -u "$AMLT_CODE_DIR/scripts/probe_student_state_smoke.py" \
      --model "$LOCAL_MODEL" --adapter "$checkpoint" \
      --train-jsonl "$C4_DATA/train.jsonl" \
      --same-jsonl "$C4_DATA/test_same_template.jsonl" \
      --cross-jsonl "$C4_DATA/test_cross_template.jsonl" \
      --output-json "$output_dir/state_probe_masked.json" \
      --task malrule_next_response --layers 0,8,16,24,32,40,48,56,63 \
      --train-per-label 50 --eval-per-label 15 --max-length 1536 --batch-size 2 \
      --epochs 250 --lr 0.1 --weight-decay 1e-4 --seed 42 \
      --state-token-alias '<|vision_pad|>' --bottleneck-mask --bottleneck-style late_fusion
  fi

  if [[ ! -f "$output_dir/state_swap_masked.json" ]]; then
    python -u "$AMLT_CODE_DIR/scripts/state_swap_misconception_smoke.py" \
      --model "$LOCAL_MODEL" --adapter "$checkpoint" \
      --jsonl "$C4_DATA/test_same_template.jsonl" \
      --output-json "$output_dir/state_swap_masked.json" \
      --task malrule_next_response --swap-target response --layers all --num-pairs 150 \
      --pair-manifest "$PAIR_MANIFEST" --max-length 1536 --seed 42 \
      --state-token-alias '<|vision_pad|>' --bottleneck-mask --bottleneck-style late_fusion
  fi

  python - "$output_dir" <<'PY'
import json, sys
from pathlib import Path
from summarize_stage_a_gate import TASK, clean_swap_summary

root = Path(sys.argv[1])
response = json.loads((root / "response_cross_masked.json").read_text())
probe = json.loads((root / "state_probe_masked.json").read_text())
layers = probe["metrics"]["state"]
best_layer, best = max(layers.items(), key=lambda item: (item[1]["cross"]["accuracy"], item[1]["cross"]["macro_f1"]))
swap = clean_swap_summary(root / "state_swap_masked.json")
clean = swap["clean"]
summary = {
    "response_cross_accuracy": response["model"]["metrics"][TASK]["accuracy"],
    "state_probe_best_cross_accuracy": best["cross"]["accuracy"],
    "state_probe_best_layer": int(best_layer),
    "clean_n": swap["clean_n"],
    "swap": clean["swap"]["flip_to_A_rate"],
    "random": clean["random"]["flip_to_A_rate"],
    "same_rule": clean["samemal"]["flip_to_A_rate"],
    "noop": clean["noop"]["flip_to_A_rate"],
}
summary["D_specific"] = summary["swap"] - summary["random"]
(root / "step_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
done

python "$AMLT_CODE_DIR/scripts/summarize_cf1_checkpoint_development_curve.py" \
  --curve-dir "$CURVE_DIR" --run-dir "$RUN_DIR" \
  --output-json "$AMLT_OUTPUT_DIR/curve_summary.json" \
  --output-md "$AMLT_OUTPUT_DIR/curve_summary.md"
sha256sum "$AMLT_OUTPUT_DIR/curve_summary.json" "$AMLT_OUTPUT_DIR/curve_summary.md" \
  | tee "$AMLT_OUTPUT_DIR/audit/output_manifest.txt"
