#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?A or B required}"
[[ "$ARM" == A || "$ARM" == B ]]
mkdir -p "$AMLT_OUTPUT_DIR/unseen_$ARM"
exec > >(tee -a "$AMLT_OUTPUT_DIR/runner_$ARM.log") 2>&1
trap 'status=$?; echo "[runner] exit=$status line=$LINENO at=$(date -u +%FT%TZ)"' EXIT
export PATH="/tmp/studentmodeling-c4-freegen-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MANIFEST="$AMLT_CODE_DIR/manifests/c4_unseen_rules_v1_20260906.json"
ADAPTER=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"][sys.argv[2]]["path"])' "$MANIFEST" "$ARM")
BASE=/mnt/amulet/studentmodeling/models/Qwen3-32B
sha256sum -c manifests/c4_unseen_rules_v1_feedsllm_20260906.sha256
test -r "$BASE/model.safetensors.index.json"
test -r "$ADAPTER/adapter_model.safetensors"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
python scripts/preflight_c4_unseen_rules.py --manifest "$MANIFEST" --adapter "$ADAPTER" --arm "$ARM"
python scripts/copy_model_parallel.py --source "$BASE" --destination /tmp/Qwen3-32B --workers 16
python -u scripts/evaluate_c4_unseen_rules.py --model /tmp/Qwen3-32B --adapter "$ADAPTER" \
    --manifest "$MANIFEST" --arm "$ARM" --output "$AMLT_OUTPUT_DIR/unseen_$ARM" \
    --runtime-report "$AMLT_OUTPUT_DIR/runtime_$ARM.json"
