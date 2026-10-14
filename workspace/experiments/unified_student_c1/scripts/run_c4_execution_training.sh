#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?A or B required}"
mkdir -p "$AMLT_OUTPUT_DIR/execution_$ARM"
exec > >(tee -a "$AMLT_OUTPUT_DIR/execution_$ARM/runner.log") 2>&1
trap 'status=$?; echo "[runner] exit=$status line=$LINENO at=$(date -u +%FT%TZ)"' EXIT
export PATH="/tmp/studentmodeling-c4-freegen-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/tmp/hf_cache HF_HUB_DISABLE_XET=1
BASE=/mnt/amulet/studentmodeling/models/Qwen3-32B
ADAPTER=/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter
MANIFEST="$AMLT_CODE_DIR/manifests/c4_execution_training_v1_20260906.json"
sha256sum -c "${C4_SOURCE_HASH_MANIFEST:-manifests/c4_execution_training_v1_20260906.sha256}"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
test -r "$ADAPTER/adapter_model.safetensors"
mkdir -p "$AMLT_OUTPUT_DIR/execution_$ARM"
python scripts/copy_model_parallel.py --source "$BASE" --destination /tmp/Qwen3-32B --workers 16
export C4_TEST_TOKENIZER=/tmp/Qwen3-32B
python tests/test_c4_execution_training.py
python -u scripts/run_c4_execution_training.py --model /tmp/Qwen3-32B --adapter "$ADAPTER" \
  --manifest "$MANIFEST" --output "$AMLT_OUTPUT_DIR/execution_$ARM" --arm "$ARM"
