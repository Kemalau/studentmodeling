#!/usr/bin/env bash
set -euo pipefail
C4_SHARD="${1:?shard required}"
export PATH="/tmp/studentmodeling-c4-freegen-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
C4_MODEL=/tmp/Qwen3-32B
C4_ADAPTER=/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
sha256sum -c manifests/c4_freegen_novel_20260905.sha256
python tests/test_c4_freegen_protocol.py
python tests/test_c4_freegen_decode.py
python scripts/copy_model_parallel.py --source /mnt/amulet/studentmodeling/models/Qwen3-32B --destination "$C4_MODEL" --workers 16
test -f "$C4_MODEL/model.safetensors.index.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u scripts/evaluate_c4_freegen_novel.py \
  --model "$C4_MODEL" --adapter "$C4_ADAPTER" \
  --manifest "$AMLT_CODE_DIR/manifests/c4_freegen_novel_seed42_20260905.json" \
  --shard "$C4_SHARD" --shards 3 --output-dir "$AMLT_OUTPUT_DIR/freegen"
