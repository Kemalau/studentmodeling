#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "T3_EEDI_TASKONLY_SHARD_FAILED line=%s rc=%s command=%s\n" "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR

: "${SHARD_INDEX:?SHARD_INDEX must be set}"
num_shards=4
[[ "$SHARD_INDEX" =~ ^[0-3]$ ]] || exit 2
case "$SHARD_INDEX" in
  0|1) expected_rows=224 ;;
  2|3) expected_rows=223 ;;
esac

export PATH="/tmp/studentmodeling-t3-specialist-sharded-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export LOCAL_MODEL=/tmp/Qwen3-32B
export ADAPTER=/mnt/amulet/studentmodeling/adapters/qwen3_32b_eedi_t3_qlora_1epoch_len1536/final_adapter
export EVAL_DATA=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
export OUT="$AMLT_OUTPUT_DIR/eval/t3_eedi_taskonly_shard$SHARD_INDEX"

mkdir -p "$HF_HUB_CACHE" "$OUT"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$LOCAL_MODEL" --workers 16
test -f "$LOCAL_MODEL/config.json"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$EVAL_DATA"
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = \
  329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2
test "$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)" = \
  bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8

python -u scripts/evaluate_formal_protocols.py \
  --model "$LOCAL_MODEL" --adapter "$ADAPTER" \
  --eval-jsonl "$EVAL_DATA" --output-dir "$OUT" \
  --tasks diagnose_misconception --sources eedi_mining \
  --shard-index "$SHARD_INDEX" --num-shards "$num_shards" \
  --max-length 1536 --max-new-tokens 160 --generation-batch-size 2 \
  --state-token-alias '<|vision_pad|>' --seed 42 \
  --save-unredacted-predictions

python - "$SHARD_INDEX" "$expected_rows" <<'PY'
import json, os, pathlib, sys
index, expected = map(int, sys.argv[1:])
root = pathlib.Path(os.environ["OUT"])
metrics = json.load(open(root / "metrics.json", encoding="utf-8"))
assert metrics["evaluation_shard"] == {"index": index, "count": 4}, metrics
assert metrics["unsharded_filtered_rows"] == 894, metrics
assert metrics["sampled_rows"] == expected == metrics["generation_rows"], metrics
assert metrics["classification_rows"] == 0 and metrics["bottleneck_mask"] is False, metrics
assert metrics["artifact_sha256"]["adapter_model"] == "329e95ad02a10cd9541f81ce3f220aaa521369f5761659a146fc2c54ad859bb2", metrics
assert metrics["artifact_sha256"]["evaluation_input"] == "bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8", metrics
rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == expected == len({str(row["example_id"]) for row in rows}), len(rows)
print(json.dumps({"status": "complete", "shard_index": index, "rows": expected}, sort_keys=True))
PY
