#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_LRFIX_V4_SMOKE_RECOVERY_AUDIT_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-state8-lrfix-v4-smoke-recovery-audit-venv}"
export PATH="$VENV_ROOT/bin:$PATH"
python(){ "$VENV_ROOT/bin/python" "$@"; }
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache HF_HUB_CACHE=/tmp/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_VERBOSITY=error

MODEL=/mnt/amulet/studentmodeling/models/Qwen3-32B
RUN_ROOT=/mnt/amulet/studentmodeling/checkpoints/state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_seed8801
CHECKPOINT_ROOT="$RUN_ROOT/checkpoint-200"
ADAPTER="$CHECKPOINT_ROOT/adapter"
STATE="$CHECKPOINT_ROOT/recurrent_state.pt"
RECEIPT="$CHECKPOINT_ROOT/checkpoint_completion.json"
COMPLETION="$RUN_ROOT/completion.json"
REPAIR="$RUN_ROOT/profile_ownership_repair.json"
ROWS=manifests/dbe_student_profile_v1/test.jsonl
EXPECTED_ADAPTER=f23b0dc3a4c840ff3e0ef9192dd1d33700950328d7f3beb992cff4d3687ffd43
EXPECTED_STATE=24c4832dc61cbf53deb2d9ad750ec1edc18226522fdfe35f5d584ca9ff88cd78
EXPECTED_TERMINAL_STATE=9a524250fe808cbe738ed802a235d507c83f5c5e4ae0827532a5ba7f3dbc45a7
EXPECTED_ROWS=d0316d845a6e1be2f4d24418ebdcd22adb579aed5858aa103da6427e506f5b03
OUTPUT="$AMLT_OUTPUT_DIR/checkpoint_early_audit/metrics.json"

for path in "$MODEL/model.safetensors.index.json" \
  "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" \
  "$STATE" "$RECEIPT" "$COMPLETION" "$REPAIR" "$ROWS"; do
  test -f "$path"
done
test "$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER"
test "$(sha256sum "$STATE" | cut -d' ' -f1)" = "$EXPECTED_STATE"
test "$(sha256sum "$ROWS" | cut -d' ' -f1)" = "$EXPECTED_ROWS"

python - "$RECEIPT" "$COMPLETION" "$REPAIR" \
  "$EXPECTED_ADAPTER" "$EXPECTED_STATE" "$EXPECTED_TERMINAL_STATE" <<'PY'
import json, sys
receipt, completion, repair = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:4])
adapter, state, terminal_state = sys.argv[4:]
assert receipt == {
    "sha256": {"adapter_model": adapter, "recurrent_state": state},
    "status": "complete",
    "step": 200,
}, receipt
assert completion["status"] == "diagnostic_complete", completion
assert completion["diagnostic"] is True and completion["micro_steps"] == 200, completion
assert completion["optimizer_steps"] == 50 and completion["history_integrity_pass"] is True, completion
assert completion["global_positions_seen"] == list(range(200)), completion
assert completion["sha256"]["adapter_model"] == adapter, completion
assert completion["sha256"]["recurrent_state"] == terminal_state, completion
assert repair["status"] == "diagnostic_complete" and repair["applied_rows"] == 140, repair
assert repair["head_learning_rate"] == 3e-4 and repair["feature_normalization"] == "layernorm", repair
assert repair["minimum_kc_train_support"] == 4 and repair["bias_prior_strength"] == 8.0, repair
assert repair["final_inference_route"] == "strict_state_only", repair
assert repair["raw_history_visible_to_decoder"] is False, repair
PY

python -m pytest -q \
  tests/test_early_audit_state8_profile_ownership_checkpoint.py \
  tests/test_evaluate_state8_reusable_interface.py
mkdir -p "$(dirname "$OUTPUT")"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

python -u scripts/early_audit_state8_profile_ownership_checkpoint.py \
  --model "$MODEL" --adapter "$ADAPTER" --checkpoint "$STATE" \
  --rows "$ROWS" --output "$OUTPUT" --state-token-alias '<|vision_pad|>' \
  --max-length 1536 --batch-size 4 --seed 20260820 \
  --permutation-replicates 2000

python - "$OUTPUT" "$EXPECTED_ADAPTER" "$EXPECTED_STATE" "$EXPECTED_ROWS" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["stage"] == "state8_profile_ownership_checkpoint_early_audit_v1", value
assert value["status"] == "complete" and value["diagnostic_only"] is True, value
assert value["checkpoint_step"] == 200 and value["students"] == 88, value
assert value["supported_student_kc_cells"] == 634, value
assert value["strict_state_only"] is True, value
assert value["profile_ownership_feature_normalization"] == "layernorm", value
assert value["manifest_sha256"] == sys.argv[4], value
assert value["artifact_sha256"] == {
    "adapter_model": sys.argv[2], "recurrent_state": sys.argv[3]
}, value
assert value["ownership_permutation_replicates"] == 2000, value
PY

cp "$COMPLETION" "$AMLT_OUTPUT_DIR/diagnostic_completion.json"
cp "$REPAIR" "$AMLT_OUTPUT_DIR/profile_ownership_repair.json"
python - "$AMLT_OUTPUT_DIR/recovery_receipt.json" \
  "$EXPECTED_ADAPTER" "$EXPECTED_STATE" "$EXPECTED_ROWS" <<'PY'
import hashlib, json, pathlib, sys
target = pathlib.Path(sys.argv[1])
evaluator = pathlib.Path("scripts/early_audit_state8_profile_ownership_checkpoint.py")
target.write_text(json.dumps({
    "status": "complete",
    "mode": "evaluation_only_from_immutable_checkpoint_200",
    "training_repeated": False,
    "official_xes_test_inspected": False,
    "adapter_sha256": sys.argv[2],
    "recurrent_state_sha256": sys.argv[3],
    "dbe_rows_sha256": sys.argv[4],
    "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n")
PY
