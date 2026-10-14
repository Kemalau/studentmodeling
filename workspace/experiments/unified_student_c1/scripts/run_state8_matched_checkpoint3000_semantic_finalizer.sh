#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_MATCHED_CP3000_FINALIZER_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${STATE8_RESCUE_MODE:?STATE8_RESCUE_MODE is required}"
test "$STATE8_RESCUE_MODE" = semantic_preserve

source=scripts/run_state8_matched_checkpoint2000_rescue.sh
rendered=/tmp/run_state8_matched_checkpoint3000_semantic_finalizer.rendered.sh
test "$(sha256sum "$source" | cut -d' ' -f1)" = c04d4059c8f94a4cb3b1146f2ef470c8943acf5b44e8e665c422265e45bd29ab

python - "$source" "$rendered" <<'PY'
import hashlib
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "c04d4059c8f94a4cb3b1146f2ef470c8943acf5b44e8e665c422265e45bd29ab"
text = raw.decode()

def replace(old: str, new: str, count: int = 1) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

# Bind the finalizer to the history-merge fix. The frozen training launcher had
# two references to the previous trainer digest: its direct preflight and the
# hash inserted into the rendered inner launcher.
replace(
    "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
    "7d5e4966a5317733fe5d5f32cc6730eb82f4ad78dedd8fa6a6967ee4aefb4e6d",
    2,
)

# Apply the already-audited checkpoint-3000 binding. The existing run directory
# contains an atomic checkpoint-1000, so the single-device selector restores it
# and the trainer performs zero optimizer steps before validation/finalization.
replace("checkpoint-2000", "checkpoint-3000", 1)
replace("receipt['step']==2000", "receipt['step']==3000", 1)
replace("si.get('step')==2000", "si.get('step')==3000", 1)
replace("export TRAIN_SEED=2066", "export TRAIN_SEED=3066", 1)
replace("checkpoint2000", "checkpoint3000", 4)
replace("'source_checkpoint_step':2000", "'source_checkpoint_step':3000", 1)
output.write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
PY

bash -n "$rendered"
bash "$rendered"
