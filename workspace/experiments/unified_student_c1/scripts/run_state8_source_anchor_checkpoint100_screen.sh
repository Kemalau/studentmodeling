#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_SOURCE_ANCHOR_CP100_SCREEN_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

# Keep the checkpoint-100 screen byte-for-byte protocol matched to the
# checkpoint-20 screen.  Only the immutable checkpoint and audited receipt
# step change; prompts, rows, decoding, and C4 conditions remain identical.
source=scripts/run_state8_source_anchor_checkpoint20_screen.sh
rendered=/tmp/run_state8_source_anchor_checkpoint100_screen.rendered.sh
source_sha=757e58405b9f96cd18d73ef840eb779d2848f7dbc51eb664e661e01ef1fea4ab
test "$(sha256sum "$source" | cut -d' ' -f1)" = "$source_sha"

python - "$source" "$rendered" <<'PY'
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")

def replace(old: str, new: str, count: int) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

replace("checkpoint-20", "checkpoint-100", 1)
replace("--arm \"$SOURCE_ANCHOR_ARM\" --step 20", "--arm \"$SOURCE_ANCHOR_ARM\" --step 100", 1)
replace("state8_source_anchor_checkpoint20_screen_v1", "state8_source_anchor_checkpoint100_screen_v1", 1)
replace("'checkpoint_step':20", "'checkpoint_step':100", 1)
replace("state8_source_anchor_cp20_eval", "state8_source_anchor_cp100_eval", 1)
replace("state8_source_anchor_cp20_diagnosis_pairs_n20", "state8_source_anchor_cp100_diagnosis_pairs_n20", 1)
output.write_text(text, encoding="utf-8")
PY

bash -n "$rendered"
bash "$rendered"
