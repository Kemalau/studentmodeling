#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_SOURCE_ANCHOR_CP100_WATCHER_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

source=scripts/watch_state8_source_anchor_checkpoint20_screen.sh
rendered=/tmp/watch_state8_source_anchor_checkpoint100_screen.rendered.sh
source_sha=186be7cde8a544e21a227e6ef51df160ef8be4e9c31f053dc93f8d435732029b
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

replace("feeds_state8_source_anchor_checkpoint20_screen_20260811.yaml", "feeds_state8_source_anchor_checkpoint100_screen_20260811.yaml", 1)
replace("state8_source_anchor_checkpoint20_screen_20260811", "state8_source_anchor_checkpoint100_screen_20260811", 1)
replace("state8-source-anchor-cp20", "state8-source-anchor-cp100", 5)
replace("checkpoint-20", "checkpoint-100", 3)
replace("checkpoint20", "checkpoint100", 1)

needle = '''  amlt run "$config" "${selectors[@]}" --dump -t Feeds > "$state/$next.preflight.yaml" \\
    2> "$state/$next.preflight.stderr" || { flock -u 8; return 1; }\n'''
replacement = needle + '''  validator_args=()\n  for job in "${missing[@]}"; do validator_args+=(--job "$job"); done\n  "$python_bin" scripts/validate_state8_source_anchor_checkpoint100_screen_amlt.py \\
    --config "$state/$next.preflight.yaml" "${validator_args[@]}" \\
    >> "$state/controller.log" 2>&1 || { flock -u 8; return 1; }\n'''
replace(needle, replacement, 1)
output.write_text(text, encoding="utf-8")
PY

bash -n "$rendered"
bash "$rendered"
