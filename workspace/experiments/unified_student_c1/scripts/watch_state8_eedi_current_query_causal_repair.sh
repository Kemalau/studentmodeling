#!/usr/bin/env bash
set -euo pipefail

source_watcher=scripts/watch_state8_current_query_causal_repair.sh
rendered=/tmp/watch_state8_eedi_current_query_causal_repair.rendered.sh
expected=9e35b1cd3c763287549877960bd0e7fc2d74d58af945f65c568450f0252c8f7a
test "$(sha256sum "$source_watcher" | cut -d' ' -f1)" = "$expected"

python - "$source_watcher" "$rendered" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")
replacements = {
    "state8_current_query_causal_repair_20260811": "state8_eedi_current_query_causal_repair_20260811",
    "state8-current-query-causal-repair-20260811": "state8-eedi-current-query-causal-repair-20260811",
    "state8-current-query-updater-mse1-seed3072": "state8-eedi-current-query-updater-mse1-seed3072",
    "current_query_updater_mse1": "eedi_current_query_updater_mse1",
    "state8_current_query_causal_repair_comparison_v1": "state8_eedi_current_query_causal_repair_comparison_v1",
}
for old, new in replacements.items():
    count = text.count(old)
    assert count > 0, old
    text = text.replace(old, new)
output.write_text(text, encoding="utf-8")
PY
bash -n "$rendered"
bash "$rendered"
