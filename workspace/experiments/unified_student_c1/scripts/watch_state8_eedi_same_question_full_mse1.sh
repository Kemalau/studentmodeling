#!/usr/bin/env bash
set -euo pipefail

source=scripts/watch_state8_eedi_same_question_causal_repair.sh
rendered=/tmp/watch_state8_eedi_same_question_full_mse1.rendered.sh
python - "$source" "$rendered" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")
replacements = {
    "state8_eedi_same_question_causal_repair_20260811":
        "state8_eedi_same_question_full_mse1_20260811",
    "state8-eedi-same-question-causal-repair-20260811":
        "state8-eedi-same-question-full-mse1-20260811",
    "state8-eedi-same-question-updater-mse1-seed3072":
        "state8-eedi-same-question-full-mse1-seed3071",
    "state8_eedi_same_question_causal_repair_semantic_summary_v1":
        "state8_eedi_same_question_full_mse1_semantic_summary_v1",
}
for old, new in replacements.items():
    count = text.count(old)
    assert count > 0, old
    text = text.replace(old, new)
output.write_text(text, encoding="utf-8")
PY
bash -n "$rendered"
bash "$rendered"
