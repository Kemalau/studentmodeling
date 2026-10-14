#!/usr/bin/env bash
set -euo pipefail

project=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
source="$project/scripts/watch_state8_eedi_curriculum_milestones.sh"
test "$(sha256sum "$source" | cut -d' ' -f1)" = c3e3b4b89d9bf0706b3682a98e3f02a2862f16afa8a8aa3b3de403989b1ad831

rendered=/tmp/watch_state8_eedi_current_query_curriculum_milestones.rendered.sh
python - "$source" "$rendered" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")

def replace(old: str, new: str, count: int = 1) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

replace(
    "results/state8_diagnosis_first_eedi_curriculum_20260812",
    "results/state8_diagnosis_first_eedi_current_query_curriculum_20260812",
)
replace(
    "state8-diagnosis-first-eedi-curriculum-recovery1-20260812",
    "state8-diagnosis-first-eedi-current-query-curriculum-20260812",
)
replace(
    "state8-diagnosis-first-eedi-curriculum-seed83",
    "state8-diagnosis-first-eedi-current-query-curriculum-seed83",
)
replace(
    "diagnosis_first_eedi_curriculum_seed83_single_ga4",
    "diagnosis_first_eedi_current_query_curriculum_seed83_single_ga4",
    1,
)
replace(
    "state8_diagnosis_first_eedi_curriculum_20260812",
    "state8_diagnosis_first_eedi_current_query_curriculum_20260812",
    1,
)
output.write_text(text, encoding="utf-8")
PY
bash -n "$rendered"
exec bash "$rendered"
