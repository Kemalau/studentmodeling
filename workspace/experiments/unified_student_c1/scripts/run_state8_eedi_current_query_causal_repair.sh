#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_CURRENT_QUERY_CAUSAL_REPAIR_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

source_runner=scripts/run_state8_source_anchor_screen.sh
patched_runner=/tmp/run_state8_eedi_current_query_causal_repair.rendered.sh
expected_source=8e0244fc8aa63408e4ccd6cbe2495fdafe512242bb96ca50484ab9d93059a379
test "$(sha256sum "$source_runner" | cut -d' ' -f1)" = "$expected_source"
test "$(sha256sum scripts/state8_current_query_routing.py | cut -d' ' -f1)" = b48bfabc2f48df1a422f46d52a9bfda9c915dce3385c50391c4ba3ab623ac875
test "$(sha256sum scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py | cut -d' ' -f1)" = 967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f

python - "$source_runner" "$patched_runner" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")

def replace(old: str, new: str, count: int = 1) -> None:
    global text
    observed = text.count(old)
    assert observed == count, (old, observed, count)
    text = text.replace(old, new)

replace(
    "export RUN_TAG=source_anchor_updater_mse1_v1",
    "export RUN_TAG=source_anchor_updater_mse1_eedi_current_query_v1",
)
replace(
    "export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "export STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256=967711843054673eab67e43a8d5392ac7329a31bf5005cc1ec5be8a60507244f",
)
replace(
    'bash -n "$rendered"\nbash "$rendered"',
    '''python - "$rendered" <<'PY2'
from pathlib import Path
import sys
path = Path(sys.argv[1])
value = path.read_text(encoding="utf-8")
old = "FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py"
new = "FORMAL_EVALUATOR=scripts/evaluate_recurrent_formal_protocols_diagnosis_first_eedi_current_query.py"
assert value.count(old) == 1, value.count(old)
path.write_text(value.replace(old, new), encoding="utf-8")
PY2
bash -n "$rendered"
bash "$rendered"''',
)
output.write_text(text, encoding="utf-8")
PY

export STATE8_CURRENT_QUERY_ROUTING=eedi_only
bash -n "$patched_runner"
bash "$patched_runner"
