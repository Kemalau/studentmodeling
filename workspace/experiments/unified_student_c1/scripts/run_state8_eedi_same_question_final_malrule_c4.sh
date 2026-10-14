#!/usr/bin/env bash
set -euo pipefail
source=scripts/run_state8_final_c4.sh
rendered=/tmp/run_state8_eedi_same_question_final_malrule_c4.sh
test "$(sha256sum scripts/render_state8_eedi_same_question_final_c4.py | cut -d' ' -f1)" = 04d5ad4f5e76cdee1ef5b095292b3602befab983f75b568d99f0bbece29ba03f
observed="$(python scripts/render_state8_eedi_same_question_final_c4.py --source "$source" --output "$rendered")"
test "$observed" = 415c9b67e48aa9c133d808a1e928ee8a87b95e66398cca16c46b9182b86c605b
test "$(sha256sum "$rendered" | cut -d' ' -f1)" = "$observed"
bash -n "$rendered"
exec bash "$rendered"
