#!/usr/bin/env bash
set -euo pipefail
source=scripts/run_state8_final_formal_shard.sh
rendered=/tmp/run_state8_eedi_same_question_final_formal_shard.sh
test "$(sha256sum scripts/render_state8_eedi_same_question_final_formal_shard.py | cut -d' ' -f1)" = 0f01dd9785453ba1d2722a047ae729546a009433cb62e5c6b27fa7a3d19adf49
observed="$(python scripts/render_state8_eedi_same_question_final_formal_shard.py --source "$source" --output "$rendered")"
test "$observed" = c496fba0f923fa9ca9cb910bde8e7640cc95dba971a9e6fc0cf5ad3ec55c4f98
test "$(sha256sum "$rendered" | cut -d' ' -f1)" = "$observed"
bash -n "$rendered"
exec bash "$rendered"
