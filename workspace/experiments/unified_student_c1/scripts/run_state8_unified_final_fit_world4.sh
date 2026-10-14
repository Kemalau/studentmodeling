#!/usr/bin/env bash
set -euo pipefail

source_launcher=scripts/run_state8_unified_final_fit.sh
rendered_launcher=/tmp/run_state8_unified_final_fit_world4.sh
expected_source_sha=e2e75ef33f6a8acf39c850f79f05162e79582366381cc48a47d0173090791f5a
expected_renderer_sha=d1a1e80ebf38a02040c70dadb43c8b3827e3f950dc6490150986771696714213
expected_rendered_sha=11a95ee616b86c61bf3f4c2964a016c9e1bd267e358a82f56a6669857e21f315

test "$(sha256sum "$source_launcher" | cut -d' ' -f1)" = "$expected_source_sha"
test "$(sha256sum scripts/render_state8_finalfit_world4_launcher.py | cut -d' ' -f1)" = "$expected_renderer_sha"
observed_rendered_sha="$(python scripts/render_state8_finalfit_world4_launcher.py \
  --source "$source_launcher" --output "$rendered_launcher")"
test "$observed_rendered_sha" = "$expected_rendered_sha"
test "$(sha256sum "$rendered_launcher" | cut -d' ' -f1)" = "$expected_rendered_sha"
bash -n "$rendered_launcher"
exec bash "$rendered_launcher"
