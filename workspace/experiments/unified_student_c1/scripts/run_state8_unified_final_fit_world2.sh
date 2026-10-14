#!/usr/bin/env bash
set -euo pipefail

source_launcher=scripts/run_state8_unified_final_fit.sh
rendered_launcher=/tmp/run_state8_unified_final_fit_world2.sh
expected_source_sha=e2e75ef33f6a8acf39c850f79f05162e79582366381cc48a47d0173090791f5a
expected_renderer_sha=1c9fffc25c12c91ef3c52f1327dc117221c12c75083672e32499dfd9b7c23ffb
expected_rendered_sha=d3947f59834cb092e721a5c22ec80a0906f8297c8bea990a15526ad25821984e

test "$(sha256sum "$source_launcher" | cut -d' ' -f1)" = "$expected_source_sha"
test "$(sha256sum scripts/render_state8_finalfit_world2_launcher.py | cut -d' ' -f1)" = "$expected_renderer_sha"
observed_rendered_sha="$(python scripts/render_state8_finalfit_world2_launcher.py \
  --source "$source_launcher" --output "$rendered_launcher")"
test "$observed_rendered_sha" = "$expected_rendered_sha"
test "$(sha256sum "$rendered_launcher" | cut -d' ' -f1)" = "$expected_rendered_sha"
bash -n "$rendered_launcher"
exec bash "$rendered_launcher"
