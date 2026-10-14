#!/usr/bin/env bash
set -euo pipefail

source_launcher=scripts/run_state8_unified_final_fit.sh
rendered_launcher=/tmp/run_state8_unified_final_fit_world2.sh
expected_source_sha=434df629e5a05746fe7c58547a7fc9e8659dc18462d71a66858a5405bf18f1dc
expected_renderer_sha=48cf48de1d75d61c8bec04f59751fbff7f78087bd8d1ad7f6aa0e80ab662213b
expected_rendered_sha=52df1f186cbd0917a9e763edc49ae50f16e8140f46cc5d6b68153475b0d4b0b4

test "$(sha256sum "$source_launcher" | cut -d' ' -f1)" = "$expected_source_sha"
test "$(sha256sum scripts/render_state8_finalfit_world2_launcher.py | cut -d' ' -f1)" = "$expected_renderer_sha"
observed_rendered_sha="$(python scripts/render_state8_finalfit_world2_launcher.py \
  --source "$source_launcher" --output "$rendered_launcher")"
test "$observed_rendered_sha" = "$expected_rendered_sha"
test "$(sha256sum "$rendered_launcher" | cut -d' ' -f1)" = "$expected_rendered_sha"
bash -n "$rendered_launcher"
exec bash "$rendered_launcher"
