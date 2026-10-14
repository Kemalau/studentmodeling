#!/usr/bin/env bash
set -euo pipefail

source_launcher=scripts/run_state8_unified_final_fit.sh
rendered_launcher=/tmp/run_state8_unified_final_fit_world4.sh
expected_source_sha=434df629e5a05746fe7c58547a7fc9e8659dc18462d71a66858a5405bf18f1dc
expected_rendered_sha=f29632edec86ba7ce516548d2ab04874b7aebe8ac47d0b49c483026d30e1284b

test "$(sha256sum "$source_launcher" | cut -d' ' -f1)" = "$expected_source_sha"
observed_rendered_sha="$(python scripts/render_state8_finalfit_world4_launcher.py \
  --source "$source_launcher" --output "$rendered_launcher")"
test "$observed_rendered_sha" = "$expected_rendered_sha"
test "$(sha256sum "$rendered_launcher" | cut -d' ' -f1)" = "$expected_rendered_sha"
bash -n "$rendered_launcher"
exec bash "$rendered_launcher"
