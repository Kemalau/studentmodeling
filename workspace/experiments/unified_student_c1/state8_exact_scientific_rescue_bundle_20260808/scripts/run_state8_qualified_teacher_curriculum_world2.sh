#!/usr/bin/env bash
set -euo pipefail

source_launcher=scripts/run_state8_qualified_teacher_curriculum.sh
rendered_launcher=/tmp/run_state8_qualified_teacher_curriculum_world2.sh
expected_source_sha=b545496b52c8aff82196c8ac917178b110e304b1ae4880ed24ca039cff626638
expected_rendered_sha=c8149a6555bbbc56eaa93b3d3324146e30abd97d8586cc199a85d81d07f722c1

test "$(sha256sum "$source_launcher" | cut -d' ' -f1)" = "$expected_source_sha"
observed_rendered_sha="$(python scripts/render_state8_world2_launcher.py \
  --source "$source_launcher" --output "$rendered_launcher")"
test "$observed_rendered_sha" = "$expected_rendered_sha"
test "$(sha256sum "$rendered_launcher" | cut -d' ' -f1)" = "$expected_rendered_sha"
bash -n "$rendered_launcher"
exec bash "$rendered_launcher"
