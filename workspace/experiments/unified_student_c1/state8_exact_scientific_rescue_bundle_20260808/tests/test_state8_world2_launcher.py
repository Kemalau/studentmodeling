from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from scripts.render_state8_world2_launcher import SOURCE_SHA256, render


ROOT = Path(__file__).resolve().parents[1]


def test_world2_renderer_preserves_frozen_world4_source(tmp_path: Path) -> None:
    source = ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh"
    before = source.read_bytes()
    assert hashlib.sha256(before).hexdigest() == SOURCE_SHA256
    output = tmp_path / "world2.sh"
    rendered_sha = render(source, output)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256(output.read_bytes()).hexdigest() == rendered_sha
    subprocess.run(["bash", "-n", str(output)], check=True)
    text = output.read_text()
    assert "export TOTAL_ROUNDS=2500" in text
    assert 'test "$WORLD_SIZE" = 2' in text
    assert "--expected-world-size 2" in text
    assert "--expected-planned-optimizer-steps 1250" in text
    assert "TOPOLOGY_ARGS=(--distributed-world-size 2)" in text
    assert "GRADIENT_ACCUMULATION_STEPS=2" in text
    assert "cfg['distributed_world_size']*cfg['gradient_accumulation_steps']==4" in text
