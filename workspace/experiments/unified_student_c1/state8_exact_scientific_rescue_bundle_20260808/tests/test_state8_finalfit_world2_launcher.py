from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from scripts.render_state8_finalfit_world2_launcher import SOURCE_SHA256, render


ROOT = Path(__file__).resolve().parents[1]


def test_world2_renderer_preserves_global_batch_and_optimizer_steps(tmp_path: Path) -> None:
    source = ROOT / "scripts/run_state8_unified_final_fit.sh"
    before = source.read_bytes()
    assert hashlib.sha256(before).hexdigest() == SOURCE_SHA256
    output = tmp_path / "world2.sh"
    rendered_sha = render(source, output)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256(output.read_bytes()).hexdigest() == rendered_sha
    subprocess.run(["bash", "-n", str(output)], check=True)
    text = output.read_text()
    assert "export TOTAL_ROUNDS=129923" in text
    assert "export TOTAL_OPTIMIZER_STEPS=16241" in text
    assert 'test "$WORLD_SIZE" = 2' in text
    assert "--expected-world-size 2" in text
    assert '--expected-planned-optimizer-steps "$TOTAL_OPTIMIZER_STEPS"' in text
    assert "--gradient-accumulation-steps 8" in text
    assert "--distributed-world-size 2" in text
    assert "config['distributed_world_size']*config['gradient_accumulation_steps']==16" in text
    assert "completion['optimizer_steps']==16241" in text


def test_world2_renderer_does_not_claim_bitwise_topology_equivalence() -> None:
    source = (ROOT / "scripts/render_state8_finalfit_world2_launcher.py").read_text()
    assert "not a claim of bitwise" in source
