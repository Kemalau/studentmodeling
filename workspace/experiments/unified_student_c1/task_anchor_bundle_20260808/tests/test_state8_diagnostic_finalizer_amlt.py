from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_state8_diagnostic_finalizer_amlt import load,validate  # noqa: E402


def test_config_passes_and_is_optimizer_free() -> None:
    validate(load(ROOT/'amlt/feeds_state8_diagnostic_finalizer_20260805.yaml'))
    source=(ROOT/'scripts/run_state8_diagnostic_finalizer.sh').read_text()
    assert '--validation-only-from-distributed-checkpoint' in source
    assert 'optimizer.step()' not in source
