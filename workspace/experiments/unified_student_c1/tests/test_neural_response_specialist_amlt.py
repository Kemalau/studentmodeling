from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_neural_response_specialist_amlt import validate  # noqa: E402


def test_neural_response_specialist_config_is_hash_bound_and_compliant() -> None:
    validate(ROOT / "amlt/feeds_neural_response_specialists_20260814.yaml", ROOT)
