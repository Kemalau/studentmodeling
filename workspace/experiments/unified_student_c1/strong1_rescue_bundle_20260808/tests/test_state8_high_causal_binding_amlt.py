from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_high_causal_binding_amlt import validate  # noqa: E402


class State8HighCausalBindingAmltTest(unittest.TestCase):
    def test_checked_in_config_is_valid(self) -> None:
        import yaml

        value = yaml.safe_load(
            (ROOT / "amlt/feeds_state8_high_causal_binding_20260806.yaml").read_text()
        )
        validate(value)


if __name__ == "__main__":
    unittest.main()
