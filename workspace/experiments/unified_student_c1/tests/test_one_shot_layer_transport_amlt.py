from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_one_shot_layer_transport_c4 import SPECS, load, render  # noqa: E402
from validate_one_shot_layer_transport_amlt import validate  # noqa: E402


class LayerTransportConfigTest(unittest.TestCase):
    def test_full_and_recovery_subset_validate(self):
        value = render(load(ROOT / "amlt/feeds_state_curriculum_stage_a_alias_gate_20260727.yaml"))
        validate(value)
        selected = SPECS["blocks"]["job"]
        value["jobs"] = [job for job in value["jobs"] if job["name"] == selected]
        validate(value, {selected})


if __name__ == "__main__":
    unittest.main()
