from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_legacy_layerwise_state_count_compat import materialize


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_state_count_compat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialized_wrapper_accepts_only_legacy_single_slot(tmp_path: Path) -> None:
    output = tmp_path / "layerwise_recurrent_student_model.py"
    receipt = materialize(
        ROOT / "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
        output,
    )
    assert receipt["accepted_state_token_count"] == 1
    assert receipt["capacity_change_allowed"] is False
    cls = load_module(output).LayerwiseRecurrentStudentModel
    signature = inspect.signature(cls.__init__)
    assert signature.parameters["state_token_count"].default == 1

    # The rejection happens before the wrapper inspects the backbone, making
    # this a cheap fail-closed capacity test.
    with pytest.raises(ValueError, match="requires state_token_count=1"):
        cls(object(), state_token_id=7, state_token_count=8)
