import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from t3_llm_judge import DEFAULT_TRAPI_DEPLOYMENT, config_from_args


DEPLOYMENT = "gpt-4o_2024-11-20"


def test_default_trapi_deployment_is_versioned_and_resolvable() -> None:
    assert DEFAULT_TRAPI_DEPLOYMENT == DEPLOYMENT
    previous = os.environ.pop("TRAPI_DEPLOYMENT", None)
    try:
        config = config_from_args("responses-model", "unused", 1, backend="trapi")
    finally:
        if previous is not None:
            os.environ["TRAPI_DEPLOYMENT"] = previous
    assert config.model == DEPLOYMENT


def test_explicit_trapi_deployment_wins_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("TRAPI_DEPLOYMENT", "stale-deployment")
    config = config_from_args(
        "responses-model", "unused", 1, backend="trapi", trapi_deployment=DEPLOYMENT
    )
    assert config.model == DEPLOYMENT
