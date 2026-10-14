import copy

import pytest

from scripts.audit_state8_curriculum_checkpoint import (
    LEGACY_GENERIC_ROUTE,
    raw_history_route_proof,
)


def test_raw_history_route_requires_explicit_false_by_default() -> None:
    assert (
        raw_history_route_proof({"raw_history_visible_to_decoder": False}, "current_query")
        == "explicit_false"
    )
    with pytest.raises(ValueError, match="raw history is visible"):
        raw_history_route_proof({"raw_history_visible_to_decoder": True}, "generic")
    with pytest.raises(ValueError, match="route is absent"):
        raw_history_route_proof({}, "current_query")


def test_legacy_generic_route_is_bound_to_exact_executable_recipe() -> None:
    assert (
        raw_history_route_proof(dict(LEGACY_GENERIC_ROUTE), "generic")
        == "legacy_exact_hashed_state_only_route"
    )
    for key in LEGACY_GENERIC_ROUTE:
        drifted = copy.deepcopy(LEGACY_GENERIC_ROUTE)
        drifted[key] = None
        with pytest.raises(ValueError, match="legacy route proof failed"):
            raw_history_route_proof(drifted, "generic")
