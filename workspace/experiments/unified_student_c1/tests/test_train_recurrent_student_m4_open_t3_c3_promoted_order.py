from __future__ import annotations

import copy

import pytest

from scripts.train_recurrent_student_m4_open_t3_c3_promoted_order import (
    CONTRACT,
    SEGMENT_ROWS,
    SEGMENT_START,
    SOURCE_ORDER_SHA256,
    order_sha256,
    validate_resume_order,
    validate_source_order,
)


def source_payload(order: list[int]) -> dict:
    return {
        "distributed": True,
        "world_size": 16,
        "round": 1500,
        "real_order": order,
        "config": {
            "train_sha256": "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd",
            "promoted_real_order_applied": True,
            "promoted_real_order_manifest_sha256": "promotion",
        },
    }


def test_source_order_requires_bound_promoted_permutation() -> None:
    order = list(range(12))
    payload = source_payload(order)
    assert validate_source_order(
        payload,
        real_count=12,
        expected_order_sha256=order_sha256(order),
        expected_promotion_sha256="promotion",
    ) == order
    payload["real_order"] = order[:-1] + [0]
    with pytest.raises(ValueError, match="complete permutation"):
        validate_source_order(
            payload,
            real_count=12,
            expected_order_sha256=order_sha256(order),
            expected_promotion_sha256="promotion",
        )


def test_resume_order_allows_only_manifest_segment_change(monkeypatch) -> None:
    import scripts.train_recurrent_student_m4_open_t3_c3_promoted_order as module

    start = 4
    rows = 2
    source = list(range(10))
    manifest = {"indices": [8, 9]}
    resumed = source[:start] + manifest["indices"] + source[start + rows :]
    payload = {
        "distributed": True,
        "real_order": resumed,
        "config": {
            "train_sha256": module.SOURCE_TRAIN_SHA256,
            "c3_promoted_order_contract": CONTRACT,
            "c3_promoted_source_order_sha256": SOURCE_ORDER_SHA256,
        },
    }
    monkeypatch.setattr(module, "SEGMENT_START", start)
    monkeypatch.setattr(module, "SEGMENT_ROWS", rows)
    assert validate_resume_order(
        payload,
        manifest,
        real_count=10,
        expected_prefix_sha256=order_sha256(source[:start]),
        expected_suffix_sha256=order_sha256(source[start + rows :]),
    ) == resumed

    broken = copy.deepcopy(payload)
    broken["real_order"][0] = 7
    with pytest.raises(ValueError, match="promoted prefix"):
        validate_resume_order(
            broken,
            manifest,
            real_count=10,
            expected_prefix_sha256=order_sha256(source[:start]),
            expected_suffix_sha256=order_sha256(source[start + rows :]),
        )


def test_production_segment_boundary_is_round1500() -> None:
    assert SEGMENT_START == 16 * 1500 == 24000
    assert SEGMENT_ROWS == 1000
