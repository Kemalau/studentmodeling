#!/usr/bin/env python3
"""Run exposure-matched C3 from the promoted round-1500 data order.

The C3 checkpoint was produced after applying a promoted full-epoch order.
Rebuilding the seed-42 order is therefore not an equivalent warm start: the
already-consumed prefix agrees, but the unconsumed suffix is intentionally
different.  This entrypoint preserves the checkpoint's hash-bound order and
allows the base trainer to replace only the manifest-declared C3 segment.

All checks are train-time provenance checks.  The decoder remains the same
strict state-plus-query model used by the open-T3 entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_recurrent_student_m4 as trainer
import train_recurrent_student_m4_open_t3 as open_t3


CONTRACT = "round1500_promoted_order_plus_controlled_segment_v1"
CONTROLLED_EXPERIMENT = "exposure_matched_auxiliary_supervision_v1"
SOURCE_TRAIN_SHA256 = "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd"
SOURCE_PROMOTION_MANIFEST_SHA256 = (
    "adc6eacca29cf22c554875d0f1e5cd8e6cbd6722966228b3aafe8a3e62ca0ee7"
)
SOURCE_ORDER_SHA256 = "a212f65b3efdee343f673e9cd2485c9ce66cc58dbd20c1f82cdebd6704226134"
SOURCE_PREFIX_SHA256 = "e5879cf158c5a000558af32afc81be883ae2a0c8174fb7db7843b8933cb735ff"
SOURCE_SUFFIX_SHA256 = "c95afc28183c84999986f9533617e5e278f91d24c0a3dec79b35306f13f0cb64"
SOURCE_WORLD_SIZE = 16
SOURCE_ROUND = 1500
SEGMENT_START = SOURCE_WORLD_SIZE * SOURCE_ROUND
SEGMENT_ROWS = 1000
EXPECTED_REAL_ROWS = 259_845


def cli_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} has no value")
    return sys.argv[index + 1]


def order_sha256(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("C3 index manifest must be a JSON object")
    if value.get("stage") != "layerwise_task_retention_index_manifest_v1":
        raise ValueError("C3 index-manifest stage drift")
    if value.get("controlled_experiment") != "C3_exposure_matched_auxiliary_supervision_v1":
        raise ValueError("C3 controlled-experiment manifest drift")
    if value.get("source_train_sha256") != SOURCE_TRAIN_SHA256:
        raise ValueError("C3 index-manifest source train drift")
    if int(value.get("start_global_position", -1)) != SEGMENT_START:
        raise ValueError("C3 index-manifest start drift")
    indices = value.get("indices")
    if not isinstance(indices, list) or len(indices) != SEGMENT_ROWS:
        raise ValueError("C3 index-manifest segment length drift")
    if any(type(index) is not int for index in indices):
        raise ValueError("C3 index manifest contains non-integer indices")
    return value


def validate_source_order(
    payload: dict[str, Any],
    *,
    real_count: int,
    expected_order_sha256: str = SOURCE_ORDER_SHA256,
    expected_promotion_sha256: str = SOURCE_PROMOTION_MANIFEST_SHA256,
) -> list[int]:
    config = payload.get("config") or {}
    order = list(payload.get("real_order") or [])
    if payload.get("distributed") is not True:
        raise ValueError("C3 source must be a distributed checkpoint")
    if int(payload.get("world_size", -1)) != SOURCE_WORLD_SIZE:
        raise ValueError("C3 source world-size drift")
    if int(payload.get("round", -1)) != SOURCE_ROUND:
        raise ValueError("C3 source round drift")
    if config.get("train_sha256") != SOURCE_TRAIN_SHA256:
        raise ValueError("C3 source train hash drift")
    if config.get("promoted_real_order_applied") is not True:
        raise ValueError("C3 source lacks promoted real-order provenance")
    if config.get("promoted_real_order_manifest_sha256") != expected_promotion_sha256:
        raise ValueError("C3 source promotion manifest drift")
    if len(order) != real_count or sorted(order) != list(range(real_count)):
        raise ValueError("C3 source real order is not a complete permutation")
    if order_sha256(order) != expected_order_sha256:
        raise ValueError("C3 source promoted real-order hash drift")
    return order


def validate_resume_order(
    payload: dict[str, Any],
    manifest: dict[str, Any],
    *,
    real_count: int,
    expected_prefix_sha256: str = SOURCE_PREFIX_SHA256,
    expected_suffix_sha256: str = SOURCE_SUFFIX_SHA256,
) -> list[int]:
    config = payload.get("config") or {}
    order = list(payload.get("real_order") or [])
    if payload.get("distributed") is not True:
        raise ValueError("C3 recovery checkpoint is not distributed")
    if len(order) != real_count:
        raise ValueError("C3 recovery real-order length drift")
    if config.get("train_sha256") != SOURCE_TRAIN_SHA256:
        raise ValueError("C3 recovery train hash drift")
    if config.get("c3_promoted_order_contract") != CONTRACT:
        raise ValueError("C3 recovery order contract drift")
    if config.get("c3_promoted_source_order_sha256") != SOURCE_ORDER_SHA256:
        raise ValueError("C3 recovery source-order provenance drift")
    stop = SEGMENT_START + SEGMENT_ROWS
    if order_sha256(order[:SEGMENT_START]) != expected_prefix_sha256:
        raise ValueError("C3 recovery changed the promoted prefix")
    if order_sha256(order[stop:]) != expected_suffix_sha256:
        raise ValueError("C3 recovery changed the promoted suffix")
    if order[SEGMENT_START:stop] != list(manifest["indices"]):
        raise ValueError("C3 recovery controlled segment drift")
    return order


def main() -> None:
    if os.environ.get("C3_CONTROLLED_EXPERIMENT") != CONTROLLED_EXPERIMENT:
        raise ValueError("C3 promoted-order entrypoint requires the frozen controlled experiment")
    manifest_path = cli_value("--diagnostic-index-manifest")
    train_path = cli_value("--train-jsonl")
    output_path = cli_value("--output-dir")
    warm_path = cli_value("--diagnostic-warm-start-checkpoint")
    resume_path = cli_value("--resume-from-checkpoint")
    if not manifest_path or not train_path or not output_path:
        raise ValueError("C3 promoted-order entrypoint is missing required paths")
    if bool(warm_path) == bool(resume_path):
        raise ValueError("C3 requires exactly one warm-start or recovery checkpoint")
    manifest_file = Path(manifest_path)
    manifest = read_manifest(manifest_file)
    expected_manifest_sha = os.environ.get("MANIFEST_SHA256", "")
    if trainer.file_sha256(manifest_file) != expected_manifest_sha:
        raise ValueError("C3 manifest environment hash drift")
    if trainer.file_sha256(Path(train_path)) != SOURCE_TRAIN_SHA256:
        raise ValueError("C3 materialized train hash drift")

    checkpoint_path = Path(warm_path or resume_path or "")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if warm_path:
        inherited_order = validate_source_order(
            payload, real_count=EXPECTED_REAL_ROWS
        )
        checkpoint_role = "promoted_round1500_source"
    else:
        inherited_order = validate_resume_order(
            payload, manifest, real_count=EXPECTED_REAL_ROWS
        )
        checkpoint_role = "c3_exact_recovery"
    # The payload includes optimizer state and can be several GiB.  Retain
    # only the audited order; the base trainer will load the checkpoint once
    # more when it restores the actual model/optimizer state.
    del payload

    original_schedule = trainer.build_epoch_schedule
    original_parse = argparse.ArgumentParser.parse_args

    def build_inherited_schedule(real_count: int, causal_count: int, seed: int):
        if real_count != EXPECTED_REAL_ROWS or seed != 42:
            raise ValueError("C3 inherited-order schedule dimensions drift")
        _, causal_schedule = original_schedule(real_count, causal_count, seed)
        return list(inherited_order), causal_schedule

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.c3_promoted_order_contract = CONTRACT
        namespace.c3_promoted_source_order_sha256 = SOURCE_ORDER_SHA256
        namespace.c3_promoted_source_manifest_sha256 = SOURCE_PROMOTION_MANIFEST_SHA256
        namespace.c3_controlled_segment_start = SEGMENT_START
        namespace.c3_controlled_segment_rows = SEGMENT_ROWS
        namespace.c3_controlled_manifest_sha256 = expected_manifest_sha
        # This value is science-bearing and must remain identical across an
        # exact service recovery.  The attempt-specific role is written only
        # to the terminal receipt below.
        namespace.c3_checkpoint_role = "promoted_order_continuation"
        return namespace

    trainer.build_epoch_schedule = build_inherited_schedule
    argparse.ArgumentParser.parse_args = parse
    try:
        open_t3.main()
    finally:
        trainer.build_epoch_schedule = original_schedule
        argparse.ArgumentParser.parse_args = original_parse

    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
        return
    receipt = {
        "stage": "c3_promoted_real_order_contract_v1",
        "status": "complete",
        "contract": CONTRACT,
        "checkpoint_role": checkpoint_role,
        "source_train_sha256": SOURCE_TRAIN_SHA256,
        "source_promotion_manifest_sha256": SOURCE_PROMOTION_MANIFEST_SHA256,
        "source_real_order_sha256": SOURCE_ORDER_SHA256,
        "source_prefix_sha256": SOURCE_PREFIX_SHA256,
        "source_suffix_sha256": SOURCE_SUFFIX_SHA256,
        "controlled_segment_start": SEGMENT_START,
        "controlled_segment_rows": SEGMENT_ROWS,
        "controlled_manifest_sha256": expected_manifest_sha,
    }
    output_dir = Path(output_path)
    (output_dir / "c3_promoted_real_order.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
