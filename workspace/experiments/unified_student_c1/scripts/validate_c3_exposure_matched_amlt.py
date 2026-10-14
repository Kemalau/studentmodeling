#!/usr/bin/env python3
"""Fail-closed validation for the exposure-matched C3 Feeds jobs."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ARMS = {
    "c3-matched-malrule-correctness-seed20260804": (
        "malrule_correctness",
        "manifests/c3_exposure_matched_20260804/c3_exposure_matched_malrule_correctness_seed20260804_v1.json",
        "ce358ed02a74fe0d5b9bdf75073e9202eff0fe689a5620fb10f8b02e0e632d25",
    ),
    "c3-matched-malrule-response-seed20260804": (
        "malrule_response",
        "manifests/c3_exposure_matched_20260804/c3_exposure_matched_malrule_response_seed20260804_v1.json",
        "e6202050ea4dbc7b00298546abb7d6e964565a807fbfda8c0165f9ca4c8dd6ea",
    ),
    "c3-matched-malrule-misconception-seed20260804": (
        "malrule_misconception",
        "manifests/c3_exposure_matched_20260804/c3_exposure_matched_malrule_misconception_seed20260804_v1.json",
        "ba74efc1eb675cab148af062fde1c6bad0c9b57fbbc1c579b68756ef340da818",
    ),
}
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-c3-matched-venv",
    "BRANCH": "eedi_full",
    "LOSS_PROFILE": "state_path",
    "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py",
    "C3_PROMOTED_ENTRYPOINT_SHA256": "c8f87bfc6cb7f9153be05c67068f519a968ef03b448cee9a2bee8b3d97b87002",
    "CAUSAL_VARIANT": "base",
    "SAME_RULE_AUGMENTER_SHA256": "1ae82541e97353d23b674af80dfb0e2d50cfd92e3a6ee99c2b55f3b0caf1d785",
    "SEGMENT_ROWS": "1000",
    "LEARNING_RATE": "5e-7",
    "EXPECTED_CAUSAL_PAIRS": "50",
    "DIAGNOSTIC_START_GLOBAL_POSITION": "24000",
    "WARM_START_CHECKPOINT_DIR": "/mnt/amulet/studentmodeling/checkpoints/layerwise_unified_world_model_seed42_taskretention_eedi_full_open_t3_state_path_v1_20260802/checkpoint-round-1500",
    "WARM_START_EXPECTED_ROUND": "1500",
    "WARM_START_EXPECTED_WORLD_SIZE": "16",
    "C3_CONTROLLED_EXPERIMENT": "exposure_matched_auxiliary_supervision_v1",
}
TOPOLOGIES = {
    "world4": {
        "sku": "80G4-A100",
        "world_size": "4",
        "gradient_accumulation_steps": "1",
        "stop_round": "6250",
    },
    "world2": {
        "sku": "80G2-A100",
        "world_size": "2",
        "gradient_accumulation_steps": "2",
        "stop_round": "12500",
    },
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    return value


def normalize_env(env: dict[str, object]) -> dict[str, str]:
    """Canonicalize AMLT's harmless scientific-notation YAML rewrite.

    ``amlt run --dump`` rewrites the source scalar ``5e-7`` as
    ``5.0e-07``.  They are the same learning rate, while every other field is
    provenance-sensitive and remains an exact string comparison.
    """
    normalized = {str(key): str(value) for key, value in env.items()}
    if "LEARNING_RATE" in normalized:
        normalized["LEARNING_RATE"] = format(
            float(normalized["LEARNING_RATE"]), ".12g"
        )
    return normalized


def validate(
    config: dict,
    *,
    selected_jobs: set[str] | None = None,
    expected_topology: str | None = None,
) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|<describe your experiment>", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    expected = set(ARMS) if selected_jobs is None else set(selected_jobs)
    if not expected or not expected <= set(ARMS):
        raise ValueError("Unknown selected C3 job")
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("C3 job-set drift")
    normalized = {}
    commands = set()
    observed_topologies: set[str] = set()
    for job in jobs:
        name = str(job["name"])
        env = normalize_env((job.get("submit_args") or {}).get("env", {}))
        matches = [
            topology
            for topology, spec in TOPOLOGIES.items()
            if job.get("sku") == spec["sku"]
            and env.get("C3_WORLD_SIZE") == spec["world_size"]
            and env.get("C3_GRADIENT_ACCUMULATION_STEPS")
            == spec["gradient_accumulation_steps"]
            and env.get("DIAGNOSTIC_STOP_AFTER_ROUND") == spec["stop_round"]
        ]
        if len(matches) != 1:
            raise ValueError(f"C3 topology/environment drift: {name}")
        observed_topologies.add(matches[0])
        if (
            job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"C3 GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"C3 retry/duration drift: {name}")
        arm, manifest, digest = ARMS[name]
        expected_env = normalize_env({
            **COMMON,
            "DIAGNOSTIC_STOP_AFTER_ROUND": TOPOLOGIES[matches[0]]["stop_round"],
            "C3_WORLD_SIZE": TOPOLOGIES[matches[0]]["world_size"],
            "C3_GRADIENT_ACCUMULATION_STEPS": TOPOLOGIES[matches[0]]["gradient_accumulation_steps"],
            "C3_ARM": arm,
            "MANIFEST": manifest,
            "MANIFEST_SHA256": digest,
        })
        if env != expected_env:
            raise ValueError(f"C3 controlled environment drift: {name}")
        common = copy.deepcopy(env)
        for key in ("C3_ARM", "MANIFEST", "MANIFEST_SHA256"):
            common.pop(key)
        normalized[name] = common
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.add(command)
    if len(commands) != 1 or len({tuple(sorted(row.items())) for row in normalized.values()}) != 1:
        raise ValueError("C3 arms differ beyond auxiliary supervision")
    if len(observed_topologies) != 1:
        raise ValueError("C3 arms use mixed topologies")
    topology = next(iter(observed_topologies))
    if expected_topology is not None and topology != expected_topology:
        raise ValueError(f"C3 expected topology={expected_topology}, observed={topology}")
    command = next(iter(commands)).replace("$$", "$")
    for fragment in (
        "93ff8cd9bac2a06d84ff5f1d6089ed4161fd2e4b387b18b259a83a9e6d9501ac",
        "30741b4183315db7f57ec0cfb26bd8a977c6d7d6b47c8b079cf682e8a0150539",
        "c8f87bfc6cb7f9153be05c67068f519a968ef03b448cee9a2bee8b3d97b87002",
        "FAILED line=%s rc=%s command=%s",
        'sha256sum "$MANIFEST"',
        "run_c3_exposure_matched_promoted_order.sh",
    ):
        if fragment not in command:
            raise ValueError(f"C3 provenance command drift: {fragment}")
    runner = (Path(__file__).resolve().parent / "run_layerwise_task_retention_rehearsal.sh")
    runner_text = runner.read_text(encoding="utf-8")
    if "materialize_legacy_layerwise_state_count_compat.py" not in runner_text:
        raise ValueError("C3 runner lacks frozen one-slot trainer compatibility")
    if "capacity_change_allowed" not in (
        Path(__file__).resolve().parent
        / "materialize_legacy_layerwise_state_count_compat.py"
    ).read_text(encoding="utf-8"):
        raise ValueError("C3 one-slot compatibility no longer fails closed")
    if "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5" not in runner_text:
        raise ValueError("C3 runner no longer binds the audited trainer")
    if "69fd4f2f4943a78a0afe9a701533d9ab034f247f87c7a09327f7523e310b7a52" not in runner_text:
        raise ValueError("C3 runner no longer binds the tested causal evaluator")
    promoted_entrypoint = (
        Path(__file__).resolve().parent
        / "train_recurrent_student_m4_open_t3_c3_promoted_order.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "round1500_promoted_order_plus_controlled_segment_v1",
        "SOURCE_ORDER_SHA256",
        "SOURCE_PREFIX_SHA256",
        "SOURCE_SUFFIX_SHA256",
        "validate_source_order",
        "validate_resume_order",
    ):
        if fragment not in promoted_entrypoint:
            raise ValueError(f"C3 promoted-order contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", choices=tuple(ARMS))
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES))
    args = parser.parse_args()
    validate(
        load(args.config),
        selected_jobs=set(args.jobs) if args.jobs else None,
        expected_topology=args.topology,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
