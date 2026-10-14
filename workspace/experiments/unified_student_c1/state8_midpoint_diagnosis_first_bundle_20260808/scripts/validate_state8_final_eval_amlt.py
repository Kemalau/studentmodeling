#!/usr/bin/env python3
"""Fail-closed AMLT validator for the complete state8 final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
FORMAL = {f"state8-final-formal-shard{index:02d}": index for index in range(16)}
C4 = {
    "state8-final-c4-seed13": (13, "9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693"),
    "state8-final-c4-seed42": (42, "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8"),
    "state8-final-c4-seed97": (97, "15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118"),
}
FILES = {
    "run_state8_final_formal_shard.sh": "bb4dcdbd949f142854198516176a61602826c0bb8af91708326694f6f2a4958e",
    "run_state8_final_c4.sh": "c02b4d74e941d0296718cf34db965e213033667cfaddb71ed69075bea3eb0982",
    "merge_state8_final_eval.py": "fb4c3d014a122cc8f077d46c0b7acdcc1e3637991388b96e402cf1e0c19495c7",
    "validate_state8_final_eval.py": "6666705a42dd6600ff3c949c9caad694d27aa40e9c644cdad050ee44e9fbb491",
    "evaluate_recurrent_formal_protocols.py": "262370592e018641b99d6282858d8990b38314c05bb6e3bbc645823834cf73ed",
    "evaluate_recurrent_causal_gate.py": "e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf",
    "evaluate_recurrent_diagnosis_c4.py": "e7421a850c2327d8df1436073742e71f796bec41c87f68403b7b00b60fac4664",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    full = set(FORMAL) | set(C4)
    expected = full if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(full) or names != expected:
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G1-A100"
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("mpi") is not False
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("VENV_ROOT") != "/tmp/studentmodeling-state8-final-eval-venv":
            raise ValueError(f"Venv drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 259200:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        for fragment in ("set -euo pipefail", "FAILED line=%s rc=%s command=%s"):
            if fragment not in command:
                raise ValueError(f"Launcher drift {name}: {fragment}")
        if name in FORMAL:
            if str(env.get("NUM_SHARDS")) != "16" or str(env.get("SHARD_INDEX")) != str(FORMAL[name]):
                raise ValueError(f"Formal shard drift: {name}")
            if "bash scripts/run_state8_final_formal_shard.sh" not in command:
                raise ValueError(f"Formal launcher drift: {name}")
        else:
            seed, digest = C4[name]
            if str(env.get("PAIR_SEED")) != str(seed) or env.get("PAIR_MANIFEST_SHA256") != digest:
                raise ValueError(f"C4 coordinate drift: {name}")
            if "bash scripts/run_state8_final_c4.sh" not in command:
                raise ValueError(f"C4 launcher drift: {name}")
    root = Path(__file__).resolve().parent
    for name, expected_hash in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected_hash:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    formal_launcher = (root / "run_state8_final_formal_shard.sh").read_text()
    c4_launcher = (root / "run_state8_final_c4.sh").read_text()
    for text in (formal_launcher, c4_launcher):
        for fragment in (
            "state8_qualified_unified_full259845_seed42_v1_20260805",
            "state8_final_fit_receipt.json",
            "state8_qualified_unified_final_fit_v1",
            "canonical_publish_receipt.json",
            "state8_canonical_final_checkpoint_publish_v1",
            "teacher_adapter_sha256",
            "teacher_gate_required",
            "history_training_mode",
            "strict_state_only",
            "history_dropout_counts",
            "online_kd_weight",
            "label_auxiliary_weight",
            "[64,8,5120]",
            "259845",
            "taxonomy_curriculum_t3heavy",
            "taxonomy_curriculum_balanced",
            "taxonomy_curriculum_t3heavy_diag",
            "taxonomy_curriculum_balanced_diag",
        ):
            if fragment not in text:
                raise ValueError(f"Final checkpoint binding drift: {fragment}")
    for fragment in (
        "--shard-index \"$SHARD_INDEX\" --num-shards \"$NUM_SHARDS\"",
        "--save-unredacted-predictions --t3-natural-schema",
        "unsharded_filtered_rows']==21676",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
    ):
        if fragment not in formal_launcher:
            raise ValueError(f"Formal protocol drift: {fragment}")
    for fragment in (
        "--num-pairs 150",
        "--progress-dir",
        "--progress-key",
        "evaluate_recurrent_diagnosis_c4.py",
        "--expected-adapter-sha256",
        "--expected-recurrent-state-sha256",
        "diagnosis_c4_seed",
        "diagnosis_natural_schema",
        "canonical_label_auxiliary_supervision",
    ):
        if fragment not in c4_launcher:
            raise ValueError(f"C4 protocol drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
