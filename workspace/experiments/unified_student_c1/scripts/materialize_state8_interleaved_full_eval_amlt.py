#!/usr/bin/env python3
"""Fill the generated Feeds template with the frozen 20-job full evaluation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
RUN_DIR = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_"
    "eedi_same_question_v1_early_label_high_seed8401"
)
PROGRESS = "state8_interleaved_seed8401_full_eval_20260814"
FORMAL_SHA = "7343ddffe618c341765353701c9472ea316a716d2037cf329c5d7f3f270f9b7e"
C4_SHA = "94da898c4f5813bbb3166f05e2eec96258bacaea80768eaee3519829b438e9dd"
PAIR_DIGESTS = {
    13: "9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693",
    42: "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8",
    97: "15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118",
}


def base_job(name: str, command: str, env: dict[str, str]) -> dict:
    return {
        "name": name,
        "sku": "80G1-A100",
        "process_count_per_node": 1,
        "mpi": False,
        "sla_tier": "Standard",
        "command": command,
        "submit_args": {
            "max_attempts": 2,
            "max_run_duration_seconds": 259200,
            "env": {
                "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
                "VENV_ROOT": "/tmp/studentmodeling-state8-interleaved-full-eval-venv",
                "STATE8_EVAL_RUN_DIR": RUN_DIR,
                **env,
            },
        },
    }


def materialize(template: Path, output: Path) -> None:
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["description"] = (
        "Frozen state8 interleaved-retention checkpoint: full 21,676-row T1-T5 "
        "formal evaluation in 16 disjoint shards plus three MalRule C4 seeds and "
        "one frozen identical-query Eedi C4 panel; evaluation-only, no promotion claim"
    )
    config["target"] = {
        "service": "manifold",
        "name": "Feeds",
        "workspace_name": WORKSPACE,
    }
    config["environment"] = {
        "image": IMAGE,
        "setup": [
            "python -m venv --system-site-packages /tmp/studentmodeling-state8-interleaved-full-eval-venv",
            "/tmp/studentmodeling-state8-interleaved-full-eval-venv/bin/python -m pip install --upgrade pip",
            "/tmp/studentmodeling-state8-interleaved-full-eval-venv/bin/python -m pip install -r requirements-feeds.txt",
        ],
    }
    config["code"] = {
        "local_dir": "$CONFIG_DIR/..",
        "ignore": [".git/", ".vscode/", "amlt/", "data/", "results/", "__pycache__/", "*.pyc"],
    }
    config["storage"] = {
        "student_store": {
            "storage_account_name": "csnewsandfeeds4150361735",
            "container_name": "amulet",
            "mount_dir": "/mnt/amulet",
            "is_output": True,
        }
    }
    common_formal = (
        "set -euo pipefail\n"
        f"test \"$$(sha256sum scripts/run_state8_interleaved_full_formal_shard.sh | cut -d' ' -f1)\" = {FORMAL_SHA}\n"
        "bash scripts/run_state8_interleaved_full_formal_shard.sh\n"
    )
    jobs = [
        base_job(
            f"state8-interleaved-full-formal-shard{index:02d}",
            common_formal,
            {"NUM_SHARDS": "16", "SHARD_INDEX": str(index)},
        )
        for index in range(16)
    ]
    common_c4 = (
        "set -euo pipefail\n"
        f"test \"$$(sha256sum scripts/run_state8_interleaved_full_c4.sh | cut -d' ' -f1)\" = {C4_SHA}\n"
        "bash scripts/run_state8_interleaved_full_c4.sh\n"
    )
    for seed, digest in PAIR_DIGESTS.items():
        jobs.append(
            base_job(
                f"state8-interleaved-full-malrule-c4-seed{seed}",
                common_c4,
                {
                    "C4_KIND": "malrule",
                    "PAIR_SEED": str(seed),
                    "PAIR_MANIFEST_SHA256": digest,
                    "STATE8_EVAL_PROGRESS_NAMESPACE": PROGRESS,
                },
            )
        )
    jobs.append(
        base_job(
            "state8-interleaved-full-eedi-c4-final150",
            common_c4,
            {"C4_KIND": "eedi", "STATE8_EVAL_PROGRESS_NAMESPACE": PROGRESS},
        )
    )
    config["jobs"] = jobs
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
