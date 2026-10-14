#!/usr/bin/env python3
"""Render the frozen State8 final-fit launcher for world2/global-batch16.

Two ranks with gradient accumulation eight preserve the exact row order,
global batch of sixteen, and 16,241 optimizer updates.  LoRA dropout is
nonzero, so this is a schedulability fallback, not a claim of bitwise
equivalence to the world4 or world16 topology.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "434df629e5a05746fe7c58547a7fc9e8659dc18462d71a66858a5405bf18f1dc"


def replace_exact(text: str, old: str, new: str, count: int) -> str:
    observed = text.count(old)
    if observed != count:
        raise ValueError(
            f"replacement drift: expected={count} observed={observed} old={old!r}"
        )
    return text.replace(old, new)


def render(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise ValueError(f"frozen launcher hash drift: {observed}")
    text = raw.decode("utf-8")
    text = replace_exact(
        text,
        "export TOTAL_ROUNDS=16241\nexport MILESTONE_ROUNDS=\"3249 6497 9745 12993\"",
        "export TOTAL_ROUNDS=129923\n"
        "export TOTAL_OPTIMIZER_STEPS=16241\n"
        "export MILESTONE_ROUNDS=\"25992 51976 77960 103944\"",
        1,
    )
    text = replace_exact(text, 'test "$WORLD_SIZE" = 16', 'test "$WORLD_SIZE" = 2', 1)
    text = replace_exact(text, "--expected-world-size 16", "--expected-world-size 2", 1)
    text = replace_exact(
        text,
        '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"',
        '--expected-planned-optimizer-steps "$TOTAL_OPTIMIZER_STEPS"',
        1,
    )
    text = replace_exact(
        text,
        "--save-steps 50 --recovery-save-steps 25)",
        "--save-steps 400 --recovery-save-steps 200)",
        1,
    )
    text = replace_exact(
        text,
        "--normalize-causal-exposure --gradient-accumulation-steps 1",
        "--normalize-causal-exposure --gradient-accumulation-steps 8",
        1,
    )
    text = replace_exact(
        text,
        "--save-steps 50 --checkpoint-retention 2",
        "--save-steps 400 --checkpoint-retention 2",
        1,
    )
    text = replace_exact(
        text,
        "--distributed-mpi --distributed-world-size 16 --seed 42",
        "--distributed-mpi --distributed-world-size 2 --seed 42",
        1,
    )
    text = replace_exact(
        text,
        "assert config['distributed_world_size']==16 and config['distributed_rounds']==16241,config",
        "assert config['distributed_world_size']==2 and config['distributed_rounds']==129923,config\n"
        "assert config['gradient_accumulation_steps']==8,config\n"
        "assert config['planned_optimizer_steps']==16241,config\n"
        "assert config['distributed_world_size']*config['gradient_accumulation_steps']==16,config\n"
        "assert completion['optimizer_steps']==16241,completion",
        1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.source, args.output))


if __name__ == "__main__":
    main()
