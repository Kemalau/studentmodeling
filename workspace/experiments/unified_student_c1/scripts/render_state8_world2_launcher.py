#!/usr/bin/env python3
"""Render the frozen State8 curriculum launcher for world2/global-batch4."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "b545496b52c8aff82196c8ac917178b110e304b1ae4880ed24ca039cff626638"


def replace_exact(text: str, old: str, new: str, count: int) -> str:
    observed = text.count(old)
    if observed != count:
        raise ValueError(f"replacement drift: expected={count} observed={observed} old={old!r}")
    return text.replace(old, new)


def render(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise ValueError(f"frozen launcher hash drift: {observed}")
    text = raw.decode("utf-8")
    text = replace_exact(text, "export TOTAL_ROUNDS=1250", "export TOTAL_ROUNDS=2500", 1)
    text = replace_exact(
        text,
        'RESUME_CADENCE_ARGS=(--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 5)',
        'RESUME_CADENCE_ARGS=(--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 10)',
        1,
    )
    text = replace_exact(text, 'test "$WORLD_SIZE" = 4', 'test "$WORLD_SIZE" = 2', 1)
    text = replace_exact(
        text,
        '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"',
        "--expected-planned-optimizer-steps 1250",
        2,
    )
    text = replace_exact(
        text,
        '--planned-optimizer-steps "$TOTAL_ROUNDS" --gradient-accumulation-steps 4)',
        "--planned-optimizer-steps 1250 --gradient-accumulation-steps 4)",
        1,
    )
    text = replace_exact(text, "--expected-world-size 4", "--expected-world-size 2", 1)
    text = replace_exact(
        text,
        "assert int(payload.get('world_size',-1))==4,payload",
        "assert int(payload.get('world_size',-1))==2,payload",
        1,
    )
    text = replace_exact(
        text,
        "TOPOLOGY_ARGS=(--distributed-world-size 4)",
        "TOPOLOGY_ARGS=(--distributed-world-size 2)",
        2,
    )
    text = replace_exact(
        text,
        "else\n  EXECUTION_ARGS=(--distributed-mpi)\n  TOPOLOGY_ARGS=(--distributed-world-size 2)\n  GRADIENT_ACCUMULATION_STEPS=1",
        "else\n  EXECUTION_ARGS=(--distributed-mpi)\n  TOPOLOGY_ARGS=(--distributed-world-size 2)\n  GRADIENT_ACCUMULATION_STEPS=2",
        1,
    )
    text = replace_exact(
        text,
        "assert cfg['state_token_count']==8 and cfg['state_architecture']=='layerwise',cfg",
        "assert cfg['state_token_count']==8 and cfg['state_architecture']=='layerwise',cfg\n"
        "assert cfg['planned_optimizer_steps']==1250,cfg\n"
        "assert cfg['distributed_world_size']==2 and cfg['gradient_accumulation_steps']==2,cfg\n"
        "assert cfg['distributed_world_size']*cfg['gradient_accumulation_steps']==4,cfg",
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
