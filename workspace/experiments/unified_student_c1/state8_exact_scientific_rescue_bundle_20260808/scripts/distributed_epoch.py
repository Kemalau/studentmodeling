"""Small fail-closed helpers for exact distributed one-epoch schedules."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def context_from_environment(enabled: bool) -> DistributedContext:
    if not enabled:
        return DistributedContext(False, 0, 0, 1)
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "-1")))
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "-1"))
    )
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "-1"))
    )
    if rank < 0 or local_rank < 0 or world_size <= 1 or rank >= world_size:
        raise ValueError(
            f"Invalid distributed environment: rank={rank} local_rank={local_rank} "
            f"world_size={world_size}"
        )
    return DistributedContext(True, rank, local_rank, world_size)


def initialize(
    context: DistributedContext,
    master_port: int,
    *,
    timeout_seconds: int = 7200,
) -> None:
    if not context.enabled:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed recurrent training requires CUDA")
    if not 1024 <= int(master_port) <= 65535:
        raise ValueError(f"Invalid master port: {master_port}")
    if int(timeout_seconds) <= 0:
        raise ValueError("Distributed timeout must be positive")
    os.environ.setdefault("RANK", str(context.rank))
    os.environ.setdefault("LOCAL_RANK", str(context.local_rank))
    os.environ.setdefault("WORLD_SIZE", str(context.world_size))
    os.environ.setdefault("MASTER_ADDR", master_address(context))
    os.environ.setdefault("MASTER_PORT", str(master_port))
    torch.cuda.set_device(context.local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=context.rank,
        world_size=context.world_size,
        device_id=torch.device("cuda", context.local_rank),
        timeout=timedelta(seconds=int(timeout_seconds)),
    )


def master_address(context: DistributedContext) -> str:
    """Resolve the rank-zero rendezvous address from the Feeds MPI hostfile."""

    explicit = os.environ.get("MASTER_ADDR")
    if explicit:
        return explicit
    if not context.enabled:
        return "127.0.0.1"
    node_count = int(os.environ.get("OMPI_MCA_orte_num_nodes", "1"))
    if node_count <= 1:
        return "127.0.0.1"
    hostfile_value = os.environ.get("SINGULARITY_MPI_HOSTFILE_PATH")
    if not hostfile_value:
        raise ValueError("Multi-node distributed environment has no MPI hostfile")
    hostfile = Path(hostfile_value)
    if not hostfile.is_file():
        raise ValueError(f"MPI hostfile is missing: {hostfile}")
    first = next(
        (line.strip() for line in hostfile.read_text().splitlines() if line.strip()),
        "",
    )
    if not first:
        raise ValueError("MPI hostfile is empty")
    token = first.split()[0]
    # Feeds writes either ``host slots=N`` or MPICH-style ``host:N``.
    hostname = token.split(":", 1)[0]
    if not hostname:
        raise ValueError(f"Invalid MPI hostfile first entry: {first!r}")
    return hostname


def rounds_for_examples(example_count: int, world_size: int) -> int:
    if example_count <= 0 or world_size <= 0:
        raise ValueError("example_count and world_size must be positive")
    return math.ceil(example_count / world_size)


def global_position(round_index: int, rank: int, world_size: int) -> int:
    if round_index < 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed epoch coordinate")
    return round_index * world_size + rank


def positions_for_rank(example_count: int, rank: int, world_size: int) -> list[int]:
    rounds = rounds_for_examples(example_count, world_size)
    return [
        position
        for round_index in range(rounds)
        if (position := global_position(round_index, rank, world_size)) < example_count
    ]


def synchronize_gradients(
    parameters: Iterable[torch.nn.Parameter],
    context: DistributedContext,
    active_ranks: int | None = None,
    normalization_count: int | None = None,
    bucket_bytes: int = 64 * 1024 * 1024,
) -> None:
    """Average gradients in bounded flat buckets, including padded-rank zeros."""
    if not context.enabled:
        return
    if active_ranks is not None and normalization_count is not None:
        raise ValueError("Specify only one gradient normalization count")
    divisor = (
        int(normalization_count)
        if normalization_count is not None
        else (context.world_size if active_ranks is None else int(active_ranks))
    )
    if divisor < 1:
        raise ValueError(
            f"gradient normalization count must be positive, got {divisor}"
        )
    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    grouped: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        if parameter.requires_grad:
            grouped.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    for (_, dtype), group in grouped.items():
        element_bytes = torch.empty((), dtype=dtype).element_size()
        max_elements = max(1, bucket_bytes // element_bytes)
        bucket: list[torch.nn.Parameter] = []
        elements = 0

        def flush() -> None:
            nonlocal bucket, elements
            if not bucket:
                return
            for parameter in bucket:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(divisor)
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
                offset += count
            bucket = []
            elements = 0

        for parameter in group:
            count = parameter.numel()
            if bucket and elements + count > max_elements:
                flush()
            bucket.append(parameter)
            elements += count
            if elements >= max_elements:
                flush()
        flush()


def all_reduce_int(value: int, context: DistributedContext) -> int:
    if not context.enabled:
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=context.local_rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def close(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()
