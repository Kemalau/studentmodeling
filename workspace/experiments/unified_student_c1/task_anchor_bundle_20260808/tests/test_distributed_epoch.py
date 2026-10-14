from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from distributed_epoch import (  # noqa: E402
    DistributedContext,
    context_from_environment,
    global_position,
    master_address,
    positions_for_rank,
    rounds_for_examples,
    synchronize_gradients,
    initialize,
)


class DistributedEpochTest(unittest.TestCase):
    def test_initialize_binds_rank_local_cuda_device(self) -> None:
        context = DistributedContext(True, rank=2, local_rank=3, world_size=4)
        with (
            patch("distributed_epoch.torch.cuda.is_available", return_value=True),
            patch("distributed_epoch.torch.cuda.set_device") as set_device,
            patch("distributed_epoch.master_address", return_value="node-0"),
            patch("distributed_epoch.dist.init_process_group") as init_group,
            patch.dict(os.environ, {}, clear=True),
        ):
            initialize(context, 29641)
        set_device.assert_called_once_with(3)
        init_group.assert_called_once_with(
            backend="nccl",
            init_method="env://",
            rank=2,
            world_size=4,
            device_id=torch.device("cuda", 3),
            timeout=init_group.call_args.kwargs["timeout"],
        )
        self.assertEqual(
            init_group.call_args.kwargs["timeout"].total_seconds(),
            7200,
        )

    def test_initialize_rejects_nonpositive_timeout(self) -> None:
        context = DistributedContext(True, 0, 0, 2)
        with self.assertRaisesRegex(ValueError, "timeout"):
            initialize(context, 29641, timeout_seconds=0)

    def test_four_ranks_cover_each_example_exactly_once(self) -> None:
        n = 259845
        positions = [
            position
            for rank in range(4)
            for position in positions_for_rank(n, rank, 4)
        ]
        self.assertEqual(len(positions), n)
        self.assertEqual(set(positions), set(range(n)))
        self.assertEqual(rounds_for_examples(n, 4), 64962)
        self.assertEqual([len(positions_for_rank(n, rank, 4)) for rank in range(4)], [64962, 64961, 64961, 64961])

    def test_global_position_is_round_major(self) -> None:
        self.assertEqual([global_position(3, rank, 4) for rank in range(4)], [12, 13, 14, 15])

    def test_eight_rank_two_round_updates_match_four_rank_four_round_updates(self) -> None:
        total = 259845

        def updates(world_size: int, accumulation: int) -> list[list[int]]:
            result = []
            rounds = rounds_for_examples(total, world_size)
            for start_round in range(0, rounds, accumulation):
                batch = []
                for round_index in range(
                    start_round, min(start_round + accumulation, rounds)
                ):
                    batch.extend(
                        position
                        for rank in range(world_size)
                        if (
                            position := global_position(
                                round_index, rank, world_size
                            )
                        )
                        < total
                    )
                result.append(batch)
            return result

        four_rank = updates(4, 4)
        eight_rank = updates(8, 2)
        self.assertEqual(four_rank, eight_rank)
        self.assertEqual(len(eight_rank), 16241)
        self.assertEqual(eight_rank[-1], list(range(259840, 259845)))

    def test_sixteen_rank_one_round_updates_match_eight_rank_two_round_updates(self) -> None:
        total = 259845

        def updates(world_size: int, accumulation: int) -> list[list[int]]:
            result = []
            rounds = rounds_for_examples(total, world_size)
            for start_round in range(0, rounds, accumulation):
                result.append(
                    [
                        position
                        for round_index in range(
                            start_round, min(start_round + accumulation, rounds)
                        )
                        for rank in range(world_size)
                        if (
                            position := global_position(round_index, rank, world_size)
                        )
                        < total
                    ]
                )
            return result

        sixteen_rank = updates(16, 1)
        self.assertEqual(sixteen_rank, updates(8, 2))
        self.assertEqual(len(sixteen_rank), 16241)

    def test_master_address_uses_head_host_for_multinode(self) -> None:
        context = DistributedContext(True, rank=9, local_rank=1, world_size=16)
        with tempfile.TemporaryDirectory() as directory:
            hostfile = Path(directory) / "hosts"
            env = {
                "OMPI_MCA_orte_num_nodes": "2",
                "SINGULARITY_MPI_HOSTFILE_PATH": str(hostfile),
            }
            hostfile.write_text("node-0 slots=8\nnode-1 slots=8\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(master_address(context), "node-0")
            hostfile.write_text("node-0:8\nnode-1:8\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(master_address(context), "node-0")

    def test_master_address_fails_closed_without_multinode_hostfile(self) -> None:
        context = DistributedContext(True, rank=0, local_rank=0, world_size=16)
        with patch.dict(os.environ, {"OMPI_MCA_orte_num_nodes": "2"}, clear=True):
            with self.assertRaisesRegex(ValueError, "no MPI hostfile"):
                master_address(context)
        with patch.dict(os.environ, {"OMPI_MCA_orte_num_nodes": "1"}, clear=True):
            self.assertEqual(master_address(context), "127.0.0.1")

    def test_reads_mpi_environment_and_rejects_singleton(self) -> None:
        env = {
            "OMPI_COMM_WORLD_RANK": "2",
            "OMPI_COMM_WORLD_LOCAL_RANK": "2",
            "OMPI_COMM_WORLD_SIZE": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            value = context_from_environment(True)
        self.assertEqual((value.rank, value.local_rank, value.world_size), (2, 2, 4))
        with patch.dict(os.environ, {**env, "OMPI_COMM_WORLD_SIZE": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Invalid distributed environment"):
                context_from_environment(True)

    def test_gradient_sync_buckets_and_materializes_missing_gradients(self) -> None:
        first = torch.nn.Parameter(torch.zeros(3))
        second = torch.nn.Parameter(torch.zeros(2))
        first.grad = torch.ones_like(first)
        calls = []

        def fake_all_reduce(value, op=None):
            calls.append(value.numel())
            value.mul_(2)

        context = DistributedContext(True, rank=0, local_rank=0, world_size=4)
        with patch("distributed_epoch.dist.all_reduce", side_effect=fake_all_reduce):
            synchronize_gradients(
                [first, second], context, normalization_count=2, bucket_bytes=1024
            )
        self.assertEqual(calls, [5])
        torch.testing.assert_close(first.grad, torch.ones_like(first))
        torch.testing.assert_close(second.grad, torch.zeros_like(second))


if __name__ == "__main__":
    unittest.main()
