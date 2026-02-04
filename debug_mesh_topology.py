#!/usr/bin/env python3
"""
Debug script to understand the exact mesh topology for Kimi K2 training.
This shows exactly what meshes are created and their sizes.

Run with: torchrun --nproc_per_node=8 debug_mesh_topology.py
(Simulates 64 GPUs on 8 GPUs by adjusting dimensions)
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def simulate_kimi_k2_mesh(rank, world_size):
    """
    Simulate the mesh topology for Kimi K2 with EP=64, CP=8, 64 GPUs
    But run on smaller scale (8 GPUs).

    For 64 GPUs, EP=64, CP=8:
    - dp_shard = 64 / (dp_replicate * cp * tp * pp) = 64 / (1 * 8 * 1 * 1) = 8
    - dp_shard_mod_ep = dp_shard * cp / ep = 8 * 8 / 64 = 1
    - dp_shard_in_ep = ep / cp = 64 / 8 = 8

    Mesh dimensions: (dp_shard_mod_ep=1, dp_shard_in_ep=8, cp=8)
    Total: 1 * 8 * 8 = 64 ranks

    For 8 GPUs simulation, we use (dp_shard_mod_ep=1, dp_shard_in_ep=2, cp=4):
    Total: 1 * 2 * 4 = 8 ranks
    """
    if rank == 0:
        print("\n" + "=" * 70)
        print("Simulating Kimi K2 Mesh Topology")
        print("=" * 70)
        print(f"\nActual world size: {world_size}")
        print("\nKimi K2 full config (64 GPUs):")
        print("  EP=64, CP=8, world_size=64")
        print("  dp_shard = 8")
        print("  dp_shard_mod_ep = 1 (key!)")
        print("  dp_shard_in_ep = 8")
        print("  Mesh: (1, 8, 8) = 64 ranks")

    # For 8 GPU simulation
    dp_shard_mod_ep = 1  # Always 1 in Kimi K2
    dp_shard_in_ep = 2  # Scaled down from 8
    cp = 4  # Scaled down from 8

    if rank == 0:
        print(f"\nSimulated config ({world_size} GPUs):")
        print(f"  dp_shard_mod_ep = {dp_shard_mod_ep}")
        print(f"  dp_shard_in_ep = {dp_shard_in_ep}")
        print(f"  cp = {cp}")
        print(
            f"  Mesh: ({dp_shard_mod_ep}, {dp_shard_in_ep}, {cp}) = {dp_shard_mod_ep * dp_shard_in_ep * cp} ranks"
        )

    # Build mesh like torchtitan's _build_mesh_with_ep
    # Note: dp_shard_mod_ep is ALWAYS included even when 1
    dims = [dp_shard_mod_ep, dp_shard_in_ep, cp]
    names = ["dp_shard_mod_ep", "dp_shard_in_ep", "cp"]

    if rank == 0:
        print(f"\nCreating mesh with dims={dims}, names={names}")

    mesh = init_device_mesh("cuda", tuple(dims), mesh_dim_names=tuple(names))

    # Create submeshes like torchtitan does
    # dp_mesh = flatten(dp_shard_mod_ep, dp_shard_in_ep) - but in deepseek_v3, it's dp_shard_cp
    # dp_shard_cp = flatten(dp_shard_mod_ep, dp_shard_in_ep, cp) = all ranks
    dp_shard_cp_mesh = mesh._flatten(mesh_dim_name="dp_shard_cp")

    # dp_mod_ep_mesh = just dp_shard_mod_ep dimension
    dp_mod_ep_mesh = mesh["dp_shard_mod_ep"]

    # ep_mesh = flatten(dp_shard_in_ep, cp)
    ep_mesh = mesh["dp_shard_in_ep", "cp"]._flatten(mesh_dim_name="ep")

    if rank == 0:
        print(f"\nSubmesh sizes:")
        print(
            f"  dp_shard_cp_mesh (for non-expert FSDP): {dp_shard_cp_mesh.size()} ranks"
        )
        print(f"  dp_mod_ep_mesh (for expert FSDP): {dp_mod_ep_mesh.size()} ranks")
        print(f"  ep_mesh (for Expert Parallelism): {ep_mesh.size()} ranks")

    # Show local rank info
    print(f"\n[Rank {rank}] My position in meshes:")
    print(f"  dp_shard_cp_mesh: rank in mesh = {dp_shard_cp_mesh.get_local_rank()}")
    print(f"  dp_mod_ep_mesh: rank in mesh = {dp_mod_ep_mesh.get_local_rank()}")

    # KEY INSIGHT: dp_mod_ep_mesh has size 1!
    # This means experts have NO FSDP sharding, they are local to each rank.
    # But they still have the FSDP wrapper for mixed precision.

    if rank == 0:
        print("\n" + "=" * 70)
        print("KEY INSIGHT:")
        print("=" * 70)
        print(f"  dp_mod_ep_mesh has SIZE {dp_mod_ep_mesh.size()}")
        print("  This means:")
        print("    - Expert params are NOT sharded across ranks")
        print("    - Each rank has its own copy of experts")
        print("    - FSDP wrapper is used for mixed precision, not sharding")
        print("    - During all-gather, only 1 rank participates (self)")
        print("    - This should NOT cause any collective communication issues!")
        print("=" * 70)

    return mesh, dp_shard_cp_mesh, dp_mod_ep_mesh, ep_mesh


def main():
    rank, world_size = setup_distributed()

    mesh, dp_shard_cp_mesh, dp_mod_ep_mesh, ep_mesh = simulate_kimi_k2_mesh(
        rank, world_size
    )

    if rank == 0:
        print("\n\nConclusion:")
        print("-" * 70)
        print("With dp_mod_ep_mesh size = 1:")
        print("  - Experts have FSDP wrapper but no actual sharding")
        print("  - All-gather for experts is a no-op (single rank)")
        print("  - The hang should NOT be caused by expert mesh")
        print("  - The hang must be caused by something else!")
        print("-" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
