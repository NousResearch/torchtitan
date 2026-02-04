#!/usr/bin/env python3
"""
Minimal reproduction script - V2
Matches EXACT mesh topology of Kimi K2 training: EP=64, CP=8, 64 GPUs

Key insight: With EP=64, CP=8:
- dp_shard_mod_ep = dp_shard * cp / ep = 8 * 8 / 64 = 1
- dp_mod_ep_mesh has SIZE 1 (no actual sharding for experts!)
- dp_mesh (dp_shard_cp) has SIZE 64 (all ranks)

The deadlock might be caused by:
1. Non-expert params need all 64 ranks for all-gather
2. Operations ordering inconsistency during optimizer step

Run with: torchrun --nproc_per_node=8 debug_optimizer_hang_v2.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

HIDDEN_DIM = 256
SEQ_LEN = 128
BATCH_SIZE = 2


class Expert(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.w2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class Experts(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])

    def forward(self, x, expert_idx=0):
        return self.experts[expert_idx](x)


class MoEBlock(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = Experts(num_experts, hidden_dim)

    def forward(self, x):
        return self.experts(x, expert_idx=0)


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.moe = MoEBlock(num_experts, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class SimpleModel(nn.Module):
    def __init__(self, hidden_dim, num_experts, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [SimpleTransformerBlock(hidden_dim, num_experts) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def main():
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if rank == 0:
        print(f"=" * 70)
        print(f"FSDP2 CPU Offload + EP Debug - V2 (Matching Kimi K2 topology)")
        print(f"=" * 70)
        print(f"World size: {world_size}")

    # Match Kimi K2 topology: EP degree = world_size (each rank has unique experts)
    # This makes dp_mod_ep_mesh have size 1 (no FSDP sharding for experts)
    ep_degree = world_size  # e.g., 8 on single node
    cp_degree = 1  # Simplified, no CP for this test

    # Calculate mesh dimensions like torchtitan does
    dp_shard = world_size // cp_degree  # = 8
    dp_shard_mod_ep = dp_shard * cp_degree // ep_degree  # = 8 * 1 / 8 = 1
    dp_shard_in_ep = ep_degree // cp_degree  # = 8 / 1 = 8

    if rank == 0:
        print(f"\nMesh topology (matching Kimi K2 pattern):")
        print(f"  EP degree: {ep_degree}")
        print(f"  CP degree: {cp_degree}")
        print(f"  dp_shard: {dp_shard}")
        print(f"  dp_shard_mod_ep: {dp_shard_mod_ep}")
        print(f"  dp_shard_in_ep: {dp_shard_in_ep}")

    # Build mesh like torchtitan's _build_mesh_with_ep
    # Dimensions: (dp_shard_mod_ep, dp_shard_in_ep, cp)
    # With cp=1, it's just (dp_shard_mod_ep, dp_shard_in_ep) = (1, 8)
    if dp_shard_mod_ep == 1:
        # 1D mesh since dp_shard_mod_ep=1
        mesh = init_device_mesh(
            "cuda", (dp_shard_in_ep,), mesh_dim_names=("dp_shard_in_ep",)
        )
        dp_mesh = mesh  # All ranks
        # dp_mod_ep_mesh would be size 1 - but we can't create a 0D mesh
        # In this case, experts have no FSDP sharding (local to each rank)
        dp_mod_ep_mesh = None

        if rank == 0:
            print(f"\nMesh structure:")
            print(f"  dp_mesh size: {dp_mesh.size()} (all ranks)")
            print(f"  dp_mod_ep_mesh: None (experts local to each rank, no sharding)")
    else:
        # 2D mesh
        mesh = init_device_mesh(
            "cuda",
            (dp_shard_mod_ep, dp_shard_in_ep),
            mesh_dim_names=("dp_shard_mod_ep", "dp_shard_in_ep"),
        )
        dp_mesh = mesh  # Flattened to all ranks
        dp_mod_ep_mesh = mesh["dp_shard_mod_ep"]

        if rank == 0:
            print(f"\nMesh structure:")
            print(f"  dp_mesh size: {dp_mesh.size()}")
            print(f"  dp_mod_ep_mesh size: {dp_mod_ep_mesh.size()}")

    # Create model
    num_experts = ep_degree  # Match EP degree
    if rank == 0:
        print(f"\nCreating model with {num_experts} experts...")
    model = SimpleModel(HIDDEN_DIM, num_experts, num_layers=4)

    # CPU offload config
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    fsdp_config = {"offload_policy": offload_policy, "mesh": dp_mesh}

    # Apply FSDP
    if rank == 0:
        print(f"\nApplying FSDP2 with CPU offload...")

    for i, layer in enumerate(model.layers):
        if dp_mod_ep_mesh is not None:
            # Experts get separate FSDP with smaller mesh
            fsdp_mod_ep_config = {
                "offload_policy": offload_policy,
                "mesh": dp_mod_ep_mesh,
            }
            if rank == 0:
                print(
                    f"  Layer {i}: Wrapping experts with dp_mod_ep_mesh (size {dp_mod_ep_mesh.size()})"
                )
            fully_shard(layer.moe.experts, **fsdp_mod_ep_config)
        else:
            # No FSDP for experts when dp_mod_ep_mesh would be size 1
            # Just apply FSDP to the whole layer
            if rank == 0:
                print(
                    f"  Layer {i}: Experts NOT wrapped (dp_mod_ep_mesh=None, local to rank)"
                )

        if rank == 0:
            print(
                f"  Layer {i}: Wrapping full layer with dp_mesh (size {dp_mesh.size()})"
            )
        fully_shard(layer, **fsdp_config)

    fully_shard(model.embed, **fsdp_config)
    fully_shard(model.output, **fsdp_config)
    fully_shard(model, **fsdp_config)

    # Create optimizer
    if rank == 0:
        print(f"\nCreating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)
    target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)

    # Training loop
    for step in range(3):
        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"Step {step + 1}")
            print(f"{'=' * 50}")

        # Forward
        if rank == 0:
            print(f"  Forward pass...", flush=True)
        output = model(x)
        loss = ((output - target) ** 2).mean()
        if rank == 0:
            print(f"  Loss: {loss.item():.4f}", flush=True)

        # Backward
        if rank == 0:
            print(f"  Backward pass...", flush=True)
        loss.backward()

        # Optimizer step
        if rank == 0:
            print(f"  Optimizer step START...", flush=True)

        dist.barrier()
        if rank == 0:
            print(f"  All ranks at barrier", flush=True)

        optimizer.step()

        if rank == 0:
            print(f"  Optimizer step END", flush=True)

        optimizer.zero_grad()

        if rank == 0:
            print(f"  Step {step + 1} COMPLETED!", flush=True)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
