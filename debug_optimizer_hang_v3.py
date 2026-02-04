#!/usr/bin/env python3
"""
Minimal reproduction script - V3
Forces experts to have SEPARATE FSDP wrapper with different mesh (even size 1)
This should reproduce the hang.

Key: The hang occurs when:
1. Experts have their own FSDP wrapper with dp_mod_ep_mesh
2. Non-experts have FSDP wrapper with dp_mesh
3. Both use CPU offload
4. dp_mod_ep_mesh != dp_mesh (different process groups)

Run with: torchrun --nproc_per_node=8 debug_optimizer_hang_v3.py
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
        print(f"FSDP2 CPU Offload + EP Debug - V3 (Force different meshes)")
        print(f"=" * 70)
        print(f"World size: {world_size}")

    # Create 2D mesh to have two DIFFERENT submeshes
    # dp_mesh: all ranks (size world_size)
    # dp_mod_ep_mesh: groups of 2 ranks (size 2 per group)
    ep_groups = world_size // 2  # e.g., 4 groups of 2 for 8 GPUs

    if rank == 0:
        print(f"\nCreating 2D mesh: ({ep_groups}, 2)")
        print(f"  dp_mesh: flattened to all {world_size} ranks")
        print(f"  dp_mod_ep_mesh: size 2 per group (different from dp_mesh!)")

    # 2D mesh: (ep_groups, dp_shard_per_ep_group)
    mesh = init_device_mesh(
        "cuda", (ep_groups, 2), mesh_dim_names=("ep_group", "dp_in_ep")
    )

    # dp_mesh spans all ranks
    dp_mesh = mesh._flatten(mesh_dim_name="dp")

    # dp_mod_ep_mesh is just the inner dimension (size 2)
    dp_mod_ep_mesh = mesh["dp_in_ep"]

    if rank == 0:
        print(f"\nActual mesh sizes:")
        print(f"  dp_mesh size: {dp_mesh.size()}")
        print(f"  dp_mod_ep_mesh size: {dp_mod_ep_mesh.size()}")
        print(f"  These are DIFFERENT meshes - this is the bug trigger!")

    # Create model
    num_experts = 8
    if rank == 0:
        print(f"\nCreating model with {num_experts} experts...")
    model = SimpleModel(HIDDEN_DIM, num_experts, num_layers=4)

    # CPU offload configs - SAME offload but DIFFERENT meshes
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    fsdp_config = {"offload_policy": offload_policy, "mesh": dp_mesh}
    fsdp_mod_ep_config = {"offload_policy": offload_policy, "mesh": dp_mod_ep_mesh}

    # Apply FSDP with DIFFERENT meshes for experts vs rest
    if rank == 0:
        print(f"\nApplying FSDP2 with CPU offload and DIFFERENT meshes...")

    for i, layer in enumerate(model.layers):
        # KEY: Experts get FSDP with dp_mod_ep_mesh (size 2)
        if rank == 0:
            print(
                f"  Layer {i}: Wrapping experts with dp_mod_ep_mesh (size {dp_mod_ep_mesh.size()})"
            )
        fully_shard(layer.moe.experts, **fsdp_mod_ep_config)

        # Rest of layer gets FSDP with dp_mesh (all ranks)
        if rank == 0:
            print(
                f"  Layer {i}: Wrapping full layer with dp_mesh (size {dp_mesh.size()})"
            )
        fully_shard(layer, **fsdp_config)

    fully_shard(model.embed, **fsdp_config)
    fully_shard(model.output, **fsdp_config)
    fully_shard(model, **fsdp_config)

    if rank == 0:
        print(
            f"\nFSDP applied. Expert params use mesh size {dp_mod_ep_mesh.size()}, "
            f"others use mesh size {dp_mesh.size()}"
        )

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

        # Optimizer step - THIS SHOULD HANG
        if rank == 0:
            print(f"  Optimizer step START (expect hang here)...", flush=True)

        dist.barrier()
        if rank == 0:
            print(f"  All ranks at barrier before optimizer.step()", flush=True)

        optimizer.step()

        if rank == 0:
            print(f"  Optimizer step END (if you see this, no hang!)", flush=True)

        optimizer.zero_grad()

        if rank == 0:
            print(f"  Step {step + 1} COMPLETED!", flush=True)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"If you see this, the hang did NOT reproduce.")
        print(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
