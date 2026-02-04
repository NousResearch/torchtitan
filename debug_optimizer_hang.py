#!/usr/bin/env python3
"""
Minimal reproduction script for FSDP2 CPU offload + Expert Parallelism optimizer hang.

This script reproduces the issue where optimizer.step() hangs when:
1. FSDP2 is used with CPU offload
2. Expert modules use a different mesh (dp_mod_ep_mesh) than non-expert modules (dp_mesh)

Run with: torchrun --nproc_per_node=8 debug_optimizer_hang.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

# Configuration
NUM_EXPERTS = 8
HIDDEN_DIM = 256
SEQ_LEN = 128
BATCH_SIZE = 2


class Expert(nn.Module):
    """Simple expert MLP."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.w2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class Experts(nn.Module):
    """Container for multiple experts."""

    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])

    def forward(self, x, expert_idx=0):
        return self.experts[expert_idx](x)


class MoEBlock(nn.Module):
    """Simple MoE block with router and experts."""

    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = Experts(num_experts, hidden_dim)

    def forward(self, x):
        # Simplified routing - just use expert 0 for testing
        return self.experts(x, expert_idx=0)


class SimpleTransformerBlock(nn.Module):
    """Simplified transformer block with attention and MoE."""

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
    """Simple model with multiple transformer blocks."""

    def __init__(self, hidden_dim, num_experts, num_layers=2):
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
    """Initialize distributed environment."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def apply_fsdp_with_ep(model, dp_mesh, dp_mod_ep_mesh, enable_cpu_offload=True):
    """Apply FSDP2 with different meshes for experts vs non-experts."""
    rank = dist.get_rank()

    # CPU offload config
    if enable_cpu_offload:
        offload_policy = CPUOffloadPolicy(pin_memory=True)
        fsdp_config = {"offload_policy": offload_policy, "mesh": dp_mesh}
        fsdp_mod_ep_config = {"offload_policy": offload_policy, "mesh": dp_mod_ep_mesh}
    else:
        fsdp_config = {"mesh": dp_mesh}
        fsdp_mod_ep_config = {"mesh": dp_mod_ep_mesh}

    # Apply FSDP to each layer
    for i, layer in enumerate(model.layers):
        # First, wrap experts with dp_mod_ep_mesh
        if rank == 0:
            print(f"  Wrapping layer {i} experts with dp_mod_ep_mesh", flush=True)
        fully_shard(layer.moe.experts, **fsdp_mod_ep_config)

        # Then, wrap the whole layer with dp_mesh
        if rank == 0:
            print(f"  Wrapping layer {i} with dp_mesh", flush=True)
        fully_shard(layer, **fsdp_config)

    # Wrap embedding and output
    fully_shard(model.embed, **fsdp_config)
    fully_shard(model.output, **fsdp_config)

    # Wrap the whole model
    fully_shard(model, **fsdp_config)

    return model


def main():
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if rank == 0:
        print(f"=" * 60)
        print(f"FSDP2 CPU Offload + Expert Parallelism Debug Script")
        print(f"=" * 60)
        print(f"World size: {world_size}")
        print(f"Num experts: {NUM_EXPERTS}")
        print(f"Hidden dim: {HIDDEN_DIM}")
        print(f"=" * 60)

    # Create device meshes
    # dp_mesh: all ranks (for non-expert params)
    # dp_mod_ep_mesh: subset of ranks (for expert params) - simulate EP
    if rank == 0:
        print(f"\nCreating device meshes...", flush=True)

    # For simplicity, we'll use:
    # - dp_mesh: 1D mesh with all ranks
    # - dp_mod_ep_mesh: smaller mesh (half the ranks per group) to simulate EP
    dp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    # Simulate EP by creating a mesh where each group has world_size/2 ranks
    # This is a simplified version - real EP would have more complex mesh setup
    ep_degree = min(2, world_size)  # EP degree of 2
    dp_mod_ep_size = world_size // ep_degree

    if dp_mod_ep_size > 1:
        # Create a 2D mesh and use submesh for dp_mod_ep
        mesh_2d = init_device_mesh(
            "cuda", (ep_degree, dp_mod_ep_size), mesh_dim_names=("ep", "dp_mod_ep")
        )
        dp_mod_ep_mesh = mesh_2d["dp_mod_ep"]
    else:
        # Fall back to full mesh if EP would give single rank groups
        dp_mod_ep_mesh = dp_mesh

    if rank == 0:
        print(f"dp_mesh size: {dp_mesh.size()}", flush=True)
        print(f"dp_mod_ep_mesh size: {dp_mod_ep_mesh.size()}", flush=True)
        print(f"EP degree: {ep_degree}", flush=True)

    # Create model
    if rank == 0:
        print(f"\nCreating model...", flush=True)
    model = SimpleModel(HIDDEN_DIM, NUM_EXPERTS, num_layers=2)

    # Apply FSDP with EP - model stays on CPU for CPU offload
    if rank == 0:
        print(f"\nApplying FSDP2 with CPU offload and EP...", flush=True)
    model = apply_fsdp_with_ep(model, dp_mesh, dp_mod_ep_mesh, enable_cpu_offload=True)
    # NOTE: Don't move model to device - FSDP2 CPU offload keeps params on CPU

    # Create optimizer
    if rank == 0:
        print(f"\nCreating optimizer...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)
    target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)

    # Training loop
    for step in range(3):
        if rank == 0:
            print(f"\n{'=' * 40}")
            print(f"Step {step + 1}")
            print(f"{'=' * 40}")

        # Forward pass
        if rank == 0:
            print(f"  Forward pass...", flush=True)
        output = model(x)
        loss = ((output - target) ** 2).mean()

        if rank == 0:
            print(f"  Loss: {loss.item():.4f}", flush=True)

        # Backward pass
        if rank == 0:
            print(f"  Backward pass...", flush=True)
        loss.backward()

        # Optimizer step - THIS IS WHERE THE HANG OCCURS
        if rank == 0:
            print(f"  Optimizer step START...", flush=True)

        dist.barrier()  # Sync before optimizer step
        if rank == 0:
            print(f"  All ranks at barrier before optimizer step", flush=True)

        optimizer.step()

        if rank == 0:
            print(f"  Optimizer step END", flush=True)

        optimizer.zero_grad()

        if rank == 0:
            print(f"  Step {step + 1} COMPLETED!", flush=True)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
