#!/usr/bin/env python3
"""
LLEP Hyperparameter Impact Analysis for Kimi K2 1T Configuration.

Simulates the impact of LLEP's three hyperparameters (α, λ, m) on:
- TPS (relative to standard EP)
- Memory overhead per GPU
- Number of weight transfers
- GPU load balance

Uses realistic H100 cost models for GEMM, P2P, AllToAll.
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from dataclasses import dataclass

sns.set_theme(style="whitegrid", font_scale=1.1)

# ============================================================================
# Hardware model (H100 SXM, bf16)
# ============================================================================
H100_BF16_TFLOPS = 700  # realistic sustained (peak 990)
H100_NVLINK_BW_GBs = 450  # per-direction, effective after protocol overhead
H100_IB_BW_GBs = 50  # InfiniBand per-direction effective
NCCL_LATENCY_US = 5  # base NCCL call latency
P2P_LATENCY_US = 10  # base P2P latency
BARRIER_LATENCY_US = 50  # dist.barrier() cost
LPT_PLANNING_US_PER_EXPERT = 0.5  # pure Python overhead per expert in LPT

# ============================================================================
# Model config (Kimi K2 1T)
# ============================================================================
@dataclass
class ModelConfig:
    name: str
    num_experts: int
    top_k: int
    dim: int  # model dim
    hidden_dim: int  # moe_inter_dim
    num_moe_layers: int
    dtype_bytes: int = 2  # bf16


KIMI_K2 = ModelConfig(
    name="Kimi K2 (384E, top8, d=7168, h=2048)",
    num_experts=384,
    top_k=8,
    dim=7168,
    hidden_dim=2048,
    num_moe_layers=60,
)


# ============================================================================
# Cost models
# ============================================================================
def swiglu_flops_per_token(dim, hidden_dim):
    """FLOPs for one token through SwiGLU: silu(x@w1)*( x@w3) @ w2."""
    # 3 matmuls: x@w1 (dim*hidden), x@w3 (dim*hidden), h@w2 (hidden*dim)
    # Each matmul = 2*M*N FLOPs
    return 3 * 2 * dim * hidden_dim


def gemm_time_us(num_tokens, dim, hidden_dim, tflops=H100_BF16_TFLOPS):
    """Time in microseconds for SwiGLU GEMM with num_tokens."""
    if num_tokens == 0:
        return 0
    flops = num_tokens * swiglu_flops_per_token(dim, hidden_dim)
    return flops / (tflops * 1e6)  # TFLOPS -> FLOP/us


def expert_weight_bytes(dim, hidden_dim, dtype_bytes=2):
    """Bytes for one expert's weights (w1 + w2 + w3)."""
    return 3 * dim * hidden_dim * dtype_bytes


def p2p_time_us(num_bytes, bw_GBs=H100_NVLINK_BW_GBs):
    """P2P transfer time in microseconds."""
    if num_bytes == 0:
        return 0
    return P2P_LATENCY_US + num_bytes / (bw_GBs * 1e3)  # GB/s -> bytes/us


def a2a_time_us(num_bytes_per_rank, ep_size, bw_GBs=H100_NVLINK_BW_GBs):
    """AllToAll time estimate (ring-based)."""
    if num_bytes_per_rank == 0:
        return 0
    # Ring AllToAll: (ep_size-1)/ep_size * total_bytes / bandwidth
    total_bytes = num_bytes_per_rank * ep_size
    return NCCL_LATENCY_US + (ep_size - 1) / ep_size * total_bytes / (bw_GBs * 1e3)


# ============================================================================
# Imbalance scenarios
# ============================================================================
def generate_expert_loads(num_experts, total_tokens, imbalance_pct, num_hot_experts):
    """Generate expert load distribution with given imbalance."""
    loads = np.zeros(num_experts)
    hot_tokens = int(total_tokens * imbalance_pct / 100)
    cold_tokens = total_tokens - hot_tokens

    # Distribute hot tokens among hot experts
    per_hot = hot_tokens // max(num_hot_experts, 1)
    for i in range(min(num_hot_experts, num_experts)):
        loads[i] = per_hot

    # Distribute remaining among all other experts
    remaining_experts = num_experts - num_hot_experts
    if remaining_experts > 0:
        per_cold = cold_tokens // remaining_experts
        for i in range(num_hot_experts, num_experts):
            loads[i] = per_cold

    # Normalize to exact total
    loads = loads * (total_tokens / max(loads.sum(), 1))
    return loads.astype(int)


IMBALANCE_SCENARIOS = {
    "Balanced": (0, 1),
    "30% → 4 experts": (30, 4),
    "50% → 4 experts": (50, 4),
    "80% → 1 expert": (80, 1),
    "95% → 1 expert": (95, 1),
}


# ============================================================================
# LLEP simulation
# ============================================================================
def simulate_standard_ep(loads, ep_size, config):
    """Simulate standard EP: compute max GPU time, memory."""
    num_experts = len(loads)
    experts_per_gpu = num_experts // ep_size
    gpu_loads = np.zeros(ep_size)
    for e in range(num_experts):
        gpu_id = e // experts_per_gpu
        gpu_loads[gpu_id] += loads[e]

    max_load = gpu_loads.max()
    # Time = max GPU's GEMM time + 2 * AllToAll
    compute_us = gemm_time_us(max_load, config.dim, config.hidden_dim)
    tokens_per_rank = loads.sum() * config.top_k // ep_size
    a2a_bytes = tokens_per_rank * config.dim * config.dtype_bytes
    comm_us = 2 * a2a_time_us(a2a_bytes, ep_size)

    total_us = compute_us + comm_us

    # Memory: max GPU's tokens * dim * dtype (activations)
    act_mem_bytes = max_load * config.dim * config.dtype_bytes
    weight_mem_bytes = experts_per_gpu * expert_weight_bytes(config.dim, config.hidden_dim, config.dtype_bytes)

    return {
        "total_us": total_us,
        "compute_us": compute_us,
        "comm_us": comm_us,
        "gpu_loads": gpu_loads,
        "max_load": max_load,
        "act_mem_GB": act_mem_bytes / 1e9,
        "weight_mem_GB": weight_mem_bytes / 1e9,
        "num_transfers": 0,
    }


def simulate_llep(loads, ep_size, config, alpha, min_tokens, lambda_thresh):
    """Simulate LLEP with given hyperparameters."""
    num_experts = len(loads)
    experts_per_gpu = num_experts // ep_size

    # Compute native GPU loads
    native_loads = np.zeros(ep_size)
    for e in range(num_experts):
        gpu_id = e // experts_per_gpu
        native_loads[gpu_id] += loads[e]

    # Imbalance ratio check (lambda)
    mean_load = native_loads.mean()
    max_native_load = native_loads.max()
    imbalance_ratio = max_native_load / mean_load if mean_load > 0 else 1.0

    if lambda_thresh > 0 and imbalance_ratio < lambda_thresh:
        # Skip LLEP, fall back to standard EP
        result = simulate_standard_ep(loads, ep_size, config)
        result["llep_active"] = False
        result["imbalance_ratio"] = imbalance_ratio
        return result

    # LPT assignment (simplified simulation of Algorithm 2)
    total_tokens = loads.sum()
    balanced_tokens = total_tokens / ep_size
    max_tokens_per_gpu = int(alpha * balanced_tokens) if balanced_tokens > 0 else total_tokens

    # Sort experts by load (LPT ordering)
    sorted_experts = np.argsort(-loads)

    assigned_load = np.zeros(ep_size)
    pending_native = native_loads.copy()
    num_transfers = 0
    foreign_experts_per_gpu = [0] * ep_size  # count of foreign experts received

    for expert_id in sorted_experts:
        expert_tokens = loads[expert_id]
        if expert_tokens == 0:
            continue

        native_gpu = expert_id // experts_per_gpu
        pending_native[native_gpu] -= expert_tokens

        effective_load = lambda g: assigned_load[g] + pending_native[g]
        native_available = max_tokens_per_gpu - effective_load(native_gpu)

        if native_available >= expert_tokens:
            assigned_load[native_gpu] += expert_tokens
        elif native_available > 0:
            assigned_load[native_gpu] += native_available
            remaining = expert_tokens - native_available
            # Spill to least loaded
            while remaining > 0:
                candidates = [(g, effective_load(g)) for g in range(ep_size) if g != native_gpu]
                candidates.sort(key=lambda x: x[1])
                spilled = False
                for helper, _ in candidates:
                    avail = max_tokens_per_gpu - effective_load(helper)
                    if avail <= 0:
                        continue
                    chunk = min(remaining, avail)
                    if chunk < min_tokens and remaining > chunk:
                        continue
                    assigned_load[helper] += chunk
                    remaining -= chunk
                    num_transfers += 1
                    foreign_experts_per_gpu[helper] += 1
                    spilled = True
                    break
                if not spilled:
                    # Force assign to least loaded
                    helper = candidates[0][0]
                    assigned_load[helper] += remaining
                    num_transfers += 1
                    foreign_experts_per_gpu[helper] += 1
                    remaining = 0
        else:
            # Fully spill
            remaining = expert_tokens
            while remaining > 0:
                candidates = [(g, assigned_load[g] + pending_native[g]) for g in range(ep_size) if g != native_gpu]
                candidates.sort(key=lambda x: x[1])
                spilled = False
                for helper, _ in candidates:
                    avail = max_tokens_per_gpu - (assigned_load[helper] + pending_native[helper])
                    if avail <= 0:
                        continue
                    chunk = min(remaining, avail)
                    if chunk < min_tokens and remaining > chunk:
                        continue
                    assigned_load[helper] += chunk
                    remaining -= chunk
                    num_transfers += 1
                    foreign_experts_per_gpu[helper] += 1
                    spilled = True
                    break
                if not spilled:
                    helper = candidates[0][0]
                    assigned_load[helper] += remaining
                    num_transfers += 1
                    foreign_experts_per_gpu[helper] += 1
                    remaining = 0

    max_load = assigned_load.max()

    # Compute time
    compute_us = gemm_time_us(max_load, config.dim, config.hidden_dim)

    # AllToAll (same as standard EP — token volume is the same)
    tokens_per_rank = total_tokens * config.top_k // ep_size
    a2a_bytes = tokens_per_rank * config.dim * config.dtype_bytes
    comm_us = 2 * a2a_time_us(a2a_bytes, ep_size)

    # P2P weight transfer overhead
    weight_bytes = expert_weight_bytes(config.dim, config.hidden_dim, config.dtype_bytes)
    p2p_us = num_transfers * p2p_time_us(weight_bytes)

    # Barrier overhead (current implementation has one per layer)
    barrier_us = BARRIER_LATENCY_US

    # LPT planning overhead
    planning_us = num_experts * LPT_PLANNING_US_PER_EXPERT

    # Gradient anchor overhead (w1_local.sum() * 0.0 — current bug)
    anchor_flops = experts_per_gpu * config.dim * config.hidden_dim
    anchor_us = anchor_flops / (H100_BF16_TFLOPS * 1e6)

    total_us = compute_us + comm_us + p2p_us + barrier_us + planning_us + anchor_us

    # Memory
    act_mem_bytes = max_load * config.dim * config.dtype_bytes
    weight_mem_bytes = experts_per_gpu * weight_bytes
    foreign_weight_mem_bytes = max(foreign_experts_per_gpu) * weight_bytes

    return {
        "total_us": total_us,
        "compute_us": compute_us,
        "comm_us": comm_us,
        "p2p_us": p2p_us,
        "barrier_us": barrier_us,
        "planning_us": planning_us,
        "anchor_us": anchor_us,
        "gpu_loads": assigned_load,
        "max_load": max_load,
        "act_mem_GB": act_mem_bytes / 1e9,
        "weight_mem_GB": weight_mem_bytes / 1e9,
        "foreign_weight_mem_GB": foreign_weight_mem_bytes / 1e9,
        "num_transfers": num_transfers,
        "llep_active": True,
        "imbalance_ratio": imbalance_ratio,
        "foreign_experts_per_gpu": foreign_experts_per_gpu,
    }


# ============================================================================
# Plotting
# ============================================================================
OUTPUT_DIR = "/mnt/data/home/nous/kimi_1t_sft/torchtitan/worklogs"

def plot_1_alpha_sweep(config, ep_size=8):
    """Fig 1: Impact of α on speedup and memory across imbalance scenarios."""
    alphas = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    token_counts = {"24K tokens": 24576, "32K tokens": 32768}

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Impact of α (max_tokens_factor) on LLEP Performance\n{config.name}, EP={ep_size}, λ=1.3, m=1024",
                 fontsize=14, fontweight="bold")

    for col, (tok_label, B) in enumerate(token_counts.items()):
        total_tokens = B * ep_size * config.top_k

        speedups = {}
        mem_overheads = {}
        for scenario_name, (imb_pct, num_hot) in IMBALANCE_SCENARIOS.items():
            loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
            ep_result = simulate_standard_ep(loads, ep_size, config)

            sp_list = []
            mem_list = []
            for alpha in alphas:
                llep_result = simulate_llep(loads, ep_size, config, alpha, 1024, 1.3)
                speedup = ep_result["total_us"] / llep_result["total_us"] if llep_result["total_us"] > 0 else 1.0
                sp_list.append(speedup)
                mem_overhead = llep_result.get("foreign_weight_mem_GB", 0)
                mem_list.append(mem_overhead)

            speedups[scenario_name] = sp_list
            mem_overheads[scenario_name] = mem_list

        # Speedup subplot
        ax1 = axes[0, col]
        for scenario_name, sp_list in speedups.items():
            ax1.plot(alphas, sp_list, "o-", label=scenario_name, linewidth=2, markersize=6)
        ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="EP baseline")
        ax1.set_xlabel("α (max_tokens_factor)")
        ax1.set_ylabel("Speedup over Standard EP")
        ax1.set_title(f"Speedup — {tok_label}")
        ax1.legend(fontsize=9)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)

        # Memory overhead subplot
        ax2 = axes[1, col]
        for scenario_name, mem_list in mem_overheads.items():
            ax2.plot(alphas, mem_list, "s-", label=scenario_name, linewidth=2, markersize=6)
        ax2.set_xlabel("α (max_tokens_factor)")
        ax2.set_ylabel("Foreign Weight Memory Overhead (GB)")
        ax2.set_title(f"Memory Overhead — {tok_label}")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig1_alpha_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_2_lambda_sweep(config, ep_size=8):
    """Fig 2: Impact of λ on speedup across imbalance scenarios."""
    lambdas = [0.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
    token_counts = {"24K tokens": 24576, "32K tokens": 32768}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Impact of λ (adaptive_threshold) on LLEP Performance\n{config.name}, EP={ep_size}, α=1.0, m=1024",
                 fontsize=14, fontweight="bold")

    for col, (tok_label, B) in enumerate(token_counts.items()):
        total_tokens = B * ep_size * config.top_k
        ax = axes[col]

        for scenario_name, (imb_pct, num_hot) in IMBALANCE_SCENARIOS.items():
            loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
            ep_result = simulate_standard_ep(loads, ep_size, config)

            speedups = []
            for lam in lambdas:
                llep_result = simulate_llep(loads, ep_size, config, 1.0, 1024, lam)
                speedup = ep_result["total_us"] / llep_result["total_us"] if llep_result["total_us"] > 0 else 1.0
                speedups.append(speedup)
            ax.plot(lambdas, speedups, "o-", label=scenario_name, linewidth=2, markersize=6)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="EP baseline")
        ax.set_xlabel("λ (adaptive_threshold)")
        ax.set_ylabel("Speedup over Standard EP")
        ax.set_title(f"{tok_label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig2_lambda_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_3_min_tokens_sweep(config, ep_size=8):
    """Fig 3: Impact of m (min_tokens_per_gemm) on speedup and #transfers."""
    m_values = [64, 128, 256, 512, 1024, 2048, 4096]
    token_counts = {"24K tokens": 24576, "32K tokens": 32768}

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Impact of m (min_tokens_per_gemm) on LLEP Performance\n{config.name}, EP={ep_size}, α=1.0, λ=1.3",
                 fontsize=14, fontweight="bold")

    for col, (tok_label, B) in enumerate(token_counts.items()):
        total_tokens = B * ep_size * config.top_k

        for scenario_name, (imb_pct, num_hot) in IMBALANCE_SCENARIOS.items():
            loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
            ep_result = simulate_standard_ep(loads, ep_size, config)

            speedups = []
            transfers = []
            for m in m_values:
                llep_result = simulate_llep(loads, ep_size, config, 1.0, m, 1.3)
                speedup = ep_result["total_us"] / llep_result["total_us"] if llep_result["total_us"] > 0 else 1.0
                speedups.append(speedup)
                transfers.append(llep_result["num_transfers"])

            axes[0, col].plot(m_values, speedups, "o-", label=scenario_name, linewidth=2, markersize=6)
            axes[1, col].plot(m_values, transfers, "s-", label=scenario_name, linewidth=2, markersize=6)

        axes[0, col].axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        axes[0, col].set_xlabel("m (min_tokens_per_gemm)")
        axes[0, col].set_ylabel("Speedup over Standard EP")
        axes[0, col].set_title(f"Speedup — {tok_label}")
        axes[0, col].set_xscale("log", base=2)
        axes[0, col].legend(fontsize=9)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].set_xlabel("m (min_tokens_per_gemm)")
        axes[1, col].set_ylabel("Number of Weight Transfers")
        axes[1, col].set_title(f"Weight Transfers — {tok_label}")
        axes[1, col].set_xscale("log", base=2)
        axes[1, col].legend(fontsize=9)
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig3_min_tokens_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_4_cost_breakdown(config, ep_size=8):
    """Fig 4: Time breakdown (compute, A2A, P2P, barrier, planning) for each scenario."""
    B = 24576
    total_tokens = B * ep_size * config.top_k

    scenarios_to_plot = [
        ("Balanced", 0, 1),
        ("50% → 4 experts", 50, 4),
        ("80% → 1 expert", 80, 1),
        ("95% → 1 expert", 95, 1),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Time Breakdown: Standard EP vs LLEP per MoE Layer\n{config.name}, EP={ep_size}, 24K tokens, α=1.0, λ=1.3, m=1024",
                 fontsize=14, fontweight="bold")

    for idx, (name, imb_pct, num_hot) in enumerate(scenarios_to_plot):
        loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
        ep = simulate_standard_ep(loads, ep_size, config)
        llep = simulate_llep(loads, ep_size, config, 1.0, 1024, 1.3)

        ax = axes[idx]
        labels = ["Standard EP", "LLEP"]

        ep_compute = ep["compute_us"]
        ep_comm = ep["comm_us"]

        llep_compute = llep["compute_us"]
        llep_comm = llep["comm_us"]
        llep_p2p = llep.get("p2p_us", 0)
        llep_barrier = llep.get("barrier_us", 0)
        llep_planning = llep.get("planning_us", 0)
        llep_anchor = llep.get("anchor_us", 0)
        llep_overhead = llep_p2p + llep_barrier + llep_planning + llep_anchor

        x = np.arange(2)
        width = 0.5

        # Stacked bars
        bars_compute = [ep_compute, llep_compute]
        bars_comm = [ep_comm, llep_comm]
        bars_overhead = [0, llep_overhead]

        b1 = ax.bar(x, bars_compute, width, label="Compute (GEMM)", color="#2196F3")
        b2 = ax.bar(x, bars_comm, width, bottom=bars_compute, label="AllToAll", color="#FF9800")
        b3 = ax.bar(x, bars_overhead, width,
                     bottom=[c + a for c, a in zip(bars_compute, bars_comm)],
                     label="LLEP overhead\n(P2P+barrier+plan)", color="#F44336")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time (μs)")
        ax.set_title(name)
        if idx == 0:
            ax.legend(fontsize=8)

        # Add total time labels
        for i, total in enumerate([ep["total_us"], llep["total_us"]]):
            ax.text(i, total + 50, f"{total:.0f}μs", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig4_cost_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_5_gpu_load_balance(config, ep_size=8):
    """Fig 5: GPU load balance heatmap — EP vs LLEP across scenarios."""
    B = 24576
    total_tokens = B * ep_size * config.top_k

    scenarios = [
        ("Balanced", 0, 1),
        ("30% → 4", 30, 4),
        ("50% → 4", 50, 4),
        ("80% → 1", 80, 1),
        ("95% → 1", 95, 1),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    fig.suptitle(f"GPU Load Distribution: Standard EP (top) vs LLEP (bottom)\n{config.name}, EP={ep_size}, 24K tokens, α=1.0, λ=1.3, m=1024",
                 fontsize=14, fontweight="bold")

    all_loads = []
    for _, (name, imb_pct, num_hot) in enumerate(scenarios):
        loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
        ep = simulate_standard_ep(loads, ep_size, config)
        llep = simulate_llep(loads, ep_size, config, 1.0, 1024, 1.3)
        all_loads.append(ep["gpu_loads"])
        all_loads.append(llep["gpu_loads"])

    vmax = max(l.max() for l in all_loads)
    vmin = 0

    for col, (name, imb_pct, num_hot) in enumerate(scenarios):
        loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
        ep = simulate_standard_ep(loads, ep_size, config)
        llep = simulate_llep(loads, ep_size, config, 1.0, 1024, 1.3)

        # EP row
        ax_ep = axes[0, col]
        ep_loads = ep["gpu_loads"]
        colors = plt.cm.RdYlGn_r(Normalize(vmin=vmin, vmax=vmax)(ep_loads))
        bars = ax_ep.bar(range(ep_size), ep_loads, color=colors, edgecolor="black", linewidth=0.5)
        ax_ep.set_title(f"{name}", fontsize=10)
        if col == 0:
            ax_ep.set_ylabel("Tokens (Standard EP)")
        ax_ep.set_xticks(range(ep_size))
        ax_ep.set_xticklabels([f"G{i}" for i in range(ep_size)], fontsize=8)
        ax_ep.set_ylim(0, vmax * 1.1)
        ratio = ep_loads.max() / ep_loads.mean() if ep_loads.mean() > 0 else 1
        ax_ep.text(0.98, 0.95, f"max/mean={ratio:.2f}x", transform=ax_ep.transAxes,
                   ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # LLEP row
        ax_llep = axes[1, col]
        llep_loads = llep["gpu_loads"]
        colors = plt.cm.RdYlGn_r(Normalize(vmin=vmin, vmax=vmax)(llep_loads))
        bars = ax_llep.bar(range(ep_size), llep_loads, color=colors, edgecolor="black", linewidth=0.5)
        if col == 0:
            ax_llep.set_ylabel("Tokens (LLEP)")
        ax_llep.set_xticks(range(ep_size))
        ax_llep.set_xticklabels([f"G{i}" for i in range(ep_size)], fontsize=8)
        ax_llep.set_ylim(0, vmax * 1.1)
        ratio = llep_loads.max() / llep_loads.mean() if llep_loads.mean() > 0 else 1
        n_transfers = llep["num_transfers"]
        ax_llep.text(0.98, 0.95, f"max/mean={ratio:.2f}x\n{n_transfers} transfers",
                     transform=ax_llep.transAxes, ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig5_gpu_load_balance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_6_heatmap_alpha_vs_m(config, ep_size=8):
    """Fig 6: 2D heatmap of speedup as function of (α, m) for different imbalance levels."""
    alphas = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    m_values = [64, 128, 256, 512, 1024, 2048, 4096]
    B = 24576
    total_tokens = B * ep_size * config.top_k

    scenarios = [
        ("Balanced", 0, 1),
        ("50% → 4 experts", 50, 4),
        ("80% → 1 expert", 80, 1),
        ("95% → 1 expert", 95, 1),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Speedup Heatmap: α × m for Different Imbalance Levels\n{config.name}, EP={ep_size}, 24K tokens, λ=1.3",
                 fontsize=14, fontweight="bold")

    for col, (name, imb_pct, num_hot) in enumerate(scenarios):
        loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
        ep = simulate_standard_ep(loads, ep_size, config)

        speedup_grid = np.zeros((len(m_values), len(alphas)))
        for i, m in enumerate(m_values):
            for j, alpha in enumerate(alphas):
                llep = simulate_llep(loads, ep_size, config, alpha, m, 1.3)
                speedup_grid[i, j] = ep["total_us"] / llep["total_us"] if llep["total_us"] > 0 else 1.0

        ax = axes[col]
        im = ax.imshow(speedup_grid, cmap="RdYlGn", aspect="auto",
                       vmin=min(0.5, speedup_grid.min()), vmax=max(2.0, speedup_grid.max()))
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a}" for a in alphas], fontsize=9)
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f"{m}" for m in m_values], fontsize=9)
        ax.set_xlabel("α (max_tokens_factor)")
        if col == 0:
            ax.set_ylabel("m (min_tokens_per_gemm)")
        ax.set_title(name, fontsize=11)

        # Annotate cells
        for i in range(len(m_values)):
            for j in range(len(alphas)):
                val = speedup_grid[i, j]
                color = "white" if val < 0.8 or val > 1.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Speedup")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig6_heatmap_alpha_vs_m.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_7_memory_breakdown(config, ep_size=8):
    """Fig 7: Memory breakdown across scenarios — native weights vs foreign weights vs activations."""
    B = 24576
    total_tokens = B * ep_size * config.top_k

    scenarios = [
        ("Balanced", 0, 1),
        ("30% → 4", 30, 4),
        ("50% → 4", 50, 4),
        ("80% → 1", 80, 1),
        ("95% → 1", 95, 1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Peak Memory per GPU: Standard EP vs LLEP\n{config.name}, EP={ep_size}, 24K tokens, α=1.0, λ=1.3, m=1024",
                 fontsize=14, fontweight="bold")

    scenario_names = [s[0] for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.35

    ep_act_mem = []
    ep_weight_mem = []
    llep_act_mem = []
    llep_weight_mem = []
    llep_foreign_mem = []

    for name, imb_pct, num_hot in scenarios:
        loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
        ep = simulate_standard_ep(loads, ep_size, config)
        llep = simulate_llep(loads, ep_size, config, 1.0, 1024, 1.3)

        ep_act_mem.append(ep["act_mem_GB"])
        ep_weight_mem.append(ep["weight_mem_GB"])
        llep_act_mem.append(llep["act_mem_GB"])
        llep_weight_mem.append(llep["weight_mem_GB"])
        llep_foreign_mem.append(llep.get("foreign_weight_mem_GB", 0))

    # EP memory
    ax1 = axes[0]
    b1 = ax1.bar(x, ep_weight_mem, width, label="Native Weights", color="#2196F3")
    b2 = ax1.bar(x, ep_act_mem, width, bottom=ep_weight_mem, label="Activations (max GPU)", color="#FF9800")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=15, fontsize=9)
    ax1.set_ylabel("Peak Memory (GB)")
    ax1.set_title("Standard EP")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LLEP memory
    ax2 = axes[1]
    b1 = ax2.bar(x, llep_weight_mem, width, label="Native Weights", color="#2196F3")
    b2 = ax2.bar(x, llep_foreign_mem, width, bottom=llep_weight_mem, label="Foreign Weights (max GPU)", color="#F44336")
    bottom2 = [w + f for w, f in zip(llep_weight_mem, llep_foreign_mem)]
    b3 = ax2.bar(x, llep_act_mem, width, bottom=bottom2, label="Activations (max GPU)", color="#FF9800")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, rotation=15, fontsize=9)
    ax2.set_ylabel("Peak Memory (GB)")
    ax2.set_title("LLEP")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Match y-axis
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig7_memory_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_8_overhead_breakdown_pie(config, ep_size=8):
    """Fig 8: Pie chart showing LLEP overhead breakdown for 80% imbalance."""
    B = 24576
    total_tokens = B * ep_size * config.top_k

    scenarios = [
        ("Balanced", 0, 1),
        ("50% → 4 experts", 50, 4),
        ("80% → 1 expert", 80, 1),
        ("95% → 1 expert", 95, 1),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"LLEP Time Breakdown per MoE Layer\n{config.name}, EP={ep_size}, 24K tokens, α=1.0, λ=1.3, m=1024",
                 fontsize=14, fontweight="bold")

    for idx, (name, imb_pct, num_hot) in enumerate(scenarios):
        loads = generate_expert_loads(config.num_experts, total_tokens, imb_pct, num_hot)
        llep = simulate_llep(loads, ep_size, config, 1.0, 1024, 1.3)

        components = {
            "Compute (GEMM)": llep["compute_us"],
            "AllToAll": llep["comm_us"],
            "P2P weights": llep.get("p2p_us", 0),
            "Barrier": llep.get("barrier_us", 0),
            "LPT planning": llep.get("planning_us", 0),
            "Gradient anchor": llep.get("anchor_us", 0),
        }

        # Filter out zeros
        labels = [k for k, v in components.items() if v > 0]
        values = [v for v in components.values() if v > 0]
        colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#4CAF50", "#795548"][:len(labels)]

        ax = axes[idx]
        wedges, texts, autotexts = ax.pie(
            values, labels=None, autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
            colors=colors, textprops={"fontsize": 8}
        )
        ax.set_title(f"{name}\nTotal: {llep['total_us']:.0f}μs", fontsize=10)
        if idx == 0:
            ax.legend(labels, loc="lower left", fontsize=7)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig8_overhead_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_9_combined_sensitivity(config, ep_size=8):
    """Fig 9: 3-panel sensitivity — speedup vs each hyperparameter at 80% imbalance, varying the other two."""
    B = 24576
    total_tokens = B * ep_size * config.top_k
    loads_80 = generate_expert_loads(config.num_experts, total_tokens, 80, 1)
    ep_result = simulate_standard_ep(loads_80, ep_size, config)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Hyperparameter Sensitivity at 80% Imbalance (1 hot expert)\n{config.name}, EP={ep_size}, 24K tokens",
                 fontsize=14, fontweight="bold")

    # Panel 1: Sweep α, fix m, vary λ
    alphas = np.arange(0.8, 2.05, 0.05)
    ax = axes[0]
    for lam in [0.0, 1.1, 1.3, 1.5, 2.0]:
        speedups = []
        for alpha in alphas:
            llep = simulate_llep(loads_80, ep_size, config, alpha, 1024, lam)
            speedups.append(ep_result["total_us"] / llep["total_us"])
        ax.plot(alphas, speedups, "-", label=f"λ={lam}", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("α (max_tokens_factor)")
    ax.set_ylabel("Speedup over EP")
    ax.set_title("Sweep α, vary λ (m=1024)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Sweep m, fix α, vary λ
    m_values = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ax = axes[1]
    for lam in [0.0, 1.1, 1.3, 1.5, 2.0]:
        speedups = []
        for m in m_values:
            llep = simulate_llep(loads_80, ep_size, config, 1.0, int(m), lam)
            speedups.append(ep_result["total_us"] / llep["total_us"])
        ax.plot(m_values, speedups, "-", label=f"λ={lam}", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("m (min_tokens_per_gemm)")
    ax.set_ylabel("Speedup over EP")
    ax.set_title("Sweep m, vary λ (α=1.0)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Sweep λ, fix m, vary α
    lambdas = np.arange(0.0, 3.05, 0.1)
    ax = axes[2]
    for alpha in [0.8, 1.0, 1.1, 1.3, 2.0]:
        speedups = []
        for lam in lambdas:
            llep = simulate_llep(loads_80, ep_size, config, alpha, 1024, lam)
            speedups.append(ep_result["total_us"] / llep["total_us"])
        ax.plot(lambdas, speedups, "-", label=f"α={alpha}", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("λ (adaptive_threshold)")
    ax.set_ylabel("Speedup over EP")
    ax.set_title("Sweep λ, vary α (m=1024)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/llep_fig9_combined_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    config = KIMI_K2
    ep_size = 8

    print(f"Generating LLEP hyperparameter analysis for {config.name}...")
    print(f"EP size: {ep_size}")
    print(f"Expert weight size: {expert_weight_bytes(config.dim, config.hidden_dim) / 1e6:.1f} MB per expert")
    print(f"SwiGLU FLOPs per token: {swiglu_flops_per_token(config.dim, config.hidden_dim) / 1e6:.1f} MFLOP")
    print()

    plot_1_alpha_sweep(config, ep_size)
    plot_2_lambda_sweep(config, ep_size)
    plot_3_min_tokens_sweep(config, ep_size)
    plot_4_cost_breakdown(config, ep_size)
    plot_5_gpu_load_balance(config, ep_size)
    plot_6_heatmap_alpha_vs_m(config, ep_size)
    plot_7_memory_breakdown(config, ep_size)
    plot_8_overhead_breakdown_pie(config, ep_size)
    plot_9_combined_sensitivity(config, ep_size)

    print("\nAll figures saved to:", OUTPUT_DIR)
