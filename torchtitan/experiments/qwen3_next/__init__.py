# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.dataloader import build_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen3next
from .model.args import Qwen3NextModelArgs
from .model.model import Qwen3NextModel
from .model.state_dict_adapter import Qwen3NextStateDictAdapter

__all__ = [
    "parallelize_qwen3next",
    "Qwen3NextModelArgs",
    "Qwen3NextModel",
    "qwen3next_configs",
]

# Adding different variants of the model

# =============================================================================
# MAIN MODEL CONFIGURATIONS
# =============================================================================

qwen3next_configs = {
    "10B_A500M": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1536,
        n_layers=32,
        n_heads=24,
        n_kv_heads=4,
        hidden_dim=4096,  # Same as 10B_A1B
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=512,  # Same as 10B_A1B
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=1,
            top_k=4,  # Reduced from 8 to halve active params
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True
        ),
    ),
    # ==========================================================================
    # TUNING CONFIGS: ~500-600M active, 4-10B total, top_k=8, sparsity=3.125%
    # Naming: {total}B_A{active}M_dim{dim}_hidden{hidden}_L{layers}_E{experts}_inter{moe_inter}
    # ==========================================================================

    # 4.0B total, 871M active, 27k tps
    "4B_A871M_dim1024_hidden1024_L32_E256_inter128": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1024,
        n_layers=32,
        n_heads=8,
        n_kv_heads=4,
        hidden_dim=1024,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=128,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 4.9B total, 851M active, 31k tps - BEST THROUGHPUT for ~5B scale
    "5B_A851M_dim1024_hidden1536_L28_E256_inter192": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1024,
        n_layers=28,
        n_heads=8,
        n_kv_heads=4,
        hidden_dim=1536,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=192,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 3.8B total, 922M active, 31k tps
    "4B_A922M_dim1280_hidden1024_L24_E256_inter128": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1280,
        n_layers=24,
        n_heads=10,
        n_kv_heads=5,
        hidden_dim=1024,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=128,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 3.8B total, 898M active, 28k tps
    "4B_A898M_dim1024_hidden1280_L32_E192_inter160": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1024,
        n_layers=32,
        n_heads=8,
        n_kv_heads=4,
        hidden_dim=1280,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=160,
        moe_args=MoEArgs(
            num_experts=192,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 8.4B total, 1.5B active, 18k tps - DEEP scaling (48 layers)
    "8B_A1500M_dim1024_hidden1536_L48_E256_inter192_deep": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1024,
        n_layers=48,
        n_heads=8,
        n_kv_heads=4,
        hidden_dim=1536,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=192,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 9.7B total, 1.7B active, 16k tps - DEEP scaling (56 layers)
    "10B_A1700M_dim1024_hidden1536_L56_E256_inter192_deep": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1024,
        n_layers=56,
        n_heads=8,
        n_kv_heads=4,
        hidden_dim=1536,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=192,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 8.4B total, 1.4B active, 26k tps - WIDE scaling (28 layers, larger dim)
    "8B_A1400M_dim1280_hidden2048_L28_E256_inter256_wide": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1280,
        n_layers=28,
        n_heads=10,
        n_kv_heads=5,
        hidden_dim=2048,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=256,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    # 10.8B total, 1.7B active, 24k tps - WIDE scaling (28 layers, larger dim)
    "11B_A1700M_dim1536_hidden2048_L28_E256_inter256_wide": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=4,
        hidden_dim=2048,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=256,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=0.001,
        ),
    ),
    "10B_A1B": Qwen3NextModelArgs(
        vocab_size=151936,
        max_seq_len=8192,
        head_dim=128,
        dim=1536,
        n_layers=32,
        n_heads=24,
        n_kv_heads=4,
        hidden_dim=4096,
        rope_theta=50000,
        moe_enabled=True,
        moe_inter_dim=512,
        #partial_rotary_factor=1.0,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True
        ),
    ),
    "40B_A3B": Qwen3NextModelArgs(
        n_layers=24,
        moe_enabled=True,
        moe_inter_dim=512,
        rope_theta=50000.0,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=10,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True
        )
    ),
    "80B_A3B": Qwen3NextModelArgs(
        moe_enabled=True,
        moe_inter_dim=512,
        rope_theta=50000,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=10,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True
        )
    ),

    # ==========================================================================
    # ABLATION CONFIGURATIONS - Smaller scale for faster iteration
    # Base: ~8B total, ~0.8B active, fits easily on single node
    # ==========================================================================

    # -------------------------------------------------------------------------
    # DENSE BASELINE (for comparison)
    # -------------------------------------------------------------------------
    "ablation_dense_baseline": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2730,  # Match active compute of MoE
        moe_enabled=False,
    ),

    # -------------------------------------------------------------------------
    # ROUTING MECHANISM ABLATIONS
    # -------------------------------------------------------------------------
    # Baseline: softmax + route_norm (Qwen3 style)
    "ablation_routing_baseline": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=1e-3,
        )
    ),
    # Sigmoid routing (DeepSeek style)
    "ablation_routing_sigmoid": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=False,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=1e-3,
        )
    ),
    # Higher route scale (sharper routing)
    "ablation_routing_scale_2": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=2.0,
            score_before_experts=False,
            shared_gate=True,
            load_balance_coeff=1e-3,
        )
    ),

    # -------------------------------------------------------------------------
    # EXPERT COUNT ABLATIONS (iso-active-compute via moe_inter_dim adjustment)
    # -------------------------------------------------------------------------
    # Fine-grained: 512 experts, top-10
    "ablation_experts_512_topk10": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=256,  # Smaller experts
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=10,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    # Medium: 256 experts, top-8 (baseline)
    "ablation_experts_256_topk8": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    # Coarse: 128 experts, top-6
    "ablation_experts_128_topk6": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=1024,  # Larger experts
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=1,
            top_k=6,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    # Very coarse: 64 experts, top-4
    "ablation_experts_64_topk4": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=2048,  # Much larger experts
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=1,
            top_k=4,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),

    # -------------------------------------------------------------------------
    # TOP-K ABLATIONS (fixed 256 experts)
    # -------------------------------------------------------------------------
    "ablation_topk_4": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=4,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    "ablation_topk_6": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=6,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    "ablation_topk_12": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=12,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    "ablation_topk_16": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=16,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),

    # -------------------------------------------------------------------------
    # LOAD BALANCE COEFFICIENT ABLATIONS
    # -------------------------------------------------------------------------
    "ablation_lb_none": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=None,  # No load balancing
        )
    ),
    "ablation_lb_1e4": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-4,  # Weak
        )
    ),
    "ablation_lb_1e2": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-2,  # Strong
        )
    ),

    # -------------------------------------------------------------------------
    # SHARED EXPERT ABLATIONS
    # -------------------------------------------------------------------------
    "ablation_shared_0": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,  # Kept but unused
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=0,  # No shared experts
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),
    "ablation_shared_2": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=1280,  # Smaller per shared expert
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=2,  # Two shared experts
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=1e-3,
        )
    ),

    # -------------------------------------------------------------------------
    # SHARED GATE ABLATION
    # -------------------------------------------------------------------------
    "ablation_shared_gate_off": Qwen3NextModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=128,
        hidden_dim=2560,
        moe_inter_dim=512,
        moe_enabled=True,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            shared_gate=False,  # No learnable gate for shared
            load_balance_coeff=1e-3,
        )
    ),
}

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Qwen3NextModel,
        model_args=qwen3next_configs,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen3next,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3NextStateDictAdapter,
    )