"""
Parity test: load NousResearch/nns3_downsized into both HF (trust_remote_code)
and our NemotronSuperModel, run same input, compare logits.

Usage: python -m torchtitan.models.nemotron_super.parity_hf
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from torchtitan.models.nemotron_super import NemotronSuperModel, NemotronSuperModelArgs
from torchtitan.models.nemotron_super.model.state_dict_adapter import NemotronSuperStateDictAdapter
from torchtitan.models.moe import MoEArgs


HF_REPO = "NousResearch/nns3_downsized"


def main():
    device = "cuda"
    dtype = torch.bfloat16

    # 1. Load HF model
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_REPO, trust_remote_code=True, torch_dtype=dtype, device_map=device,
    )
    hf_model.eval()
    print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    # 2. Get HF state dict in original key format
    print("Extracting HF state dict...")
    hf_sd = {}
    for name, param in hf_model.named_parameters():
        hf_sd[name] = param.data.cpu()
    for name, buf in hf_model.named_buffers():
        hf_sd[name] = buf.cpu()

    # 3. Build our model with matching config
    # Pattern MEMEM*E = 3 blocks: ME, ME, M*E
    print("Building our model...")
    args = NemotronSuperModelArgs(
        dim=4096,
        n_layers=3,
        n_heads=32,
        n_kv_heads=2,
        head_dim=128,
        vocab_size=131072,
        max_seq_len=4096,
        norm_eps=1e-5,
        rope_theta=10000.0,
        attn_layer_idxs=[2],  # 3rd block has attention
        mamba_num_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        conv_kernel=4,
        chunk_size=128,
        n_groups=8,
        mamba_expand=2,
        mamba_hidden_act="silu",
        use_conv_bias=True,
        use_mamba_proj_bias=False,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=0.0001,
        num_nextn_predict_layers=0,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=22,
            score_func="sigmoid",
            route_norm=True,
            route_scale=5.0,
            gate_bias=False,
            score_before_experts=False,
            num_expert_groups=1,
            num_limited_groups=1,
            gated_experts=False,
            expert_act="relu2",
            expert_intermediate_size=2688,
            shared_expert_intermediate_size=5376,
            latent_size=1024,
            expert_bias=False,
        ),
    )

    our_model = NemotronSuperModel(args).to(device).to(dtype)

    # 4. Convert HF weights through adapter
    print("Converting weights through adapter...")
    adapter = NemotronSuperStateDictAdapter(args, None)
    titan_sd = adapter.from_hf(hf_sd)
    print(f"  Adapter produced {len(titan_sd)} keys")

    # Load into our model
    our_sd = our_model.state_dict()
    missing = set(our_sd.keys()) - set(titan_sd.keys())
    extra = set(titan_sd.keys()) - set(our_sd.keys())
    if missing:
        # Filter out non-essential (buffers that default to zero)
        real_missing = [k for k in missing if "rope_cache" not in k and "tokens_per_expert" not in k and "expert_routing" not in k]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} missing keys:")
            for k in sorted(real_missing)[:10]:
                print(f"    {k}")
    if extra:
        print(f"  WARNING: {len(extra)} extra keys from adapter")

    # Load with strict=False to allow missing buffers
    result = our_model.load_state_dict(titan_sd, strict=False)
    print(f"  Load result: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    our_model.eval()
    print(f"  Our params: {sum(p.numel() for p in our_model.parameters()):,}")

    # 5. Run same input through both
    print("\nRunning forward pass...")
    input_ids = torch.randint(1, 1000, (1, 32), device=device)

    with torch.no_grad():
        hf_out = hf_model(input_ids)
        hf_logits = hf_out.logits

        our_logits = our_model(input_ids)
        if isinstance(our_logits, tuple):
            our_logits = our_logits[0]

    # 6. Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"HF logits:  shape={hf_logits.shape}, std={hf_logits.float().std().item():.4f}")
    print(f"Our logits: shape={our_logits.shape}, std={our_logits.float().std().item():.4f}")

    diff = (hf_logits.float() - our_logits.float()).abs()
    print(f"Max abs diff:  {diff.max().item():.4f}")
    print(f"Mean abs diff: {diff.mean().item():.4f}")
    rel = diff.max().item() / (hf_logits.float().abs().max().item() + 1e-10)
    print(f"Max rel diff:  {rel:.2%}")

    # Check per-position
    per_pos_diff = diff.squeeze(0).max(dim=-1).values
    print(f"\nPer-position max diff (first 8): {per_pos_diff[:8].tolist()}")

    # Top-1 predictions
    hf_top1 = hf_logits.argmax(dim=-1)
    our_top1 = our_logits.argmax(dim=-1)
    match = (hf_top1 == our_top1).float().mean().item()
    print(f"Top-1 prediction match: {match:.1%}")

    if rel < 0.01:
        print("\nPASS")
    elif rel < 0.05:
        print("\nCLOSE (bf16 noise?)")
    else:
        print(f"\nFAIL - {rel:.1%} relative error")


if __name__ == "__main__":
    main()
