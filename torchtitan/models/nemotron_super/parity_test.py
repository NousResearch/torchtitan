"""
Parity test: compare our NemotronSuperModel against HF probes.

Step 1: Generate HF probes (run with hfvenv):
  source ~/Downloads/hfvenv/.venv/bin/activate
  python /tmp/hf_probe.py

Step 2: Compare (run with torchtitan venv):
  python -m torchtitan.models.nemotron_super.parity_test

Expects /tmp/hf_probes.pt from step 1.
"""
import torch
import warnings
from pathlib import Path
from safetensors import safe_open

from torchtitan.models.nemotron_super import NemotronSuperModel, NemotronSuperModelArgs
from torchtitan.models.nemotron_super.model.state_dict_adapter import NemotronSuperStateDictAdapter
from torchtitan.models.moe import MoEArgs

warnings.filterwarnings("ignore")

PROBES_PATH = "/tmp/hf_probes.pt"
HF_CKPT = str(Path.home() / ".cache/huggingface/hub/models--NousResearch--nns3_downsized/snapshots/f2a4b84e49d7bed795b501ab02e12b0e2517fdf6")

# Pattern MEMEM*E = 3 blocks: (M,E) (M,E) (M,*,E)
# HF flat layers: 0=M, 1=E, 2=M, 3=E, 4=M, 5=*, 6=E
# After flat 0 (mamba) = block 0 after mamba
# After flat 1 (moe) = block 0 output
# After flat 2 (mamba) = block 1 after mamba
# After flat 3 (moe) = block 1 output
# After flat 4 (mamba) = block 2 after mamba
# After flat 5 (attn) = block 2 after attn
# After flat 6 (moe) = block 2 output

MODEL_ARGS = NemotronSuperModelArgs(
    dim=4096, n_layers=3, n_heads=32, n_kv_heads=2, head_dim=128,
    vocab_size=131072, max_seq_len=4096, norm_eps=1e-5, rope_theta=10000.0,
    attn_layer_idxs=[2],
    mamba_num_heads=128, mamba_head_dim=64, ssm_state_size=128,
    conv_kernel=4, chunk_size=128, n_groups=8, mamba_expand=2,
    mamba_hidden_act="silu", use_conv_bias=True, use_mamba_proj_bias=False,
    time_step_min=0.001, time_step_max=0.1, time_step_floor=0.0001,
    num_nextn_predict_layers=0,
    moe_args=MoEArgs(
        num_experts=512, num_shared_experts=1, top_k=22,
        score_func="sigmoid", route_norm=True, route_scale=5.0,
        gate_bias=False, score_before_experts=False,
        num_expert_groups=1, num_limited_groups=1,
        gated_experts=False, expert_act="relu2",
        expert_intermediate_size=2688, shared_expert_intermediate_size=5376,
        latent_size=1024, expert_bias=False,
    ),
)


def load_model():
    adapter = NemotronSuperStateDictAdapter(MODEL_ARGS, None)
    hf_sd = {}
    with safe_open(f"{HF_CKPT}/model.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            hf_sd[k] = f.get_tensor(k)
    titan_sd = adapter.from_hf(hf_sd)
    del hf_sd

    model = NemotronSuperModel(MODEL_ARGS).to(torch.bfloat16)
    model.load_state_dict(titan_sd, strict=False)
    del titan_sd

    for layer in model.layers.values():
        layer.mamba._force_torch_path = True

    model.cuda().eval()
    return model


def main():
    print(f"Loading probes from {PROBES_PATH}...")
    probes = torch.load(PROBES_PATH, map_location="cpu", weights_only=True)
    seq_len = probes["input_ids"].shape[1]
    print(f"Sequence length: {seq_len}")

    print("Loading model...")
    model = load_model()
    input_ids = probes["input_ids"].cuda()

    # Hook everything in one forward pass
    outs = {}
    def hook(name):
        def fn(mod, inp, out):
            i = inp[0] if isinstance(inp, tuple) else inp
            o = out[0] if isinstance(out, tuple) else out
            outs[f"{name}_in"] = i.detach().cpu()
            outs[f"{name}_out"] = o.detach().cpu()
        return fn

    hooks = []
    hooks.append(model.tok_embeddings.register_forward_hook(hook("emb")))
    for block_id, layer in model.layers.items():
        hooks.append(layer.mamba_norm.register_forward_hook(hook(f"b{block_id}_mamba_norm")))
        hooks.append(layer.mamba.register_forward_hook(hook(f"b{block_id}_mamba")))
        if layer.has_attn:
            hooks.append(layer.attn_norm.register_forward_hook(hook(f"b{block_id}_attn_norm")))
            hooks.append(layer.attention.register_forward_hook(hook(f"b{block_id}_attn")))
        hooks.append(layer.ffn_norm.register_forward_hook(hook(f"b{block_id}_ffn_norm")))
        hooks.append(layer.moe.register_forward_hook(hook(f"b{block_id}_moe")))

    print("Forward pass...")
    with torch.no_grad():
        our_logits = model(input_ids).cpu()
    for h in hooks:
        h.remove()

    # === Logits ===
    hf_logits = probes["logits"].float()
    diff = (hf_logits - our_logits.float()).abs()
    rel = diff.max().item() / (hf_logits.abs().max().item() + 1e-10)
    match = (hf_logits.argmax(-1) == our_logits.float().argmax(-1)).float().mean().item()
    print(f"\n=== Logits ===")
    print(f"  HF std={hf_logits.std():.4f}  Our std={our_logits.float().std():.4f}")
    print(f"  Max rel diff: {rel:.2%}  Top-1 match: {match:.1%}")

    # === Embeddings ===
    print(f"\n=== Embeddings ===")
    d = (probes["embeddings"].float() - outs["emb_out"].float()).abs().max().item()
    print(f"  diff: {d:.2e}")

    # === Per sublayer ===
    # Map: (hf_flat, sublayer_desc) -> (our_hook_key for input, our_hook_key for computing output)
    print(f"\n=== Per sublayer ===")
    checks = [
        # block 0: flat 0=mamba, flat 1=moe
        (0, "mamba input", "b0_mamba_norm_in", probes["layer_0_input"]),
        (0, "after mamba", None, probes["layer_0_output"]),
        (1, "moe input",   "b0_ffn_norm_in", probes["layer_1_input"]),
        (1, "after moe",   None, probes["layer_1_output"]),
        # block 1: flat 2=mamba, flat 3=moe
        (2, "mamba input", "b1_mamba_norm_in", probes["layer_2_input"]),
        (2, "after mamba", None, probes["layer_2_output"]),
        (3, "moe input",   "b1_ffn_norm_in", probes["layer_3_input"]),
        (3, "after moe",   None, probes["layer_3_output"]),
        # block 2: flat 4=mamba, flat 5=attn, flat 6=moe
        (4, "mamba input", "b2_mamba_norm_in", probes["layer_4_input"]),
        (4, "after mamba", None, probes["layer_4_output"]),
        (5, "attn input",  "b2_attn_norm_in", probes["layer_5_input"]),
        (5, "after attn",  None, probes["layer_5_output"]),
        (6, "moe input",   "b2_ffn_norm_in", probes["layer_6_input"]),
        (6, "after moe",   None, probes["layer_6_output"]),
    ]

    for flat_idx, label, our_key, hf_tensor in checks:
        hf = hf_tensor.float()
        if our_key is not None:
            ours = outs[our_key].float()
        else:
            # "after X" = input + X_output, need to reconstruct
            # For mamba: block_input + mamba_out
            # For moe: moe_norm_in + moe_out
            # For attn: attn_norm_in + attn_out
            if "mamba" in label:
                block = {0: "0", 2: "1", 4: "2"}[flat_idx]
                ours = outs[f"b{block}_mamba_norm_in"].float() + outs[f"b{block}_mamba_out"].float()
            elif "attn" in label:
                ours = outs["b2_attn_norm_in"].float() + outs["b2_attn_out"].float()
            elif "moe" in label:
                block = {1: "0", 3: "1", 6: "2"}[flat_idx]
                ours = outs[f"b{block}_ffn_norm_in"].float() + outs[f"b{block}_moe_out"].float()

        d = (hf - ours).abs().max().item()
        r = d / (hf.abs().max().item() + 1e-10)
        status = "OK" if r < 0.001 else "WARN" if r < 0.05 else "FAIL"
        print(f"  [{status:4s}] flat {flat_idx} {label:15s}  abs={d:.4f}  rel={r:.2%}")

    print(f"\n{'PASS' if rel < 0.02 else 'CLOSE' if rel < 0.15 else 'FAIL'}")


if __name__ == "__main__":
    main()
