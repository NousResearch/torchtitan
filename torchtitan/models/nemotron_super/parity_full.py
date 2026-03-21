"""
Full-scale layer-by-layer parity test:
  Our NemotronSuperModel vs HF nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16

Runs in 3 phases (process-isolated via disk for stability):
  Phase 1: HF model forward -> probes saved to /tmp/hf_probes.pt
  Phase 2: Our model forward (weights from safetensors) -> probes saved
  Phase 3: Compare all sublayer probes

Usage:
  python -m torchtitan.models.nemotron_super.parity_full [--seq-len 64] [--seed 42]
"""

import argparse
import gc
import os
import sys
import time
from collections import OrderedDict

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from torchtitan.models.nemotron_super import NemotronSuperModel, NemotronSuperModelArgs
from torchtitan.models.nemotron_super.model.state_dict_adapter import NemotronSuperStateDictAdapter
from torchtitan.models.moe import MoEArgs


HF_REPO = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"
HF_PROBES = "/tmp/nemotron_hf_probes.pt"
OUR_PROBES = "/tmp/nemotron_our_probes.pt"
NUM_GPUS = torch.cuda.device_count()

MODEL_ARGS = NemotronSuperModelArgs(
    dim=4096, n_layers=40, n_heads=32, n_kv_heads=2, head_dim=128,
    vocab_size=131072, max_seq_len=4096, norm_eps=1e-5, rope_theta=10000.0,
    attn_layer_idxs=[3, 7, 11, 16, 21, 26, 31, 35],
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


def gpu_mem():
    return f"GPU mem: {sum(torch.cuda.memory_allocated(i)/1e9 for i in range(NUM_GPUS)):.1f}GB"


def free_gpu():
    gc.collect()
    torch.cuda.synchronize()
    for i in range(NUM_GPUS):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    torch.cuda.synchronize()


def fmt_diff(abs_diff, ref_abs_max):
    rel = abs_diff / (ref_abs_max + 1e-10)
    if rel < 1e-4:
        return " OK ", abs_diff, rel
    elif rel < 0.01:
        return "FINE", abs_diff, rel
    elif rel < 0.05:
        return "WARN", abs_diff, rel
    else:
        return "FAIL", abs_diff, rel


# ===========================================================================
# Phase 1: HF model
# ===========================================================================
def phase1_hf(input_ids):
    print("=" * 72)
    print("PHASE 1: HF model forward")
    print("=" * 72)

    t0 = time.time()
    print(f"Loading {HF_REPO} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_REPO, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
    )
    hf_model.eval()
    print(f"  Loaded {sum(p.numel() for p in hf_model.parameters())/1e9:.1f}B params in {time.time()-t0:.0f}s")

    backbone = hf_model.model
    layers = backbone.layers
    pattern = hf_model.config.hybrid_override_pattern
    print(f"  {len(layers)} flat layers")

    # Force torch path on HF mamba layers too (consistent numerics with our model)
    n_forced = 0
    for layer in layers:
        if hasattr(layer, "mixer") and hasattr(layer.mixer, "_force_torch_path"):
            layer.mixer._force_torch_path = True
            n_forced += 1
        elif hasattr(layer, "mixer") and "Mamba" in type(layer.mixer).__name__:
            layer.mixer._force_torch_path = True
            n_forced += 1
    print(f"  Forced torch path on {n_forced} HF mamba layers")

    # --- Hooks: capture residual stream at every flat layer boundary ---
    probes = OrderedDict()

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                probes[f"{name}_in"] = inp[0].detach().float().cpu()
            o = out[0] if isinstance(out, tuple) else out
            probes[f"{name}_out"] = o.detach().float().cpu()
        return hook_fn

    hooks = []
    hooks.append(backbone.embeddings.register_forward_hook(make_hook("emb")))
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(f"flat_{i}_{pattern[i]}")))
    if hasattr(backbone, "norm_f"):
        hooks.append(backbone.norm_f.register_forward_hook(make_hook("final_norm")))

    print(f"  Forward pass ...")
    t1 = time.time()
    with torch.no_grad():
        hf_out = hf_model(input_ids.to(hf_model.device))
    probes["logits"] = hf_out.logits.detach().float().cpu()
    print(f"  Done in {time.time()-t1:.1f}s  logits std={probes['logits'].std():.4f}")

    for h in hooks:
        h.remove()

    # Save probes to disk
    probes["pattern"] = pattern
    probes["input_ids"] = input_ids.cpu()
    print(f"  Saving {len(probes)} probes to {HF_PROBES} ...")
    torch.save(probes, HF_PROBES)

    # Cleanup
    del hf_model, hf_out, probes
    free_gpu()
    print(f"  Cleanup done. {gpu_mem()}")


# ===========================================================================
# Phase 2: Our model (load weights directly from safetensors)
# ===========================================================================
def phase2_ours(input_ids):
    print("\n" + "=" * 72)
    print("PHASE 2: Our model forward")
    print("=" * 72)

    # Find the safetensors snapshot
    from huggingface_hub import snapshot_download
    snap = snapshot_download(HF_REPO, local_files_only=True)
    print(f"  Snapshot: {snap}")

    # Load all safetensors into a single state dict
    print("  Loading safetensors + converting through adapter ...")
    t0 = time.time()
    import glob
    st_files = sorted(glob.glob(os.path.join(snap, "model-*.safetensors")))
    print(f"  {len(st_files)} shard files")

    hf_sd = {}
    for sf in st_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_sd[key] = f.get_tensor(key)

    print(f"  Loaded {len(hf_sd)} HF tensors in {time.time()-t0:.0f}s")

    # Convert through adapter
    adapter = NemotronSuperStateDictAdapter(MODEL_ARGS, None)
    titan_sd = adapter.from_hf(hf_sd)
    del hf_sd
    gc.collect()
    print(f"  Adapter produced {len(titan_sd)} keys")

    # Build model
    print("  Building NemotronSuperModel ...")
    t1 = time.time()
    model = NemotronSuperModel(MODEL_ARGS).to(torch.bfloat16)
    result = model.load_state_dict(titan_sd, strict=False)
    real_missing = [k for k in result.missing_keys
                    if "rope_cache" not in k and "tokens_per_expert" not in k
                    and "expert_routing" not in k]
    if real_missing:
        print(f"  WARNING: {len(real_missing)} missing keys:")
        for k in sorted(real_missing)[:5]:
            print(f"    {k}")
    del titan_sd
    gc.collect()
    print(f"  Built in {time.time()-t1:.0f}s")

    # Force torch path for Mamba2 to match HF's torch_forward exactly
    for layer in model.layers.values():
        layer.mamba._force_torch_path = True

    # Distribute across GPUs (naive pipeline)
    n_layers = MODEL_ARGS.n_layers
    layers_per_gpu = (n_layers + NUM_GPUS - 1) // NUM_GPUS
    model.tok_embeddings = model.tok_embeddings.to("cuda:0")

    device_map = {}
    for i, (key, layer) in enumerate(model.layers.items()):
        gpu_id = min(i // layers_per_gpu, NUM_GPUS - 1)
        dev = f"cuda:{gpu_id}"
        layer.to(dev)
        device_map[key] = dev

    last_gpu = f"cuda:{NUM_GPUS - 1}"
    model.norm = model.norm.to(last_gpu)
    model.output = model.output.to(last_gpu)
    model.eval()
    print(f"  Distributed. {gpu_mem()}")

    # --- Hooks ---
    probes = OrderedDict()

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                probes[f"{name}_in"] = inp[0].detach().float().cpu()
            o = out[0] if isinstance(out, tuple) else out
            probes[f"{name}_out"] = o.detach().float().cpu()
        return hook_fn

    hooks = []
    hooks.append(model.tok_embeddings.register_forward_hook(make_hook("emb")))
    for block_id, layer in model.layers.items():
        hooks.append(layer.mamba_norm.register_forward_hook(make_hook(f"b{block_id}_mamba_norm")))
        hooks.append(layer.mamba.register_forward_hook(make_hook(f"b{block_id}_mamba")))
        if layer.has_attn:
            hooks.append(layer.attn_norm.register_forward_hook(make_hook(f"b{block_id}_attn_norm")))
            hooks.append(layer.attention.register_forward_hook(make_hook(f"b{block_id}_attn")))
        hooks.append(layer.ffn_norm.register_forward_hook(make_hook(f"b{block_id}_ffn_norm")))
        hooks.append(layer.moe.register_forward_hook(make_hook(f"b{block_id}_moe")))
    hooks.append(model.norm.register_forward_hook(make_hook("final_norm")))

    # --- Manual forward (handles multi-GPU) ---
    print(f"  Forward pass ...")
    t2 = time.time()
    with torch.no_grad():
        x = model.tok_embeddings(input_ids.to("cuda:0"))
        seq_len = input_ids.shape[1]
        rope = model.rope_cache[:seq_len].to("cuda:0")

        for block_id, layer in model.layers.items():
            dev = device_map[block_id]
            x = x.to(dev)
            r = rope.to(dev)
            x = layer(x, r, None)

        x = x.to(last_gpu)
        x = model.norm(x)
        logits = model.output(x).detach().float().cpu()

    probes["logits"] = logits
    print(f"  Done in {time.time()-t2:.1f}s  logits std={logits.std():.4f}")

    for h in hooks:
        h.remove()

    # Save
    print(f"  Saving {len(probes)} probes to {OUR_PROBES} ...")
    torch.save(probes, OUR_PROBES)

    del model, probes
    free_gpu()


# ===========================================================================
# Phase 3: Compare
# ===========================================================================
def phase3_compare():
    print("\n" + "=" * 72)
    print("PHASE 3: Layer-by-layer comparison")
    print("=" * 72)

    hf_probes = torch.load(HF_PROBES, map_location="cpu", weights_only=False)
    our_probes = torch.load(OUR_PROBES, map_location="cpu", weights_only=False)
    pattern = hf_probes.pop("pattern")
    hf_probes.pop("input_ids", None)

    attn_set = set(MODEL_ARGS.attn_layer_idxs)
    results = []
    failed = []

    def check(label, hf_val, our_val):
        d = (hf_val - our_val).abs().max().item()
        status, abs_d, rel_d = fmt_diff(d, hf_val.abs().max().item())
        results.append((label, status, abs_d, rel_d))
        print(f"  [{status}]  {label:<44s}  abs={abs_d:<12.6f}  rel={rel_d:.4%}")
        if status == "FAIL":
            failed.append(label)

    # Embeddings
    print()
    if "emb_out" in hf_probes and "emb_out" in our_probes:
        check("Embedding output", hf_probes["emb_out"], our_probes["emb_out"])

    # Build flat -> grouped mapping
    flat_map = []
    for block_id in range(MODEL_ARGS.n_layers):
        flat_map.append((block_id, "mamba"))
        if block_id in attn_set:
            flat_map.append((block_id, "attn"))
        flat_map.append((block_id, "moe"))

    print(f"\n  {'Layer':<44s}  {'':>6s}  {'AbsDiff':>12s}  {'RelDiff':>10s}")
    print(f"  {'-'*44}  {'-'*6}  {'-'*12}  {'-'*10}")

    for flat_i, (block_id, sublayer) in enumerate(flat_map):
        lt = pattern[flat_i]
        hf_in = f"flat_{flat_i}_{lt}_in"
        hf_out = f"flat_{flat_i}_{lt}_out"

        if sublayer == "mamba":
            our_res = f"b{block_id}_mamba_norm_in"
            our_sub = f"b{block_id}_mamba_out"
        elif sublayer == "attn":
            our_res = f"b{block_id}_attn_norm_in"
            our_sub = f"b{block_id}_attn_out"
        elif sublayer == "moe":
            our_res = f"b{block_id}_ffn_norm_in"
            our_sub = f"b{block_id}_moe_out"

        lbl = f"flat {flat_i:2d} | block {block_id:2d} {sublayer:>5s}"

        # Compare residual stream INPUT
        if hf_in in hf_probes and our_res in our_probes:
            check(f"{lbl}  input", hf_probes[hf_in], our_probes[our_res])

        # Compare residual stream OUTPUT (input + sublayer_output)
        if hf_out in hf_probes and our_res in our_probes and our_sub in our_probes:
            our_residual = our_probes[our_res] + our_probes[our_sub]
            check(f"{lbl}  output", hf_probes[hf_out], our_residual)

    # Final norm
    print()
    if "final_norm_out" in hf_probes and "final_norm_out" in our_probes:
        check("Final RMSNorm", hf_probes["final_norm_out"], our_probes["final_norm_out"])

    # Logits
    hf_logits = hf_probes["logits"]
    our_logits = our_probes["logits"]
    check("Logits", hf_logits, our_logits)

    top1 = (hf_logits.argmax(-1) == our_logits.argmax(-1)).float().mean().item()
    print(f"\n  Top-1 match: {top1:.1%}")

    hf5 = hf_logits.topk(5, dim=-1).indices
    our5 = our_logits.topk(5, dim=-1).indices
    top5 = sum(
        len(set(hf5[0,i].tolist()) & set(our5[0,i].tolist())) / 5.0
        for i in range(hf_logits.shape[1])
    ) / hf_logits.shape[1]
    print(f"  Top-5 overlap: {top5:.1%}")

    n_ok = sum(1 for _, s, _, _ in results if s in (" OK ", "FINE"))
    n_warn = sum(1 for _, s, _, _ in results if s == "WARN")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")

    print(f"\n{'='*72}")
    print(f"  {n_ok} OK/FINE,  {n_warn} WARN,  {n_fail} FAIL   ({len(results)} checks)")
    if failed:
        print(f"\n  FAILED:")
        for f in failed:
            print(f"    {f}")
    if n_fail == 0 and top1 > 0.95:
        print(f"\n  VERDICT: PASS")
    elif n_fail == 0:
        print(f"\n  VERDICT: CLOSE (bf16 noise)")
    else:
        print(f"\n  VERDICT: FAIL")
    print("=" * 72)
    return n_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=0, help="Run only phase 1/2/3 (0=all)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Nemotron Super 120B parity test | {NUM_GPUS}x {torch.cuda.get_device_name(0)}")
    print(f"seq_len={args.seq_len}, seed={args.seed}")

    input_ids = torch.randint(1, 10000, (1, args.seq_len))
    print(f"tokens[0:10]: {input_ids[0,:10].tolist()}")

    if args.phase in (0, 1):
        phase1_hf(input_ids)

    if args.phase in (0, 2):
        # Re-seed to ensure same input
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        input_ids = torch.randint(1, 10000, (1, args.seq_len))
        phase2_ours(input_ids)

    if args.phase in (0, 3):
        n_fail = phase3_compare()
        sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
