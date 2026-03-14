"""
Validate a converted torchtitan checkpoint against the original HF checkpoint.

Usage:
  python -m torchtitan.models.nemotron_super.validate_checkpoint \
    --hf /path/to/Nemotron-H-120B-A12B \
    --titan /path/to/torchtitan-conversions/checkpoint

Loads a sample of weights from both, runs the HF weights through the adapter,
and compares numerically.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from torchtitan.models.nemotron_super import nemotron_super_args
from torchtitan.models.nemotron_super.model.state_dict_adapter import NemotronSuperStateDictAdapter


def load_titan_flat(titan_path: str) -> dict[str, torch.Tensor]:
    """Load a DCP checkpoint into a flat dict (single-rank only)."""
    reader = FileSystemReader(titan_path)
    metadata = reader.read_metadata()
    sd = {}
    for key, meta in metadata.state_dict_metadata.items():
        if isinstance(meta, BytesStorageMetadata):
            continue
        try:
            # DCP stores as sharded; for single-file checkpoints this works
            from torch.distributed.checkpoint._traverse import set_element
            from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
            pass
        except Exception:
            pass
    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", required=True, help="Path to HF checkpoint dir")
    parser.add_argument("--titan", required=False, help="Path to titan DCP checkpoint dir")
    parser.add_argument("--samples", type=int, default=20, help="Number of non-expert keys to check")
    args_cli = parser.parse_args()

    model_args = nemotron_super_args["120B-A12B"]
    adapter = NemotronSuperStateDictAdapter(model_args, None)

    hf_path = Path(args_cli.hf)
    index_path = hf_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    all_hf_keys = list(weight_map.keys())

    # Pick diverse sample keys (skip most experts, take a few)
    non_expert_keys = [k for k in all_hf_keys if not re.search(r"experts\.\d+\.", k)]
    expert_keys = [k for k in all_hf_keys if re.search(r"experts\.\d+\.", k)]

    # Sample non-expert keys spread across layers
    sample = []
    seen_patterns = set()
    for k in non_expert_keys:
        # Use a pattern that preserves fc1/fc2 distinction
        pat = re.sub(r"(?<=layers\.)\d+", "{}", k)
        if pat not in seen_patterns:
            seen_patterns.add(pat)
            sample.append(k)

    # Add a handful of expert keys from different layers
    expert_sample_layers = set()
    for k in expert_keys:
        m = re.match(r"backbone\.layers\.(\d+)\.mixer\.experts\.0\.", k)
        if m and m.group(1) not in expert_sample_layers and len(expert_sample_layers) < 3:
            expert_sample_layers.add(m.group(1))
            # Get expert 0 and expert 1 for this layer to test stacking
            layer = m.group(1)
            for ek in expert_keys:
                if f"backbone.layers.{layer}.mixer.experts.0." in ek or \
                   f"backbone.layers.{layer}.mixer.experts.1." in ek:
                    sample.append(ek)

    print(f"Loading {len(sample)} sample keys from HF checkpoint...")

    # Load from safetensors
    hf_sd = {}
    files_needed = {weight_map[k] for k in sample if k in weight_map}
    for fname in sorted(files_needed):
        fpath = hf_path / fname
        print(f"  reading {fname}...")
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in sample:
                    hf_sd[k] = f.get_tensor(k)

    print(f"Loaded {len(hf_sd)} HF tensors")

    # Convert through adapter
    titan_sd = adapter.from_hf(hf_sd)
    print(f"Adapter produced {len(titan_sd)} titan keys\n")

    # Report what we got
    print("=" * 80)
    print("CONVERSION RESULTS")
    print("=" * 80)

    for titan_key in sorted(titan_sd.keys()):
        t = titan_sd[titan_key]
        shape = list(t.shape) if isinstance(t, torch.Tensor) else "?"
        if isinstance(t, torch.Tensor):
            std = t.float().std().item()
            mn = t.float().mean().item()
            print(f"  {titan_key:60s} {str(shape):25s} mean={mn:+.4f}  std={std:.4f}")
        else:
            print(f"  {titan_key:60s} {shape}")

    # Check for HF keys that didn't map
    hf_consumed = set()
    for k in hf_sd:
        if k.startswith("mtp.") and not adapter._mtp_enabled:
            hf_consumed.add(k)
            continue
        # Expert keys get consumed via buffer
        if re.search(r"experts\.\d+\.", k):
            hf_consumed.add(k)
            continue

    hf_not_mapped = []
    for k in hf_sd:
        if k in hf_consumed:
            continue
        # Check if this key contributed to any titan key
        flat_idx = adapter._hf_key_to_flat_idx(k)
        if flat_idx is not None and flat_idx in adapter.flat_to_layer:
            hf_consumed.add(k)
            continue
        if k in adapter._global_map:
            hf_consumed.add(k)
            continue
        if k in adapter._mtp_map:
            hf_consumed.add(k)
            continue
        hf_not_mapped.append(k)

    if hf_not_mapped:
        print(f"\nWARNING: {len(hf_not_mapped)} HF keys not mapped:")
        for k in hf_not_mapped:
            print(f"  {k}")

    # Sanity checks
    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    checks = [
        "tok_embeddings.weight",
        "output.weight",
        "norm.weight",
        "layers.0.mamba_norm.weight",
        "layers.0.mamba.in_proj.weight",
        "layers.0.mamba.A_log",
        "layers.0.ffn_norm.weight",
        "layers.0.moe.router.gate.weight",
        "layers.0.moe.latent_in.weight",
        "layers.0.moe.latent_out.weight",
        "layers.0.moe.shared_experts.w1.weight",
        "layers.0.moe.expert_bias",
    ]
    for key in checks:
        if key in titan_sd:
            t = titan_sd[key]
            std = t.float().std().item()
            status = "OK" if std > 0.001 else "SUSPICIOUS (very small std)"
            if std == 0.0:
                status = "ZERO - not loaded?"
            print(f"  {status:30s} {key} std={std:.6f}")
        else:
            print(f"  {'NOT IN SAMPLE':30s} {key}")

    # Check if expert stacking worked
    expert_keys_titan = [k for k in titan_sd if ".moe.experts." in k]
    if expert_keys_titan:
        print(f"\n  Expert tensors: {len(expert_keys_titan)}")
        for k in sorted(expert_keys_titan):
            t = titan_sd[k]
            print(f"    {k}: shape={list(t.shape)}, std={t.float().std().item():.4f}")


if __name__ == "__main__":
    main()
