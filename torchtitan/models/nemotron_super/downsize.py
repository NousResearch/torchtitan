"""
Downsize NemotronH to 3 layers (M E, M * E, M E) and push to HF.

Usage:
  python -m torchtitan.models.nemotron_super.downsize \
    --input /home/shared/torchtitan/assets/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 \
    --output nns3_downsized
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/shared/torchtitan/assets/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16")
    parser.add_argument("--output", default="NousResearch/nns3_downsized")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of our layers (each = M [*] E)")
    parser.add_argument("--local-dir", default="/tmp/nns3_downsized")
    args = parser.parse_args()

    input_path = Path(args.input)
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = input_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Original pattern: 88 flat layers
    orig_pattern = config["hybrid_override_pattern"]
    print(f"Original pattern ({len(orig_pattern)} flat layers): {orig_pattern[:40]}...")

    # Parse into blocks: each block = M [*] E
    blocks = []
    i = 0
    while i < len(orig_pattern):
        block = {"start": i}
        assert orig_pattern[i] == "M"
        i += 1
        if i < len(orig_pattern) and orig_pattern[i] == "*":
            block["has_attn"] = True
            i += 1
        else:
            block["has_attn"] = False
        assert orig_pattern[i] == "E"
        block["end"] = i
        i += 1
        blocks.append(block)

    print(f"Parsed {len(blocks)} blocks, {sum(b['has_attn'] for b in blocks)} with attention")

    # Pick which blocks to keep: first N, ensuring at least one has attention
    n_layers = args.n_layers
    keep = list(range(n_layers))
    # Make sure we include at least one attention block
    has_attn = any(blocks[i]["has_attn"] for i in keep)
    if not has_attn:
        # Swap last kept block for first attention block
        for j, b in enumerate(blocks):
            if b["has_attn"]:
                keep[-1] = j
                break

    print(f"Keeping blocks: {keep}")
    for idx in keep:
        b = blocks[idx]
        flat_range = list(range(b["start"], b["end"] + 1))
        print(f"  Block {idx}: flat layers {flat_range} {'(+attn)' if b['has_attn'] else ''}")

    # Build flat layer mapping: old_flat -> new_flat
    flat_map = {}  # old_flat_idx -> new_flat_idx
    new_flat = 0
    new_pattern = ""
    for new_block_idx, old_block_idx in enumerate(keep):
        b = blocks[old_block_idx]
        for old_flat in range(b["start"], b["end"] + 1):
            flat_map[old_flat] = new_flat
            new_pattern += orig_pattern[old_flat]
            new_flat += 1

    print(f"New pattern ({len(new_pattern)} flat layers): {new_pattern}")
    print(f"Flat layer mapping: {flat_map}")

    # Load index
    index_path = input_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Categorize keys
    keep_keys = {}  # old_key -> new_key
    for old_key in weight_map:
        # Global keys
        if not old_key.startswith("backbone.layers.") and not old_key.startswith("mtp."):
            keep_keys[old_key] = old_key
            continue

        # Skip MTP
        if old_key.startswith("mtp."):
            continue

        # Layer keys
        m = re.match(r"backbone\.layers\.(\d+)\.(.*)", old_key)
        if not m:
            continue
        old_flat_idx = int(m.group(1))
        suffix = m.group(2)

        if old_flat_idx in flat_map:
            new_flat_idx = flat_map[old_flat_idx]
            new_key = f"backbone.layers.{new_flat_idx}.{suffix}"
            keep_keys[old_key] = new_key

    print(f"\nKeeping {len(keep_keys)} keys (from {len(weight_map)} total)")

    # Load and remap weights
    files_needed = set(weight_map[k] for k in keep_keys)
    print(f"Reading from {len(files_needed)} safetensor files...")

    new_sd = {}
    for fname in sorted(files_needed):
        fpath = input_path / fname
        print(f"  {fname}...")
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in keep_keys:
                    new_sd[keep_keys[k]] = f.get_tensor(k)

    print(f"Collected {len(new_sd)} tensors")

    # Update config
    new_config = dict(config)
    new_config["num_hidden_layers"] = len(new_pattern)
    new_config["hybrid_override_pattern"] = new_pattern
    new_config["num_nextn_predict_layers"] = 0  # no MTP
    del_keys = ["mtp_hybrid_override_pattern"]
    for dk in del_keys:
        new_config.pop(dk, None)

    # Save
    print(f"\nSaving to {local_dir}...")

    # Save config
    with open(local_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    # Save weights as single safetensors file
    save_file(new_sd, str(local_dir / "model.safetensors"))

    # Save index (single file, no sharding needed)
    new_index = {
        "metadata": {
            "total_parameters": sum(t.numel() for t in new_sd.values()),
            "total_size": sum(t.numel() * t.element_size() for t in new_sd.values()),
        },
        "weight_map": {k: "model.safetensors" for k in new_sd},
    }
    with open(local_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy tokenizer files if present
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "tokenizer.model", "vocab.json", "merges.txt"]:
        src = input_path / tok_file
        if src.exists():
            import shutil
            shutil.copy2(src, local_dir / tok_file)
            print(f"  Copied {tok_file}")

    total_params = sum(t.numel() for t in new_sd.values())
    print(f"\nDowsized model: {total_params:,} params ({total_params/1e9:.2f}B)")
    print(f"Pattern: {new_pattern}")

    # Push to HF
    print(f"\nPushing to {args.output}...")
    api = HfApi()
    api.create_repo(args.output, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=args.output)
    print("Done!")


if __name__ == "__main__":
    main()
