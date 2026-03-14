"""
Diagnostic: check if state_dict_adapter produces keys/shapes that match the model.
Run on a single GPU with the HF checkpoint path.

Usage: python -m torchtitan.models.nemotron_super.check_loading /path/to/hf/checkpoint
"""
import sys
import json
import re
import torch
from pathlib import Path
from safetensors import safe_open

from torchtitan.models.nemotron_super import NemotronSuperModel, NemotronSuperModelArgs, nemotron_super_args
from torchtitan.models.nemotron_super.model.state_dict_adapter import NemotronSuperStateDictAdapter


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None

    args = nemotron_super_args["120B-A12B"]
    adapter = NemotronSuperStateDictAdapter(args, None)

    # Build model on meta device to get expected shapes
    with torch.device("meta"):
        model = NemotronSuperModel(args)

    model_sd = {name: p.shape for name, p in model.named_parameters()}
    model_bufs = {name: b.shape for name, b in model.named_buffers()}
    model_all = {**model_sd, **model_bufs}

    print(f"Model expects {len(model_sd)} params + {len(model_bufs)} buffers")

    if ckpt_path is None:
        print("No checkpoint path given, skipping shape check against real weights")
        print("\nModel param names:")
        for name, shape in sorted(model_sd.items()):
            print(f"  {name}: {list(shape)}")
        return

    # Load a few real tensors from checkpoint to verify shapes
    index_path = Path(ckpt_path) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Pick a sample of keys to check (one per pattern)
    sample_keys = set()
    seen_patterns = set()
    for k in weight_map:
        abstract = re.sub(r'\d+', '{}', k)
        if abstract not in seen_patterns:
            seen_patterns.add(abstract)
            sample_keys.add(k)

    # Load sample tensors
    print(f"\nLoading {len(sample_keys)} sample keys from checkpoint...")
    sample_sd = {}
    files_needed = set(weight_map[k] for k in sample_keys)
    for fname in files_needed:
        fpath = Path(ckpt_path) / fname
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in sample_keys:
                    sample_sd[k] = f.get_tensor(k)

    print(f"Loaded {len(sample_sd)} tensors")

    # Run through adapter
    titan_sd = adapter.from_hf(sample_sd)
    print(f"Adapter produced {len(titan_sd)} titan keys")

    # Check each titan key against model
    errors = []
    for titan_key, tensor in sorted(titan_sd.items()):
        if titan_key in model_all:
            expected = model_all[titan_key]
            actual = tensor.shape
            if expected != actual:
                errors.append(f"SHAPE MISMATCH: {titan_key}: model={list(expected)}, ckpt={list(actual)}")
            else:
                print(f"  OK  {titan_key}: {list(actual)}")
        else:
            errors.append(f"EXTRA KEY (not in model): {titan_key}")

    # Check for model keys not covered
    adapter_titan_keys = set(titan_sd.keys())
    for name in sorted(model_sd.keys()):
        abstract = re.sub(r'layers\.\d+', 'layers.{}', name)
        # Check if any adapter key matches this pattern
        found = name in adapter_titan_keys
        if not found:
            # Check if it's a pattern that would be covered by other layers
            for ak in adapter_titan_keys:
                if re.sub(r'layers\.\d+', 'layers.{}', ak) == abstract:
                    found = True
                    break
        if not found and "rope_cache" not in name:
            errors.append(f"MISSING (not loaded from ckpt): {name}")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll checks passed!")


if __name__ == "__main__":
    main()
