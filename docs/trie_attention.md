# Trie Attention in TorchTitan

Trie attention enables efficient training on tree-structured conversation data (e.g., MCTS rollouts, branching dialogues). Instead of duplicating shared prefixes across training samples, trie attention uses a custom attention mask that allows tokens to attend only to their ancestors in the tree.

## Overview

### The Problem

When training on tree-structured data like MCTS rollouts, the naive approach duplicates shared prefixes:

```
Tree:           Naive packing (wasteful):
    root            Sample 1: root → A → C
   /    \           Sample 2: root → A → D  (root, A duplicated!)
  A      B          Sample 3: root → B → E
 / \    / \         Sample 4: root → B → F  (root, B duplicated!)
C   D  E   F
```

### The Solution: Zero-Redundancy Packing

Trie attention packs all nodes once in DFS order and uses attention masking to enforce the tree structure:

```
Zero-redundancy pack: [root, A, C, D, B, E, F]

Attention mask ensures:
- C can attend to: root, A, C (its ancestors + self)
- D can attend to: root, A, D (its ancestors + self)
- E can attend to: root, B, E (its ancestors + self)
- C cannot attend to: B, D, E, F (not ancestors)
```

## How It Works

### DFS Interval Containment

Each node gets entry time (`tin`) and exit time (`tout`) from DFS traversal. Node `kv` is an ancestor of node `q` iff:

```
tin[kv] <= tin[q] AND tout[q] <= tout[kv]
```

This is the interval containment property - ancestor intervals always contain descendant intervals.

### Attention Mask

The trie attention mask combines:
1. **Ancestor check**: `tin[kv] <= tin[q] AND tout[q] <= tout[kv]`
2. **Causal ordering**: `q_idx >= kv_idx` (for tokens within the same node)

```python
def trie_causal_mask(b, h, q_idx, kv_idx):
    is_ancestor = (tin[b, kv_idx] <= tin[b, q_idx]) & (tout[b, q_idx] <= tout[b, kv_idx])
    is_causal = q_idx >= kv_idx
    return is_ancestor & is_causal
```

### Position IDs

Position IDs are **depth-based** (cumulative tokens from root), not linear sequence position. This ensures correct RoPE for sibling branches - tokens at the same depth get the same positional encoding regardless of their sequence position.

## Usage

### 1. Prepare Tree-Structured Data

Input format (JSONL):
```json
{"messages": [
  {"node_id": 0, "parent_id": null, "role": "system", "content": "You are helpful."},
  {"node_id": 1, "parent_id": 0, "role": "user", "content": "Hello"},
  {"node_id": 2, "parent_id": 1, "role": "assistant", "content": "Hi there!"},
  {"node_id": 3, "parent_id": 1, "role": "assistant", "content": "Hello! How can I help?"}
]}
```

### 2. Preprocess

```bash
python scripts/preprocess_trie_data.py \
    --input_file data/conversations.jsonl \
    --output_dir data/preprocessed_trie \
    --tokenizer_path Qwen/Qwen3-0.6B \
    --seq_len 4096
```

Output columns: `inputs`, `labels`, `tin`, `tout`, `position_ids`

### 3. Train with Trie Attention

Use the `debugmodel_trie` flavor or set `attn_mask_type = "trie_causal"` in your config:

```toml
[model]
name = "qwen3"
flavor = "debugmodel_trie"  # or any flavor with use_flex_attn=True, attn_mask_type="trie_causal"

[training]
dataset = "preprocessed"
dataset_path = "data/preprocessed_trie"
```

## Tree Splitting

When a tree exceeds `seq_len`, it's split into multiple packs:

1. Enumerate all root-to-leaf paths
2. Greedily bin-pack paths to minimize duplication
3. Each pack contains complete paths with zero internal redundancy
4. Shared prefixes may be duplicated **across** packs (unavoidable)

Example:
```
Large tree that doesn't fit in seq_len:

        root
          |
          A
         / \
        B   C
       /|   |\
      ... (many nodes)

Split into:
- Pack 1: [root, A, B, ...descendants of B...]
- Pack 2: [root, A, C, ...descendants of C...]

"root, A" duplicated across packs, but each pack is zero-redundancy internally.
```

## Generating Synthetic Data

For testing, generate synthetic tree conversations:

```bash
python scripts/generate_trie_data.py \
    --output_file data/synthetic_trees.jsonl \
    --num_trees 1000 \
    --min_depth 3 \
    --max_depth 6 \
    --min_branches 2 \
    --max_branches 4
```

## Benchmarking

Compare trie attention vs standard causal:

```bash
python benchmarks/trie_attention_benchmark.py \
    --batch_size 4 \
    --seq_len 2048 \
    --prefix_ratio 0.5
```

## Implementation Details

### Files

- `torchtitan/models/attention.py` - `get_trie_causal_mask_mod()` function
- `torchtitan/models/qwen3/model/model.py` - `"trie_causal"` case in `get_attention_masks()`
- `torchtitan/models/qwen3/__init__.py` - `"debugmodel_trie"` model flavor
- `torchtitan/hf_datasets/preprocessed.py` - Dataloader handling for tin/tout/position_ids
- `scripts/preprocess_trie_data.py` - Preprocessing with tree splitting
- `scripts/generate_trie_data.py` - Synthetic data generation
- `benchmarks/trie_attention_benchmark.py` - Performance benchmarking

### Data Format

The preprocessed parquet contains:
- `inputs`: Token IDs in DFS order
- `labels`: Labels with -100 for non-assistant tokens
- `tin`: DFS entry time per token (same for all tokens in a node)
- `tout`: DFS exit time per token (same for all tokens in a node)
- `position_ids`: Depth-based positions (cumulative tokens from root)

### FlexAttention

Trie attention uses PyTorch's FlexAttention with a custom mask modifier. The mask is compiled and cached for efficiency. Set `use_flex_attn=True` in model args to enable.
