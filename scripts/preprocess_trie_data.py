# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Preprocessing script for trie-structured conversation data.

This script converts tree-structured conversation data into a format suitable
for training with trie attention. Key features:

1. **Zero-redundancy packing**: Nodes are arranged in DFS order, trie attention
   handles masking so each node appears exactly once per sample.

2. **Depth-based position IDs**: Position IDs are based on depth from root,
   not linear sequence position. This ensures correct RoPE for sibling branches.

3. **Tree splitting**: When a tree exceeds seq_len, we split by grouping
   root-to-leaf paths. Each split contains complete paths with zero redundancy.
   Shared prefixes may be duplicated across splits (unavoidable).

Input format (JSONL):
{"messages": [{"node_id": 0, "parent_id": null, "content": "...", "role": "system"}, ...]}

Output format (Arrow/Parquet):
- inputs: tokenized sequence (DFS order within each split)
- labels: labels with appropriate masking
- tin: DFS entry time per token
- tout: DFS exit time per token
- position_ids: depth-based positions (distance from root in tokens)

Usage:
    python scripts/preprocess_trie_data.py \
        --input_file data/conversations.jsonl \
        --output_dir data/preprocessed \
        --tokenizer_path Qwen/Qwen2.5-0.5B \
        --seq_len 4096
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


@dataclass
class TokenizedNode:
    """A node with its tokenized content and metadata."""
    node_id: int
    parent_id: int | None
    tokens: list[int]
    role: str
    depth: int  # Depth from root (in nodes, not tokens)
    children: list[int]


def build_tree_from_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[dict[int, TokenizedNode], int | None]:
    """Build tree of tokenized nodes from messages.

    Returns:
        nodes: Dict mapping node_id to TokenizedNode
        root_id: Root node ID
    """
    # First pass: build parent->children mapping and find root
    children_map: dict[int, list[int]] = defaultdict(list)
    parent_map: dict[int, int | None] = {}
    root_id = None

    for msg in messages:
        node_id = msg["node_id"]
        parent_id = msg.get("parent_id")
        parent_map[node_id] = parent_id

        if parent_id is None:
            root_id = node_id
        else:
            children_map[parent_id].append(node_id)

    if root_id is None:
        return {}, None

    # Second pass: compute depths via BFS from root
    depths: dict[int, int] = {root_id: 0}
    queue = [root_id]
    while queue:
        node_id = queue.pop(0)
        for child_id in children_map[node_id]:
            depths[child_id] = depths[node_id] + 1
            queue.append(child_id)

    # Third pass: create TokenizedNode objects
    nodes: dict[int, TokenizedNode] = {}
    for msg in messages:
        node_id = msg["node_id"]
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        nodes[node_id] = TokenizedNode(
            node_id=node_id,
            parent_id=parent_map[node_id],
            tokens=tokens,
            role=msg.get("role", "user"),
            depth=depths[node_id],
            children=children_map.get(node_id, []),
        )

    return nodes, root_id


def get_all_leaves(nodes: dict[int, TokenizedNode]) -> list[int]:
    """Get all leaf node IDs."""
    return [nid for nid, node in nodes.items() if not node.children]


def get_path_to_root(nodes: dict[int, TokenizedNode], leaf_id: int) -> list[int]:
    """Get path from leaf to root (returns root-to-leaf order)."""
    path = []
    current = leaf_id
    while current is not None:
        path.append(current)
        current = nodes[current].parent_id
    return list(reversed(path))


def get_path_token_count(nodes: dict[int, TokenizedNode], path: list[int]) -> int:
    """Count total tokens in a path."""
    return sum(len(nodes[nid].tokens) for nid in path)


def compute_subtree_token_counts(
    nodes: dict[int, TokenizedNode],
    root_id: int
) -> dict[int, int]:
    """Compute total token count for subtree rooted at each node."""
    counts: dict[int, int] = {}

    def dfs(node_id: int) -> int:
        node = nodes[node_id]
        total = len(node.tokens)
        for child_id in node.children:
            total += dfs(child_id)
        counts[node_id] = total
        return total

    dfs(root_id)
    return counts


def split_tree_into_packs(
    nodes: dict[int, TokenizedNode],
    root_id: int,
    seq_len: int,
) -> list[set[int]]:
    """Split tree into packs, each fitting within seq_len.

    Strategy: Group root-to-leaf paths such that each group's total
    unique nodes fit within seq_len. Uses greedy bin-packing.

    Returns:
        List of sets, each set contains node_ids for one pack.
    """
    subtree_counts = compute_subtree_token_counts(nodes, root_id)
    total_tokens = subtree_counts[root_id]

    # If whole tree fits, return single pack with all nodes
    if total_tokens <= seq_len:
        return [set(nodes.keys())]

    # Get all root-to-leaf paths
    leaves = get_all_leaves(nodes)
    paths = [get_path_to_root(nodes, leaf_id) for leaf_id in leaves]

    # Sort paths by length (tokens) descending - pack longest first
    paths.sort(key=lambda p: get_path_token_count(nodes, p), reverse=True)

    packs: list[set[int]] = []

    for path in paths:
        path_set = set(path)
        path_tokens = get_path_token_count(nodes, path)

        # Try to fit into existing pack
        best_pack_idx = -1
        best_marginal_cost = float('inf')

        for i, pack in enumerate(packs):
            # Marginal cost = tokens we'd add (excluding already-present nodes)
            new_nodes = path_set - pack
            marginal_tokens = sum(len(nodes[nid].tokens) for nid in new_nodes)
            current_pack_tokens = sum(len(nodes[nid].tokens) for nid in pack)

            if current_pack_tokens + marginal_tokens <= seq_len:
                if marginal_tokens < best_marginal_cost:
                    best_marginal_cost = marginal_tokens
                    best_pack_idx = i

        if best_pack_idx >= 0:
            packs[best_pack_idx].update(path_set)
        else:
            # Need new pack - check if path alone fits
            if path_tokens <= seq_len:
                packs.append(path_set)
            else:
                # Path itself is too long - need to truncate
                # Take as many nodes as possible from root
                truncated = set()
                running_tokens = 0
                for nid in path:
                    node_tokens = len(nodes[nid].tokens)
                    if running_tokens + node_tokens <= seq_len:
                        truncated.add(nid)
                        running_tokens += node_tokens
                    else:
                        break
                if truncated:
                    packs.append(truncated)

    return packs


def pack_to_training_sample(
    nodes: dict[int, TokenizedNode],
    pack_node_ids: set[int],
    seq_len: int,
    pad_token_id: int,
) -> dict[str, list[int]]:
    """Convert a pack of nodes into a training sample.

    Performs DFS traversal to get proper tin/tout and ordering.
    Position IDs are based on cumulative token depth from root.
    """
    # Find root of this pack (node with no parent in pack)
    pack_root = None
    for nid in pack_node_ids:
        parent = nodes[nid].parent_id
        if parent is None or parent not in pack_node_ids:
            pack_root = nid
            break

    if pack_root is None:
        raise ValueError("Could not find pack root")

    # DFS traversal within pack
    all_tokens: list[int] = []
    all_labels: list[int] = []
    all_tin: list[int] = []
    all_tout: list[int] = []
    all_position_ids: list[int] = []

    tin_map: dict[int, int] = {}
    tout_map: dict[int, int] = {}
    time_counter = [0]

    # Track token-level depth (cumulative tokens from root)
    def get_token_depth(node_id: int, nodes: dict[int, TokenizedNode], pack_node_ids: set[int]) -> int:
        """Get cumulative token count from pack root to this node (exclusive)."""
        depth = 0
        current = node_id
        path = []
        while current is not None and current in pack_node_ids:
            path.append(current)
            current = nodes[current].parent_id
        # path is leaf-to-root, we want root-to-node (exclusive of node itself)
        path = list(reversed(path))
        for nid in path[:-1]:  # Exclude the node itself
            depth += len(nodes[nid].tokens)
        return depth

    def dfs(node_id: int) -> None:
        node = nodes[node_id]
        tin_map[node_id] = time_counter[0]
        time_counter[0] += 1

        # Get base position (token depth from root)
        base_pos = get_token_depth(node_id, nodes, pack_node_ids)

        # Add tokens for this node
        for i, tok in enumerate(node.tokens):
            all_tokens.append(tok)
            all_tin.append(tin_map[node_id])
            # tout will be filled after DFS completes for this node
            all_tout.append(-1)  # Placeholder
            all_position_ids.append(base_pos + i)

            # Labels: mask non-assistant tokens
            if node.role == "assistant":
                all_labels.append(tok)
            else:
                all_labels.append(-100)

        # Recurse into children that are in pack
        for child_id in node.children:
            if child_id in pack_node_ids:
                dfs(child_id)

        tout_map[node_id] = time_counter[0]
        time_counter[0] += 1

    dfs(pack_root)

    # Fill in tout values
    token_idx = 0
    def fill_tout(node_id: int) -> None:
        nonlocal token_idx
        node = nodes[node_id]
        for _ in node.tokens:
            all_tout[token_idx] = tout_map[node_id]
            token_idx += 1
        for child_id in node.children:
            if child_id in pack_node_ids:
                fill_tout(child_id)

    fill_tout(pack_root)

    # Pad to seq_len
    actual_len = len(all_tokens)
    pad_len = seq_len - actual_len

    if pad_len < 0:
        raise ValueError(f"Pack has {actual_len} tokens, exceeds seq_len {seq_len}")

    # Padding values: use large tin and small tout so no valid attention
    max_time = time_counter[0] + 1
    all_tokens.extend([pad_token_id] * pad_len)
    all_labels.extend([-100] * pad_len)
    all_tin.extend([max_time] * pad_len)
    all_tout.extend([0] * pad_len)  # tout < tin means no ancestor relationship
    all_position_ids.extend([0] * pad_len)  # Doesn't matter for padded tokens

    return {
        "inputs": all_tokens,
        "labels": all_labels,
        "tin": all_tin,
        "tout": all_tout,
        "position_ids": all_position_ids,
    }


def process_conversation(
    conversation: dict[str, Any],
    tokenizer: Any,
    seq_len: int,
) -> list[dict[str, list[int]]]:
    """Process a conversation tree into training samples.

    Returns list of samples (may be multiple if tree was split).
    """
    messages = conversation.get("messages", [])
    if not messages:
        return []

    nodes, root_id = build_tree_from_messages(messages, tokenizer)
    if root_id is None:
        return []

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # Split tree into packs
    packs = split_tree_into_packs(nodes, root_id, seq_len)

    # Convert each pack to training sample
    samples = []
    for pack in packs:
        try:
            sample = pack_to_training_sample(nodes, pack, seq_len, pad_token_id)
            samples.append(sample)
        except ValueError as e:
            print(f"Warning: skipping pack: {e}")
            continue

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess trie-structured conversation data"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSON/JSONL file with conversations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to HuggingFace tokenizer",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Target sequence length",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "arrow"],
        default="parquet",
        help="Output format",
    )
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load input data
    print(f"Loading data from {args.input_file}...")
    input_path = Path(args.input_file)

    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            conversations = [json.loads(line) for line in f]
    else:
        with open(input_path) as f:
            data = json.load(f)
            conversations = data if isinstance(data, list) else [data]

    print(f"Loaded {len(conversations)} conversations")

    # Process conversations
    all_samples: list[dict[str, list[int]]] = []
    trees_processed = 0
    trees_skipped = 0

    for i, conv in enumerate(conversations):
        samples = process_conversation(conv, tokenizer, args.seq_len)
        if samples:
            all_samples.extend(samples)
            trees_processed += 1
        else:
            trees_skipped += 1

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(conversations)} conversations, {len(all_samples)} samples so far...")

    print(f"Trees processed: {trees_processed}")
    print(f"Trees skipped: {trees_skipped}")
    print(f"Total samples: {len(all_samples)}")

    if not all_samples:
        print("No valid samples to save")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to Arrow table
    table = pa.table(
        {
            "inputs": [s["inputs"] for s in all_samples],
            "labels": [s["labels"] for s in all_samples],
            "tin": [s["tin"] for s in all_samples],
            "tout": [s["tout"] for s in all_samples],
            "position_ids": [s["position_ids"] for s in all_samples],
        }
    )

    # Save
    if args.format == "parquet":
        output_file = output_dir / "train.parquet"
        pq.write_table(table, output_file)
    else:
        output_file = output_dir / "train.arrow"
        with pa.OSFile(str(output_file), "wb") as f:
            writer = pa.ipc.RecordBatchFileWriter(f, table.schema)
            writer.write_table(table)
            writer.close()

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
