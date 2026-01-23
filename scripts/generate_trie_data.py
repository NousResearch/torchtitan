# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate synthetic tree-structured conversation data for testing trie attention.

Creates conversation trees with:
- Configurable branching factor and depth
- Realistic conversation patterns (system -> user -> assistant cycles)
- Multiple response branches simulating MCTS-style exploration

Usage:
    python scripts/generate_trie_data.py \
        --output_file data/synthetic_trees.jsonl \
        --num_trees 200 \
        --min_depth 3 \
        --max_depth 6 \
        --min_branches 2 \
        --max_branches 4
"""

import argparse
import json
import random
from pathlib import Path


# Sample content templates
SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are an expert mathematician. Show your work step by step.",
    "You are a coding assistant. Write clean, well-documented code.",
    "You are a creative writing assistant. Be imaginative and engaging.",
    "You are a science tutor. Explain concepts clearly with examples.",
]

USER_QUESTIONS = [
    "Can you explain how {topic} works?",
    "What is the best approach to solve {problem}?",
    "Help me understand {concept} better.",
    "Write a {thing} that does {action}.",
    "What are the key differences between {a} and {b}?",
    "How would you implement {feature}?",
    "Can you review this and suggest improvements?",
    "What are the pros and cons of {approach}?",
]

TOPICS = [
    "neural networks", "gradient descent", "transformers", "attention mechanisms",
    "backpropagation", "optimization", "regularization", "batch normalization",
    "dropout", "learning rate scheduling", "data augmentation", "transfer learning",
    "fine-tuning", "prompt engineering", "RLHF", "DPO", "PPO", "GRPO",
]

PROBLEMS = [
    "matrix multiplication", "sorting algorithms", "graph traversal",
    "dynamic programming", "binary search", "tree balancing",
    "hash collisions", "memory management", "concurrency",
]

CONCEPTS = [
    "recursion", "memoization", "time complexity", "space complexity",
    "big O notation", "amortized analysis", "cache locality",
]

ASSISTANT_RESPONSES = [
    "Let me break this down step by step. First, {step1}. Then, {step2}. Finally, {step3}.",
    "Great question! The key insight here is {insight}. This means {implication}.",
    "There are several approaches to consider: 1) {approach1}, 2) {approach2}, 3) {approach3}.",
    "Here's how I would think about this: {thought}. The main consideration is {consideration}.",
    "Based on my understanding, {explanation}. An important note: {note}.",
]

STEPS = [
    "we need to understand the basic principles",
    "we apply the core algorithm",
    "we optimize for performance",
    "we handle edge cases",
    "we validate the results",
    "we consider alternative approaches",
]

INSIGHTS = [
    "the problem has a recursive structure",
    "we can use dynamic programming here",
    "there's a mathematical pattern to exploit",
    "the naive approach has exponential complexity",
    "caching intermediate results helps significantly",
]

FOLLOWUPS = [
    "Could you elaborate on {part}?",
    "What about the case where {condition}?",
    "How does this compare to {alternative}?",
    "Can you show a concrete example?",
    "What are the potential pitfalls?",
    "How would this scale with larger inputs?",
]


def generate_content(role: str, depth: int, branch_idx: int) -> str:
    """Generate realistic content based on role and position in tree."""
    if role == "system":
        return random.choice(SYSTEM_PROMPTS)
    elif role == "user":
        if depth == 1:  # First user message
            template = random.choice(USER_QUESTIONS)
            return template.format(
                topic=random.choice(TOPICS),
                problem=random.choice(PROBLEMS),
                concept=random.choice(CONCEPTS),
                thing="function",
                action="process data efficiently",
                a=random.choice(TOPICS),
                b=random.choice(TOPICS),
                feature="caching",
                approach=random.choice(TOPICS),
            )
        else:  # Follow-up question
            template = random.choice(FOLLOWUPS)
            return template.format(
                part="that last point",
                condition="the input is very large",
                alternative="the standard approach",
            )
    else:  # assistant
        template = random.choice(ASSISTANT_RESPONSES)
        steps = random.sample(STEPS, 3)
        return template.format(
            step1=steps[0],
            step2=steps[1],
            step3=steps[2],
            insight=random.choice(INSIGHTS),
            implication="we should structure our solution accordingly",
            approach1="use a greedy algorithm",
            approach2="apply divide and conquer",
            approach3="try dynamic programming",
            thought="this is fundamentally about optimization",
            consideration="balancing time and space complexity",
            explanation="the key is to identify the subproblems",
            note="edge cases require special handling",
        )


def generate_tree(
    min_depth: int,
    max_depth: int,
    min_branches: int,
    max_branches: int,
) -> list[dict]:
    """Generate a single conversation tree.

    Structure follows: system -> (user -> assistant)* pattern
    Branching happens at assistant responses (multiple possible responses).
    """
    messages = []
    node_counter = [0]

    def next_id():
        nid = node_counter[0]
        node_counter[0] += 1
        return nid

    # Root is always system prompt
    root_id = next_id()
    messages.append({
        "node_id": root_id,
        "parent_id": None,
        "role": "system",
        "content": generate_content("system", 0, 0),
    })

    def add_conversation_turn(parent_id: int, depth: int, branch_idx: int):
        """Add a user-assistant turn, potentially with branching responses."""
        if depth > max_depth:
            return

        # Add user message
        user_id = next_id()
        messages.append({
            "node_id": user_id,
            "parent_id": parent_id,
            "role": "user",
            "content": generate_content("user", depth, branch_idx),
        })

        # Decide how many assistant response branches
        if depth < min_depth:
            # Force at least some branching early
            num_branches = random.randint(min_branches, max_branches)
        else:
            # Probabilistically branch or terminate
            if random.random() < 0.3:  # 30% chance to terminate
                num_branches = 1
            else:
                num_branches = random.randint(1, max_branches)

        # Add assistant responses (branches)
        for b in range(num_branches):
            assistant_id = next_id()
            messages.append({
                "node_id": assistant_id,
                "parent_id": user_id,
                "role": "assistant",
                "content": generate_content("assistant", depth, b),
            })

            # Maybe continue the conversation
            if depth < max_depth and random.random() < 0.7:
                add_conversation_turn(assistant_id, depth + 1, b)

    # Start conversation from system prompt
    add_conversation_turn(root_id, 1, 0)

    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic tree-structured conversation data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/synthetic_trees.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num_trees",
        type=int,
        default=200,
        help="Number of conversation trees to generate",
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=3,
        help="Minimum conversation depth (user-assistant turns)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="Maximum conversation depth",
    )
    parser.add_argument(
        "--min_branches",
        type=int,
        default=2,
        help="Minimum branching factor for responses",
    )
    parser.add_argument(
        "--max_branches",
        type=int,
        default=4,
        help="Maximum branching factor for responses",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_trees} conversation trees...")
    print(f"Depth range: {args.min_depth}-{args.max_depth}")
    print(f"Branch range: {args.min_branches}-{args.max_branches}")

    total_nodes = 0
    total_leaves = 0

    with open(output_path, "w") as f:
        for i in range(args.num_trees):
            messages = generate_tree(
                args.min_depth,
                args.max_depth,
                args.min_branches,
                args.max_branches,
            )

            # Count stats
            total_nodes += len(messages)
            node_ids = {m["node_id"] for m in messages}
            parent_ids = {m["parent_id"] for m in messages if m["parent_id"] is not None}
            leaves = node_ids - parent_ids
            total_leaves += len(leaves)

            conversation = {"messages": messages}
            f.write(json.dumps(conversation) + "\n")

            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{args.num_trees} trees...")

    print(f"\nDone! Saved to {output_path}")
    print(f"Total trees: {args.num_trees}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total leaves (root-to-leaf paths): {total_leaves}")
    print(f"Average nodes per tree: {total_nodes / args.num_trees:.1f}")
    print(f"Average leaves per tree: {total_leaves / args.num_trees:.1f}")


if __name__ == "__main__":
    main()
