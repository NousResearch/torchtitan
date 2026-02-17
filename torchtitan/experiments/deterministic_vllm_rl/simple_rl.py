# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple RL training loop with GRPO-style advantage estimation.

This demonstrates:
1. Loading a model in TorchTitan format for training
2. Converting weights to vLLM format for fast rollouts
3. Generating samples using vLLM
4. Computing rewards (trivial/random for now)
5. Computing advantages using GRPO-style group ranking
6. Performing a policy gradient update on TorchTitan model
7. Optional real dataset support (GSM8K math dataset)
"""

import os
import re

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer

from vllm import LLM, SamplingParams

# Try to import and initialize batch invariance (API varies by vLLM version)
try:
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    # Newer vLLM versions require attention_backend argument
    try:
        init_batch_invariance()
    except TypeError:
        # Try with default attention backend for newer vLLM
        try:
            from vllm.v1.attention.backends.registry import get_attn_backend
            backend = get_attn_backend()
            init_batch_invariance(backend)
        except ImportError:
            # Fall back to environment variable approach
            import os
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            print("Using VLLM_BATCH_INVARIANT=1 environment variable for batch invariance")
except ImportError:
    # Batch invariance not available in this vLLM version
    import os
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    print("Batch invariance module not found, using environment variable")

from torchtitan.experiments.deterministic_vllm_rl.weights.converter import (
    torchtitan_to_vllm,
    vllm_to_torchtitan,
)
from torchtitan.experiments.deterministic_vllm_rl.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)

from torchtitan.models.qwen3.model.args import Qwen3ModelArgs


class VLLMRolloutEngine:
    """
    vLLM engine for fast rollouts with weight updates.

    Note: vLLM loads from model_config.model path, so we create a temporary
    directory with updated weights and restart the engine. This is faster than
    recreating temp dirs repeatedly and handles config/tokenizer files properly.

    Args:
        model_path: Path to HuggingFace model (for config/tokenizer)
        temp_checkpoint_dir: Directory to save temporary weight checkpoints
    """

    def __init__(self, model_path: str, temp_checkpoint_dir: str = "./converted"):
        self.base_model_path = model_path
        self.temp_model_dir = os.path.abspath(
            os.path.join(temp_checkpoint_dir, "vllm_temp_model")
        )
        os.makedirs(self.temp_model_dir, exist_ok=True)

        import glob

        # Copy config/tokenizer files from base model to temp dir
        import shutil

        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ]:
            src = os.path.join(model_path, file)
            if os.path.exists(src):
                shutil.copy2(src, self.temp_model_dir)

        # Copy the original model shard files if they exist
        # We'll overwrite these with our single model.safetensors later
        for shard_file in glob.glob(os.path.join(model_path, "model-*.safetensors")):
            dst = os.path.join(self.temp_model_dir, os.path.basename(shard_file))
            shutil.copy2(shard_file, dst)

        # Copy index file if it exists
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            shutil.copy2(index_file, self.temp_model_dir)

        self.llm = None
        print("vLLM rollout engine initialized (will load on first use)")

    def update_weights(self, vllm_compat_state: dict) -> None:
        """
        Update vLLM model weights from vLLM-compat state dict.

        This converts weights to vLLM format, saves them, and reloads using
        vLLM's reload_weights() API after updating the model path config.

        Args:
            vllm_compat_state: vLLM-compat model state dict (with gate_up_proj/down_proj)
        """
        # Convert vLLM-compat -> vLLM (torchtitan_to_vllm handles both formats)
        vllm_state = torchtitan_to_vllm(vllm_compat_state)

        # Save to temp model directory
        checkpoint_path = os.path.join(self.temp_model_dir, "model.safetensors")

        # Update the shard files that vLLM will actually load
        # We need to split our weights to match the original 2-shard structure
        import glob
        import json

        shard_files = sorted(
            glob.glob(os.path.join(self.temp_model_dir, "model-*.safetensors"))
        )
        index_file = os.path.join(self.temp_model_dir, "model.safetensors.index.json")

        if len(shard_files) == 2 and os.path.exists(index_file):
            # Load the index to see which weights go in which shard
            with open(index_file, "r") as f:
                index_data = json.load(f)

            weight_map = index_data["weight_map"]

            # Split weights according to the index
            shard1_weights = {}
            shard2_weights = {}

            for key, value in vllm_state.items():
                shard_file = weight_map.get(key, shard_files[0])
                if "model-00001-of-00002" in shard_file:
                    shard1_weights[key] = value
                else:
                    shard2_weights[key] = value

            # Ensure weights stay in bfloat16
            shard1_weights = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in shard1_weights.items()
            }
            shard2_weights = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in shard2_weights.items()
            }

            # Save to the shard files
            save_file(shard1_weights, shard_files[0])
            save_file(shard2_weights, shard_files[1])
        else:
            # Ensure weights stay in bfloat16
            vllm_state = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in vllm_state.items()
            }
            # Fallback: save as single file
            save_file(vllm_state, checkpoint_path)

        # First time: create the engine
        if self.llm is None:
            self.llm = LLM(
                model=self.temp_model_dir,
                trust_remote_code=True,
                max_model_len=2048,
                dtype="bfloat16",
                gpu_memory_utilization=0.3,  # Reduced from 0.5
                seed=42,  # Fixed seed for determinism
                enforce_eager=True,
            )
            print("✓ Created new vLLM engine")
        else:
            # Use collective_rpc to call reload_weights on all workers
            # This reloads weights from temp_model_dir without recreating the engine
            self.llm.collective_rpc("reload_weights")

    @torch.no_grad()
    def generate(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        n_samples_per_prompt: int = 4,
    ) -> tuple[
        list[str], torch.Tensor, list[list[int]], list[list[float]], list[list[int]]
    ]:
        """
        Generate samples using vLLM.

        Args:
            prompt_texts: List of prompt strings
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            n_samples_per_prompt: Number of samples per prompt

        Returns:
            completions: List of completion strings
            log_probs: [batch] - Sum of log probs for each completion
            token_ids: List of token ID lists for each completion (generated tokens only)
            token_log_probs: List of per-token log prob lists for each completion
            prompt_token_ids: List of prompt token ID lists for each completion
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_samples_per_prompt,
            seed=42,
            logprobs=1,
            prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
        )

        outputs = self.llm.generate(prompt_texts, sampling_params)

        # Extract completions and log probs
        completions = []
        log_probs_list = []
        token_ids_list = []
        token_log_probs_list = []
        prompt_token_ids_list = []

        for output in outputs:
            # Extract prompt token IDs from the output
            prompt_token_ids = output.prompt_token_ids

            for sample in output.outputs:
                completions.append(sample.text)

                # Store prompt tokens for this sample
                prompt_token_ids_list.append(prompt_token_ids)

                # Extract token IDs (generated tokens only)
                token_ids = sample.token_ids
                token_ids_list.append(token_ids)

                # Extract per-token log probs
                per_token_log_probs = [
                    list(logprob_dict.values())[0].logprob
                    for logprob_dict in sample.logprobs
                ]
                token_log_probs_list.append(per_token_log_probs)

                # Sum log probs across generated tokens
                total_log_prob = sum(per_token_log_probs)
                log_probs_list.append(total_log_prob)

        log_probs = torch.tensor(log_probs_list, dtype=torch.float32)

        return (
            completions,
            log_probs,
            token_ids_list,
            token_log_probs_list,
            prompt_token_ids_list,
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "llm"):
            del self.llm
            torch.cuda.empty_cache()


def download_and_convert_model(
    model_name: str, cache_dir: str = "./models", output_dir: str = "./converted"
) -> tuple[str, str]:
    """
    Download model from HuggingFace and convert to TorchTitan format.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-1.7B")
        cache_dir: Directory to cache the downloaded model
        output_dir: Directory to save converted weights

    Returns:
        titan_checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to downloaded HuggingFace model
    """
    import sys
    print(f"[DEBUG] download_and_convert_model called with model_name={model_name}", flush=True)
    sys.stdout.flush()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEBUG] Created output dir: {output_dir}", flush=True)
    sys.stdout.flush()

    # Download model from HuggingFace
    print(f"Downloading {model_name} from HuggingFace...", flush=True)
    sys.stdout.flush()
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )
    print(f"  Downloaded to: {model_path}", flush=True)
    sys.stdout.flush()

    # Convert to TorchTitan format
    print("Converting weights to TorchTitan format...", flush=True)
    sys.stdout.flush()
    print(f"[DEBUG] Calling vllm_to_torchtitan({model_path})...", flush=True)
    sys.stdout.flush()
    titan_state = vllm_to_torchtitan(model_path)
    print(f"[DEBUG] vllm_to_torchtitan returned {len(titan_state)} weights", flush=True)
    sys.stdout.flush()
    
    titan_checkpoint_path = os.path.join(output_dir, "qwen3_torchtitan.safetensors")
    print(f"[DEBUG] Saving to {titan_checkpoint_path}...", flush=True)
    sys.stdout.flush()
    save_file(titan_state, titan_checkpoint_path)
    print(f"  Saved TorchTitan weights to: {titan_checkpoint_path}", flush=True)
    sys.stdout.flush()

    return titan_checkpoint_path, model_path


def load_model(checkpoint_path: str, model_path: str, use_vllm_compat: bool = True):
    """
    Load TorchTitan model from checkpoint.

    Args:
        checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to HuggingFace model (for config)
        use_vllm_compat: If True, use vLLM-compatible model, else use standard model

    Returns:
        model: Loaded TorchTitan model
    """
    # Load HuggingFace config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create model args
    model_args = Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
    )

    # state_dict is in standard TorchTitan format (w1, w2, w3)
    state_dict = load_file(checkpoint_path)

    # Create default PEFT config (disabled)
    from torchtitan.config.job_config import PEFT
    peft_config = PEFT()

    if use_vllm_compat:
        # Create and load model (using vLLM-compat for bitwise determinism)
        from torchtitan.experiments.deterministic_vllm_rl.models.qwen3 import (
            Qwen3VLLMCompatModel,
        )

        model = Qwen3VLLMCompatModel(model_args)
        # Convert to vLLM-compat format (merged gate_up_proj, down_proj)
        vllm_compat_state = torchtitan_to_vllm_compat(state_dict)
        model.load_state_dict(vllm_compat_state, strict=False)
    else:
        # Use standard TorchTitan model
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args, peft_config)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=False)

    model.to(torch.bfloat16)

    return model


def extract_numeric_answer(text: str) -> str | None:
    """
    Extract numeric answer from model completion.

    Looks for patterns like "#### 123" or final numbers in the text.

    Args:
        text: Completion text

    Returns:
        Extracted answer as string, or None if not found
    """
    # GSM8K uses #### to denote the final answer
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        # Remove commas from numbers
        return match.group(1).replace(",", "")

    # Fallback: look for last number in text
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def math_reward_function(
    completions: list[str],
    expected_answers: list[str],
    group_size: int = 4,
) -> torch.Tensor:
    """
    Reward function for math problems (e.g., GSM8K).

    Gives high reward for correct answers, low for incorrect.

    Args:
        completions: List of completion strings
        expected_answers: List of expected answers (one per prompt, repeated for group_size)
        group_size: Number of samples per prompt

    Returns:
        rewards: [batch]
    """
    rewards = []

    for idx, completion in enumerate(completions):
        # Map completion index to prompt index
        prompt_idx = idx // group_size
        expected = expected_answers[prompt_idx].strip().lower()

        # Extract answer from completion
        predicted = extract_numeric_answer(completion)

        if predicted is None:
            # No valid answer found
            reward = 0.0
        elif predicted.lower() == expected:
            # Correct answer
            reward = 1.0
        else:
            # Wrong answer
            reward = 0.0

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)


def load_gsm8k_dataset(split: str = "train", num_samples: int = 100):
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ("train" or "test")
        num_samples: Number of samples to load

    Returns:
        prompts: List of problem prompts
        answers: List of expected answers (numeric strings)
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split=split)

        prompts = []
        answers = []

        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            question = item["question"]
            answer = item["answer"]

            # Extract the final numeric answer from the answer field
            # GSM8K answers are like "some explanation\n#### 42"
            answer_num = extract_numeric_answer(answer)
            if answer_num is None:
                continue

            # Format prompt for the model
            prompt = f"Question: {question}\nAnswer:"

            prompts.append(prompt)
            answers.append(answer_num)

        return prompts, answers

    except ImportError:
        print("⚠ datasets library not installed. Install with: pip install datasets")
        return None, None
    except Exception as e:
        print(f"⚠ Failed to load GSM8K dataset: {e}")
        return None, None


def trivial_reward_function(
    completions: list[str],
    tokenizer=None,
    expected_answers: list[str] | None = None,
    group_size: int = 4,
) -> torch.Tensor:
    """
    Reward function based on correctness and lowercase preference.

    Penalizes non-English characters to keep output in English.
    Rewards correct answers to factual questions.
    Penalizes capital letters to encourage lowercase output.

    Args:
        completions: List of completion strings
        tokenizer: Tokenizer to count tokens
        expected_answers: List of expected answers (one per prompt, repeated for group_size)
        group_size: Number of samples per prompt

    Returns:
        rewards: [batch]
    """
    batch_size = len(completions)
    rewards = []

    for idx, completion in enumerate(completions):
        # Start with base reward of 1.0
        reward = 1.0

        total_chars = len(completion)
        if total_chars == 0:
            rewards.append(0.0)
            continue

        # Penalty for non-English characters (keep it in English)
        # Count non-ASCII characters
        non_ascii_count = sum(1 for c in completion if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / total_chars
        # Strong penalty if >10% non-ASCII
        if non_ascii_ratio > 0.1:
            reward *= 0.1  # 10x penalty

        # Penalty for capital letters (encourage lowercase)
        uppercase_count = sum(1 for c in completion if c.isupper())
        uppercase_ratio = uppercase_count / total_chars
        # Apply penalty proportional to uppercase ratio
        # 0% uppercase = no penalty (1.0x)
        # 100% uppercase = strong penalty (0.1x)
        # Linear interpolation: penalty = 1.0 - 0.9 * uppercase_ratio
        uppercase_penalty = 1.0 - 0.9 * uppercase_ratio
        reward *= uppercase_penalty

        # Bonus for correct answers
        if expected_answers is not None:
            # Map completion index to prompt index
            prompt_idx = idx // group_size
            expected_answer = expected_answers[prompt_idx].lower()
            completion_lower = completion.lower()

            # Check if answer is in completion
            if expected_answer in completion_lower:
                reward *= 2.0  # 2x bonus for correct answer
            else:
                reward *= 0.5  # Penalty for wrong answer

        rewards.append(reward)

    rewards = torch.tensor(rewards, dtype=torch.float32)

    return rewards


def compute_grpo_advantages(
    rewards: torch.Tensor, group_size: int = 4, beta: float = 0.1
) -> torch.Tensor:
    """
    Compute advantages using GRPO-style exponential weighting.

    GRPO uses exponential advantages within groups which can be numerically
    unstable without bitwise determinism. Small differences in reward computation
    can lead to drastically different exp(reward/beta) values.

    This implementation uses the proper GRPO formulation:
    advantage_i = exp(reward_i / beta) / Z - 1
    where Z = mean(exp(reward_j / beta)) for j in group

    Args:
        rewards: [batch]
        group_size: Number of samples per prompt (batch must be divisible by this)
        beta: Temperature parameter for exponential weighting (lower = more unstable)

    Returns:
        advantages: [batch]
    """
    batch_size = rewards.shape[0]
    assert (
        batch_size % group_size == 0
    ), f"Batch size {batch_size} must be divisible by group_size {group_size}"

    num_groups = batch_size // group_size
    rewards_grouped = rewards.view(num_groups, group_size)

    # GRPO exponential advantages: exp(reward / beta)
    # This is numerically unstable and will explode without bitwise invariance!
    exp_rewards = torch.exp(rewards_grouped / beta)

    # Normalize by group mean (this is where instability shows up)
    group_mean_exp = exp_rewards.mean(dim=1, keepdim=True)

    # Advantage = normalized_exp - 1
    advantages_grouped = exp_rewards / group_mean_exp - 1.0

    # Flatten back
    advantages = advantages_grouped.view(-1)

    return advantages


def compute_grpo_advantages_stable(
    rewards: torch.Tensor, group_size: int = 4
) -> torch.Tensor:
    """
    Compute advantages using simple mean-centering (stable fallback).

    This is a simplified version that just uses mean-centering within groups.
    Use this if you want stable training without bitwise invariance.

    Args:
        rewards: [batch]
        group_size: Number of samples per prompt (batch must be divisible by this)

    Returns:
        advantages: [batch]
    """
    batch_size = rewards.shape[0]
    assert (
        batch_size % group_size == 0
    ), f"Batch size {batch_size} must be divisible by group_size {group_size}"

    num_groups = batch_size // group_size
    rewards_grouped = rewards.view(num_groups, group_size)

    # Compute advantages: reward - group_mean
    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    advantages_grouped = rewards_grouped - group_means

    # Flatten back
    advantages = advantages_grouped.view(-1)

    return advantages


def policy_gradient_loss(
    log_probs: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """
    Compute policy gradient loss.

    L = -E[log π(a|s) * A(s,a)]

    Args:
        log_probs: [batch, seq_len] - Log probs of generated tokens
        advantages: [batch] - Advantages for each sample

    Returns:
        loss: scalar
    """
    # Sum log probs across sequence for each sample
    total_log_probs = log_probs.sum(dim=1)  # [batch]

    # Policy gradient: -log_prob * advantage
    pg_loss = -(total_log_probs * advantages).mean()

    return pg_loss


def compute_distill_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_list: list[list[list[list]]],
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute KL divergence loss for on-policy distillation.

    Uses teacher's top-K logprobs to compute a sparse KL divergence,
    which pushes the student distribution closer to the teacher.

    Args:
        student_logits: [batch, seq_len, vocab_size] - Student model logits
        teacher_topk_list: Teacher's top-K logprobs per position
                          Structure: [batch][position][top_k] = [token_id, logprob]
        mask: [batch, seq_len] - Mask for valid positions (1 = valid, 0 = pad)
        temperature: Temperature for softening distributions

    Returns:
        Scalar KL divergence loss
    """
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape

    total_kl = 0.0
    num_valid_positions = 0

    for b in range(batch_size):
        teacher_seq = teacher_topk_list[b] if b < len(teacher_topk_list) else []

        for t in range(seq_len):
            if mask[b, t] < 0.5:
                continue  # Skip masked positions

            if t >= len(teacher_seq) or not teacher_seq[t]:
                continue  # No teacher logprobs for this position

            # Get student log probs at this position
            student_log_probs = F.log_softmax(
                student_logits[b, t, :].float() / temperature, dim=-1
            )

            # Get teacher's top-K for this position
            teacher_topk = teacher_seq[t]  # List of [token_id, logprob]

            # Compute KL divergence over teacher's top-K tokens
            # KL(teacher || student) = sum_k p_teacher(k) * (log p_teacher(k) - log p_student(k))
            position_kl = 0.0
            for token_id, teacher_logprob in teacher_topk:
                if not isinstance(token_id, int) or token_id < 0 or token_id >= vocab_size:
                    continue

                # Teacher probability (from logprob)
                teacher_prob = torch.exp(torch.tensor(teacher_logprob / temperature, device=device))

                # Student log prob for this token
                student_log_prob = student_log_probs[token_id]

                # KL contribution: p_teacher * (log p_teacher - log p_student)
                # = p_teacher * log p_teacher - p_teacher * log p_student
                kl_contrib = teacher_prob * (teacher_logprob / temperature - student_log_prob)
                position_kl += kl_contrib

            total_kl += position_kl
            num_valid_positions += 1

    if num_valid_positions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_kl / num_valid_positions


def compute_policy_gradient_loss_vllm(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    vllm_token_log_probs: list[list[float]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    batch: dict | None = None,
    distill_kl_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO policy gradient loss by re-evaluating completions under current policy.

    Optionally includes on-policy distillation loss when batch contains teacher logprobs.

    Args:
        model: Current policy model
        vllm_token_ids: Generated token IDs for each completion
        vllm_token_log_probs: Per-token log probs from vLLM (reference)
        prompt_token_ids: Prompt token IDs for each completion
        advantages: [batch] - Advantages for each sample
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient
        batch: Optional batch dict containing 'onpolicydistill_logprobs' from teacher
        distill_kl_coef: Coefficient for distillation KL loss (0 = disabled)

    Returns:
        loss: Total loss (PG + entropy + KL + distillation if enabled)
        metrics: Training metrics dict (includes per-token logprob deltas)
    """
    device = next(model.parameters()).device
    advantages = advantages.to(device)

    # Compute reference log probs from per-token values
    # Use PyTorch's sum() to match the reduction order used for total_log_probs
    # This ensures exactly zero KL divergence with batch invariance
    ref_log_probs = torch.stack(
        [
            torch.tensor(lps, dtype=torch.float32, device=device).sum()
            for lps in vllm_token_log_probs
        ]
    )

    # Compute log probs under current policy (WITH GRADIENTS)
    batch_token_log_probs = []
    batch_total_log_probs = []

    # Track per-token differences for the first sample
    first_sample_deltas = []

    for idx, (prompt_toks, gen_toks, vllm_toks_lp) in enumerate(
        zip(prompt_token_ids, vllm_token_ids, vllm_token_log_probs)
    ):
        # Concatenate prompt + generated tokens
        full_sequence = prompt_toks + gen_toks
        full_tensor = torch.tensor(
            full_sequence, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Forward pass
        logits = model(full_tensor)
        # Use F.log_softmax which is overridden by batch_invariant mode for determinism
        # Convert to float32 to match vLLM's sampler behavior (use .to() to preserve gradients)
        log_probs = F.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
        target_tokens = full_tensor[:, 1:]

        # Extract log probs for generated tokens only
        prompt_len = len(prompt_toks)
        gen_start_idx = prompt_len - 1
        gen_end_idx = gen_start_idx + len(gen_toks)

        gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
        gen_token_ids = target_tokens[0, gen_start_idx:gen_end_idx]
        token_lps = gen_token_logprobs.gather(1, gen_token_ids.unsqueeze(-1)).squeeze(
            -1
        )

        batch_token_log_probs.append(token_lps)
        batch_total_log_probs.append(token_lps.sum())

        # For the first sample, store raw tensors for bitwise comparison
        if idx == 0:
            # Keep bfloat16 tensors for bitwise comparison
            titan_lps_bf16 = token_lps.detach().cpu()  # Keep as bfloat16
            titan_lps_f32 = (
                token_lps.detach().cpu().float()
            )  # Convert to float32 for display

            for token_id, vllm_lp, titan_lp_bf16, titan_lp_f32 in zip(
                gen_toks, vllm_toks_lp, titan_lps_bf16, titan_lps_f32
            ):
                first_sample_deltas.append(
                    {
                        "token_id": token_id,
                        "vllm_logprob": vllm_lp,
                        "titan_logprob_bf16": titan_lp_bf16,
                        "titan_logprob_f32": titan_lp_f32.item(),
                    }
                )

    total_log_probs = torch.stack(batch_total_log_probs)

    # Verify bitwise determinism between vLLM and TorchTitan
    if first_sample_deltas:
        vllm_lps_f32 = torch.tensor(
            [d["vllm_logprob"] for d in first_sample_deltas], dtype=torch.float32
        )
        titan_lps_f32 = torch.tensor(
            [d["titan_logprob_f32"] for d in first_sample_deltas], dtype=torch.float32
        )

        bitwise_identical = torch.equal(vllm_lps_f32, titan_lps_f32)

        if bitwise_identical:
            print(
                f"  ✓ vLLM-TorchTitan bitwise determinism verified: {len(first_sample_deltas)} tokens match exactly"
            )
        else:
            num_different = (vllm_lps_f32 != titan_lps_f32).sum().item()
            deltas = (vllm_lps_f32 - titan_lps_f32).abs()
            max_delta = deltas.max().item()
            avg_delta = deltas.mean().item()
            print(
                f"  ⚠ vLLM-TorchTitan logprobs differ: {num_different}/{len(first_sample_deltas)} tokens"
            )
            print(f"    Max delta: {max_delta:.6e}, Avg delta: {avg_delta:.6e}")
            print(
                f"    vLLM logprobs:     {[f'{lp:.10f}' for lp in vllm_lps_f32[:5].tolist()]}"
            )
            print(
                f"    TorchTitan logprobs: {[f'{lp:.10f}' for lp in titan_lps_f32[:5].tolist()]}"
            )

    # PPO clipped objective
    log_ratio = total_log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)
    unclipped_loss = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
    clipped_loss = clipped_ratio * advantages
    pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    # Entropy bonus
    all_token_log_probs = torch.cat(batch_token_log_probs)
    entropy = -all_token_log_probs.mean()
    entropy_bonus = -entropy_coef * entropy

    # KL divergence penalty
    kl_div = (ratio - 1 - log_ratio).mean()

    # Total loss
    total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
        .float()
        .mean()
        .item(),
        "per_token_deltas": first_sample_deltas,  # Per-token logprob differences for first sample
    }

    # ========== On-Policy Distillation ==========
    # Automatically apply distillation if batch contains teacher logprobs
    if (
        batch is not None
        and batch.get("onpolicydistill_logprobs")
        and distill_kl_coef > 0
    ):
        print(f"  [Distillation] Computing KL loss with coef={distill_kl_coef}")

        # Recompute logits for full sequences to get vocab distribution
        # We need logits (not just selected log probs) for KL divergence
        all_logits = []
        all_masks = []

        for prompt_toks, gen_toks in zip(prompt_token_ids, vllm_token_ids):
            full_sequence = prompt_toks + gen_toks
            full_tensor = torch.tensor(
                full_sequence, dtype=torch.long, device=device
            ).unsqueeze(0)

            # Forward pass for logits
            with torch.enable_grad():
                logits = model(full_tensor)  # [1, seq_len, vocab_size]

            # We want logits for positions that predict generated tokens
            # i.e., positions prompt_len-1 to prompt_len+gen_len-2
            prompt_len = len(prompt_toks)
            gen_start_idx = prompt_len - 1
            gen_end_idx = gen_start_idx + len(gen_toks)

            # Extract logits for generated positions [1, gen_len, vocab_size]
            gen_logits = logits[:, gen_start_idx:gen_end_idx, :]
            all_logits.append(gen_logits)

            # Create mask (all 1s for generated tokens)
            mask = torch.ones(1, len(gen_toks), device=device)
            all_masks.append(mask)

        # Pad and stack (handle variable lengths)
        max_gen_len = max(lg.shape[1] for lg in all_logits)
        vocab_size = all_logits[0].shape[2]

        stacked_logits = torch.zeros(
            len(all_logits), max_gen_len, vocab_size, device=device
        )
        stacked_masks = torch.zeros(len(all_logits), max_gen_len, device=device)

        for i, (lg, mk) in enumerate(zip(all_logits, all_masks)):
            stacked_logits[i, : lg.shape[1], :] = lg[0]
            stacked_masks[i, : mk.shape[1]] = mk[0]

        # Compute distillation loss
        distill_loss = compute_distill_kl_loss(
            student_logits=stacked_logits,
            teacher_topk_list=batch["onpolicydistill_logprobs"],
            mask=stacked_masks,
        )

        metrics["distill_loss"] = distill_loss.item()
        total_loss = total_loss + distill_kl_coef * distill_loss
        print(f"  [Distillation] Loss: {distill_loss.item():.4f}")

    return total_loss, metrics


def rl_update_step(
    model,
    tokenizer,
    vllm_engine: VLLMRolloutEngine,
    prompt_texts: list[str],
    optimizer: torch.optim.Optimizer,
    expected_answers: list[str] | None = None,
    group_size: int = 8,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    use_vllm_compat: bool = True,
    num_rollout_batches: int = 1,
    reward_fn=None,
    grpo_beta: float = 0.1,
    use_stable_grpo: bool = False,
) -> dict:
    """
    Perform one RL update step using vLLM for rollouts.

    Args:
        model: Policy model (TorchTitan)
        tokenizer: Tokenizer
        vllm_engine: Persistent vLLM engine
        prompt_texts: List of prompt strings
        optimizer: Optimizer
        expected_answers: List of expected answers for each prompt
        group_size: Number of samples per prompt for GRPO
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        use_vllm_compat: Whether to use vLLM-compatible model
        num_rollout_batches: Number of rollout batches per update (more rollouts = more samples)
        reward_fn: Reward function (defaults to trivial_reward_function)
        grpo_beta: Beta parameter for GRPO exponential weighting (lower = more unstable)
        use_stable_grpo: If True, use stable GRPO (mean-centering) instead of exponential

    Returns:
        metrics: Dict of training metrics
    """
    # Default reward function
    if reward_fn is None:
        reward_fn = trivial_reward_function

    # Update vLLM weights from current policy (only once per update)
    titan_state = model.state_dict()
    vllm_compat_state = torchtitan_to_vllm_compat(titan_state)
    vllm_engine.update_weights(vllm_compat_state)

    # Accumulate gradients over multiple rollout batches
    optimizer.zero_grad()

    all_completions = []
    all_rewards = []
    all_advantages = []
    total_loss = 0.0
    batch_metrics = []

    for batch_idx in range(num_rollout_batches):
        # Generate samples using vLLM
        (
            completions,
            vllm_log_probs,
            vllm_token_ids,
            vllm_token_log_probs,
            prompt_token_ids,
        ) = vllm_engine.generate(
            prompt_texts,
            max_new_tokens,
            temperature,
            n_samples_per_prompt=group_size,
        )

        # Compute rewards using provided reward function
        if reward_fn == trivial_reward_function:
            rewards = reward_fn(completions, tokenizer, expected_answers, group_size)
        elif reward_fn == math_reward_function:
            rewards = reward_fn(completions, expected_answers, group_size)
        else:
            rewards = reward_fn(completions, expected_answers, group_size)

        # Normalize rewards for stability (mean=0, std=1)
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 1e-8:
            rewards_normalized = (rewards - reward_mean) / reward_std
        else:
            rewards_normalized = rewards - reward_mean

        # Compute advantages using GRPO
        if use_stable_grpo:
            advantages = compute_grpo_advantages_stable(rewards_normalized, group_size)
        else:
            advantages = compute_grpo_advantages(
                rewards_normalized, group_size, beta=grpo_beta
            )

        # Compute loss using current policy
        loss, loss_metrics = compute_policy_gradient_loss_vllm(
            model,
            vllm_token_ids,
            vllm_token_log_probs,
            prompt_token_ids,
            advantages,
            kl_coef=0.1,
        )

        # Accumulate loss (will be averaged later)
        loss = loss / num_rollout_batches
        loss.backward()
        total_loss += loss.item()

        # Track metrics
        all_completions.extend(completions[:2])  # Sample 2 from each batch
        all_rewards.append(reward_mean.item())
        all_advantages.append(advantages.mean().item())
        batch_metrics.append(loss_metrics)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()

    # Aggregate metrics across batches
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_advantage = sum(all_advantages) / len(all_advantages)

    # Use metrics from last batch for detailed stats
    final_metrics = batch_metrics[-1]

    # Return aggregated metrics
    metrics = {
        "loss": total_loss,
        "reward_mean": avg_reward,
        "reward_std": batch_metrics[-1].get("reward_std", 0.0),
        "advantage_mean": avg_advantage,
        "advantage_std": batch_metrics[-1].get("advantage_std", 0.0),
        "sample_completions": all_completions[:2],  # First 2 for inspection
        "num_rollout_batches": num_rollout_batches,
        "total_samples": len(prompt_texts) * group_size * num_rollout_batches,
        **final_metrics,  # Include final batch metrics
    }

    return metrics


def compute_gradient_stats(model: torch.nn.Module) -> dict:
    """
    Compute gradient statistics for training hygiene monitoring.

    Args:
        model: Model with gradients computed

    Returns:
        Dictionary of gradient statistics
    """
    total_norm = 0.0
    max_norm = 0.0
    num_params = 0
    num_zero_grads = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)
            num_params += 1

            # Check for zero gradients
            if param_norm < 1e-10:
                num_zero_grads += 1

    total_norm = total_norm ** 0.5

    return {
        "grad_norm": total_norm,
        "grad_max_norm": max_norm,
        "grad_num_params": num_params,
        "grad_zero_frac": num_zero_grads / max(num_params, 1),
    }


def compute_weight_deltas(model: torch.nn.Module, initial_state: dict) -> dict:
    """
    Compute weight changes from initial state based on magnitude (L2 norm).

    Args:
        model: Current model
        initial_state: Initial model state dict

    Returns:
        Dictionary of weight delta statistics by module
    """
    deltas = {}
    module_stats = {}

    with torch.no_grad():
        current_state = model.state_dict()

        for name, current_param in current_state.items():
            if name not in initial_state:
                continue

            # Move current param to CPU to compare with initial (avoid GPU OOM)
            current_param_cpu = current_param.cpu()
            initial_param = initial_state[name]
            delta = current_param_cpu - initial_param

            # Extract module name (e.g., "layers.0.attention.wq" -> "layers.0")
            parts = name.split(".")
            if len(parts) >= 2:
                module_name = ".".join(parts[:2])
            else:
                module_name = parts[0]

            # Compute magnitude (L2 norm) of change
            delta_norm = torch.linalg.vector_norm(delta).item()
            param_norm = torch.linalg.vector_norm(current_param_cpu).item()

            # Relative change: ||delta|| / ||param||
            relative_change = delta_norm / (param_norm + 1e-8)

            # Accumulate module-level stats
            if module_name not in module_stats:
                module_stats[module_name] = {"norms": [], "relative": []}

            module_stats[module_name]["norms"].append(delta_norm)
            module_stats[module_name]["relative"].append(relative_change)

        # Average module-level stats
        for module_name, stats in module_stats.items():
            deltas[f"weight_delta/{module_name}/magnitude"] = sum(stats["norms"]) / len(
                stats["norms"]
            )
            deltas[f"weight_delta/{module_name}/relative_change"] = sum(
                stats["relative"]
            ) / len(stats["relative"])

    return deltas


def main():
    """Simple RL training loop using vLLM for fast rollouts.

    Supports two modes:
    1. Local mode (default): Use local GSM8K dataset for training
    2. Atropos mode (--atropos): Fetch batches from Atropos API server

    For on-policy distillation:
    - Run with --serve to expose the training model as HTTP API
    - Configure Atropos environment with distillation_enabled=True
    - Atropos will fetch teacher logprobs and include them in batches
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="TorchTitan RL Training with On-Policy Distillation Support"
    )

    # Model config
    parser.add_argument(
        "--model", default="Qwen/Qwen3-1.7B", help="HuggingFace model name"
    )
    parser.add_argument("--cache-dir", default="./models", help="Model cache directory")
    parser.add_argument(
        "--output-dir", default="./converted", help="Output directory for checkpoints"
    )

    # Training config
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--group-size", type=int, default=8, help="Samples per prompt for GRPO"
    )
    parser.add_argument(
        "--num-rollout-batches", type=int, default=2, help="Rollout batches per update"
    )

    # GRPO config
    parser.add_argument(
        "--stable-grpo",
        action="store_true",
        help="Use stable GRPO (mean-centering) instead of exponential",
    )
    parser.add_argument(
        "--grpo-beta", type=float, default=0.1, help="GRPO exponential beta"
    )

    # Atropos API config
    parser.add_argument(
        "--atropos", action="store_true", help="Use Atropos API for training data"
    )
    parser.add_argument(
        "--atropos-url", default="http://localhost:8000", help="Atropos API URL"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for Atropos"
    )
    parser.add_argument(
        "--max-token-len", type=int, default=2048, help="Max token length"
    )

    # Distillation config
    parser.add_argument(
        "--distill-kl-coef",
        type=float,
        default=0.1,
        help="Distillation KL coefficient (0 to disable)",
    )

    # Inference server config
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Expose training model as HTTP API for on-policy inference",
    )
    parser.add_argument(
        "--serve-port", type=int, default=9000, help="Port for inference server"
    )

    # Dataset config
    parser.add_argument(
        "--use-gsm8k", action="store_true", default=True, help="Use GSM8K dataset"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of dataset samples"
    )

    # Wandb config (uses env vars: WANDB_PROJECT, WANDB_TEAM, WANDB_RUN_NAME)
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging. Configure via env vars: "
        "WANDB_PROJECT, WANDB_TEAM, WANDB_RUN_NAME",
    )

    args = parser.parse_args()

    # Map args to config variables
    model_name = args.model
    cache_dir = args.cache_dir
    output_dir = args.output_dir

    # Training config
    group_size = args.group_size
    num_rollout_batches = args.num_rollout_batches
    num_steps = args.steps
    learning_rate = args.lr

    # GRPO config
    use_stable_grpo = args.stable_grpo
    grpo_beta = args.grpo_beta

    # Dataset config
    use_real_dataset = args.use_gsm8k
    num_dataset_samples = args.num_samples

    # Atropos/Distillation config
    use_atropos_api = args.atropos
    atropos_url = args.atropos_url
    distill_kl_coef = args.distill_kl_coef

    # Initialize wandb if enabled (uses TorchTitan's env var pattern)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            import time

            # Use env vars like rest of TorchTitan, with fallbacks
            project = os.environ.get("WANDB_PROJECT", "torchtitan-rl")
            entity = os.environ.get("WANDB_TEAM", None)
            run_name = os.environ.get("WANDB_RUN_NAME", None)

            if run_name is None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                mode = "atropos" if use_atropos_api else "local"
                run_name = f"{model_name.split('/')[-1]}_{mode}_{timestamp}"

            # Config dict for wandb
            wandb_config = {
                "model": model_name,
                "learning_rate": learning_rate,
                "num_steps": num_steps,
                "group_size": group_size,
                "num_rollout_batches": num_rollout_batches,
                "grpo_beta": grpo_beta,
                "use_stable_grpo": use_stable_grpo,
                "distill_kl_coef": distill_kl_coef,
                "use_atropos_api": use_atropos_api,
                "atropos_url": atropos_url if use_atropos_api else None,
                "batch_size": args.batch_size if use_atropos_api else None,
                "max_token_len": args.max_token_len if use_atropos_api else None,
                "serve_inference": args.serve,
                "serve_port": args.serve_port if args.serve else None,
            }

            wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=wandb_config,
                reinit=True,
            )
            print(f"\n✓ Wandb initialized: {wandb_run.url}")
        except ImportError:
            print("⚠ wandb not installed. Install with: pip install wandb")
            args.wandb = False
        except Exception as e:
            print(f"⚠ Failed to initialize wandb: {e}")
            args.wandb = False

    # Check if batch invariance is enabled
    use_vllm_compat = False
    try:
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
        use_vllm_compat = vllm_is_batch_invariant()
    except ImportError:
        # Check environment variable as fallback
        use_vllm_compat = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

    if use_vllm_compat:
        print("✓ Batch invariance detected - using vLLM-compatible model")
        # Add backward pass support to vLLM's batch_invariant mode
        print("  Adding gradient support to vLLM's batch_invariant mode...")
        try:
            from torchtitan.experiments.deterministic_vllm_rl.batch_invariant_backward import (
                enable_batch_invariant_backward_mode,
            )
            enable_batch_invariant_backward_mode()
        except Exception as e:
            print(f"  ⚠ Could not enable batch invariant backward mode: {e}")
            print("  Continuing without bitwise determinism guarantees")
    else:
        print("⚠ Batch invariance NOT detected - using standard model")
        if not use_stable_grpo:
            print(
                "  WARNING: Exponential GRPO may be unstable without bitwise invariance!"
            )

    # Download and convert model
    print("=" * 80)
    print(f"Setting up model: {model_name}")
    print("=" * 80)
    titan_checkpoint_path, model_path = download_and_convert_model(
        model_name, cache_dir, output_dir
    )

    # Load TorchTitan model for training
    print("\nLoading TorchTitan model for training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        titan_checkpoint_path, model_path, use_vllm_compat=use_vllm_compat
    )
    model = model.to(device)
    model.train()

    # Save initial weights for delta computation (on CPU to save GPU memory)
    print("Saving initial weights for tracking...")
    initial_state = {
        name: param.clone().cpu() for name, param in model.state_dict().items()
    }

    # Initialize persistent vLLM engine for rollouts
    print("\nInitializing vLLM engine for rollouts...")
    vllm_engine = VLLMRolloutEngine(model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Start inference server if requested (exposes training model via HTTP)
    inference_server = None
    if args.serve:
        from torchtitan.experiments.deterministic_vllm_rl.inference_server import (
            InferenceServer,
            InferenceConfig,
        )

        print("\n" + "=" * 80)
        print("Starting Inference Server")
        print("=" * 80)
        print(f"  This exposes the TRAINING model via HTTP - true shared memory!")
        print(f"  Atropos can connect to: http://localhost:{args.serve_port}/v1")

        inference_config = InferenceConfig(port=args.serve_port)
        inference_server = InferenceServer(model, tokenizer, inference_config)
        inference_server.start()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # TensorBoard writer
    writer = SummaryWriter("./outputs/rl_training")
    print("\n" + "=" * 80)
    print("TensorBoard logging enabled at: ./outputs/rl_training")
    print("=" * 80)

    # ========== TRAINING MODE SELECTION ==========
    if use_atropos_api:
        # ========== ATROPOS API MODE ==========
        from torchtitan.experiments.deterministic_vllm_rl.atropos_client import (
            AtroposClient,
        )

        print("\n" + "=" * 80)
        print("ATROPOS API MODE")
        print("=" * 80)
        print(f"  Connecting to: {atropos_url}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Max token len: {args.max_token_len}")
        print(f"  Distillation KL coef: {distill_kl_coef}")
        print("=" * 80)

        client = AtroposClient(api_url=atropos_url)
        if not client.register(
            batch_size=args.batch_size,
            max_token_len=args.max_token_len,
            num_steps=num_steps,
        ):
            print("Failed to register with Atropos API. Exiting.")
            return

        for step in range(num_steps):
            print(f"\n[Step {step}] Waiting for batch from Atropos...")
            batch = client.get_batch(block=True)

            if batch is None:
                print("No more batches available. Training complete.")
                break

            batch_dict = batch.to_dict()
            print(f"  Received batch with {len(batch)} sequences")

            # Check for distillation data
            has_distill = batch_dict.get("onpolicydistill_logprobs") is not None
            if has_distill:
                print(f"  [Distillation] Teacher logprobs present in batch")

            # Update vLLM weights from current policy
            titan_state = model.state_dict()
            vllm_compat_state = torchtitan_to_vllm_compat(titan_state)
            vllm_engine.update_weights(vllm_compat_state)

            # Process batch: extract tokens, compute advantages, etc.
            tokens = batch_dict["tokens"]
            masks = batch_dict["masks"]
            scores = batch_dict["scores"]

            # Convert scores to advantages (simple normalization)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            advantages = (scores_tensor - scores_tensor.mean()) / (
                scores_tensor.std() + 1e-8
            )

            # Forward pass and loss computation
            optimizer.zero_grad()

            # For Atropos batches, we need to handle the format differently
            # The tokens already include prompt + generated, masks indicate generated
            all_logits = []
            all_token_log_probs = []

            for seq_idx, (seq_tokens, seq_mask) in enumerate(zip(tokens, masks)):
                seq_tensor = torch.tensor(
                    seq_tokens, dtype=torch.long, device=device
                ).unsqueeze(0)

                # Forward pass
                logits = model(seq_tensor)  # [1, seq_len, vocab]

                # Compute log probs
                log_probs = F.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
                target_tokens = seq_tensor[:, 1:]

                # Gather log probs for actual tokens
                token_log_probs = log_probs.gather(
                    2, target_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Apply mask (only count generated tokens)
                mask_tensor = torch.tensor(
                    seq_mask[1:], dtype=torch.float32, device=device
                ).unsqueeze(0)
                masked_log_probs = token_log_probs * mask_tensor

                all_logits.append(logits)
                all_token_log_probs.append(masked_log_probs.sum())

            # Stack log probs
            total_log_probs = torch.stack(all_token_log_probs)

            # Policy gradient loss
            pg_loss = -(total_log_probs * advantages.to(device)).mean()

            # Add distillation loss if present
            distill_loss = torch.tensor(0.0, device=device)
            if has_distill and distill_kl_coef > 0:
                # Pad and stack logits for distillation
                max_len = max(lg.shape[1] for lg in all_logits)
                vocab_size = all_logits[0].shape[2]
                stacked_logits = torch.zeros(
                    len(all_logits), max_len, vocab_size, device=device
                )
                stacked_masks = torch.zeros(len(all_logits), max_len, device=device)

                for i, lg in enumerate(all_logits):
                    stacked_logits[i, : lg.shape[1], :] = lg[0]
                    stacked_masks[i, : len(masks[i])] = torch.tensor(
                        masks[i], dtype=torch.float32
                    )

                distill_loss = compute_distill_kl_loss(
                    student_logits=stacked_logits[:, :-1, :],
                    teacher_topk_list=batch_dict["onpolicydistill_logprobs"],
                    mask=stacked_masks[:, 1:],
                )
                print(f"  [Distillation] Loss: {distill_loss.item():.4f}")

            total_loss = pg_loss + distill_kl_coef * distill_loss

            # Backward pass
            total_loss.backward()

            # Compute gradient stats before clipping
            grad_stats = compute_gradient_stats(model)

            # Gradient clipping
            grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # Optimizer step
            optimizer.step()

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Compute comprehensive metrics
            metrics = {
                # Loss components
                "loss/total": total_loss.item(),
                "loss/policy_gradient": pg_loss.item(),
                "loss/distillation": distill_loss.item() if has_distill else 0.0,
                # Reward statistics
                "reward/mean": scores_tensor.mean().item(),
                "reward/std": scores_tensor.std().item(),
                "reward/min": scores_tensor.min().item(),
                "reward/max": scores_tensor.max().item(),
                # Advantage statistics
                "advantage/mean": advantages.mean().item(),
                "advantage/std": advantages.std().item(),
                "advantage/min": advantages.min().item(),
                "advantage/max": advantages.max().item(),
                # Gradient statistics (training hygiene)
                "gradients/norm_before_clip": grad_stats["grad_norm"],
                "gradients/norm_after_clip": grad_norm_clipped.item(),
                "gradients/max_norm": grad_stats["grad_max_norm"],
                "gradients/zero_fraction": grad_stats["grad_zero_frac"],
                # Learning rate
                "training/learning_rate": current_lr,
                # Batch info
                "training/batch_size": len(tokens),
                "training/total_tokens": sum(len(t) for t in tokens),
                "training/step": step,
            }

            # Log to TensorBoard
            for key, value in metrics.items():
                writer.add_scalar(key.replace("/", "_"), value, step)

            # Log to wandb if enabled
            if args.wandb and wandb_run:
                import wandb
                wandb.log(metrics, step=step)

            print(
                f"  Loss: {total_loss.item():.4f} | "
                f"PG: {pg_loss.item():.4f} | "
                f"Reward: {scores_tensor.mean().item():.3f} | "
                f"GradNorm: {grad_stats['grad_norm']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Report progress to Atropos
            client.report_step(
                step,
                {
                    "loss": total_loss.item(),
                    "pg_loss": pg_loss.item(),
                    "distill_loss": distill_loss.item() if has_distill else 0.0,
                    "reward_mean": scores_tensor.mean().item(),
                },
            )

        client.close()

    else:
        # ========== LOCAL DATASET MODE ==========
        # Load dataset
        print("\n" + "=" * 80)
        print("Dataset Configuration")
        print("=" * 80)

        if use_real_dataset:
            print(f"Attempting to load GSM8K dataset ({num_dataset_samples} samples)...")
            prompt_texts, expected_answers = load_gsm8k_dataset(
                split="train", num_samples=num_dataset_samples
            )

            if prompt_texts is None or len(prompt_texts) == 0:
                print("⚠ Failed to load dataset, falling back to default prompts")
                use_real_dataset = False

        if not use_real_dataset:
            # Fallback: simple prompts with verifiable answers
            print("Using default prompts (factual questions)")
            prompts_with_answers = [
                ("The capital of France is", "paris"),
                ("What is 7 times 8?", "56"),
                ("The first president of the United States was", "washington"),
                ("The chemical symbol for water is", "h2o"),
                ("The largest planet in our solar system is", "jupiter"),
            ]
            prompt_texts = [p[0] for p in prompts_with_answers]
            expected_answers = [p[1] for p in prompts_with_answers]

        # Select reward function
        reward_fn = math_reward_function if use_real_dataset else trivial_reward_function

        print(f"Loaded {len(prompt_texts)} prompts")
        print(f"Reward function: {reward_fn.__name__}")
        print(f"First prompt: {prompt_texts[0][:80]}...")

        # Training loop
        print(f"\nStarting RL training for {num_steps} steps...")
        print(f"  Prompts: {len(prompt_texts)}")
        print(f"  Samples per prompt: {group_size}")
        print(f"  Rollout batches per update: {num_rollout_batches}")
        print(
            f"  Total samples per update: {len(prompt_texts) * group_size * num_rollout_batches}"
        )
        print(
            f"  GRPO mode: {'Stable (mean-centering)' if use_stable_grpo else f'Exponential (beta={grpo_beta})'}"
        )
        print("=" * 80)

        for step in range(num_steps):
            metrics = rl_update_step(
                model,
                tokenizer,
                vllm_engine,
                prompt_texts,
                optimizer,
                expected_answers=expected_answers,
                group_size=group_size,
                max_new_tokens=20 if not use_real_dataset else 100,
                temperature=1.0,
                use_vllm_compat=use_vllm_compat,
                num_rollout_batches=num_rollout_batches,
                reward_fn=reward_fn,
                grpo_beta=grpo_beta,
                use_stable_grpo=use_stable_grpo,
            )

            # Compute weight deltas from initial state
            weight_deltas = compute_weight_deltas(model, initial_state)

            # Compute gradient stats
            grad_stats = compute_gradient_stats(model)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Comprehensive metrics for wandb
            wandb_metrics = {
                # Loss components
                "loss/total": metrics["loss"],
                "loss/policy_gradient": metrics["pg_loss"],
                "loss/kl_divergence": metrics["kl_div"],
                "loss/entropy": metrics["entropy"],
                # Distillation (if present)
                "loss/distillation": metrics.get("distill_loss", 0.0),
                # PPO metrics
                "ppo/ratio_mean": metrics["ratio_mean"],
                "ppo/clip_fraction": metrics["ratio_clipped_frac"],
                # Reward statistics
                "reward/mean": metrics["reward_mean"],
                "reward/std": metrics.get("reward_std", 0.0),
                # Advantage statistics
                "advantage/mean": metrics["advantage_mean"],
                "advantage/std": metrics.get("advantage_std", 0.0),
                # Gradient statistics (training hygiene)
                "gradients/norm": grad_stats["grad_norm"],
                "gradients/max_norm": grad_stats["grad_max_norm"],
                "gradients/zero_fraction": grad_stats["grad_zero_frac"],
                # Learning rate
                "training/learning_rate": current_lr,
                # Samples
                "training/total_samples": metrics["total_samples"],
                "training/step": step,
            }

            # Add weight delta metrics
            for key, value in weight_deltas.items():
                wandb_metrics[key] = value

            # Log to TensorBoard
            for key, value in wandb_metrics.items():
                writer.add_scalar(key.replace("/", "_"), value, step)

            # Log to wandb if enabled
            if args.wandb and wandb_run:
                import wandb
                wandb.log(wandb_metrics, step=step)

                # Log sample completions as a table every 10 steps
                if step % 10 == 0 and metrics.get("sample_completions"):
                    completion_table = wandb.Table(
                        columns=["step", "completion"],
                        data=[[step, c[:200]] for c in metrics["sample_completions"][:2]],
                    )
                    wandb.log({"samples/completions": completion_table}, step=step)

            print(
                f"\nStep {step:3d} | Loss: {metrics['loss']:.4f} | "
                f"Reward: {metrics['reward_mean']:+.3f} | "
                f"KL: {metrics['kl_div']:.4f} | "
                f"Clip: {metrics['ratio_clipped_frac']:.2%} | "
                f"LR: {current_lr:.2e}"
            )
            print(f"  Sample: {metrics['sample_completions'][0][:80]}...")

            # Check for NaN/Inf (sign of instability)
            if not torch.isfinite(torch.tensor(metrics["loss"])):
                print("\n" + "!" * 80)
                print("ERROR: Loss is NaN/Inf! Training diverged.")
                print(
                    "This likely means the exponential GRPO is unstable without bitwise invariance."
                )
                print("Try setting use_stable_grpo=True or enabling batch invariance mode.")
                print("!" * 80)
                break

    print("\n" + "=" * 80)
    print("Training complete!")
    print("View TensorBoard: tensorboard --logdir=./outputs/rl_training")
    if args.wandb and wandb_run:
        print(f"View Wandb: {wandb_run.url}")
    print("=" * 80)

    # Cleanup
    writer.close()
    if args.wandb and wandb_run:
        import wandb
        wandb.finish()
    if inference_server:
        inference_server.stop()
    del vllm_engine


if __name__ == "__main__":
    main()
