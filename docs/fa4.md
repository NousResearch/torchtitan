# FlashAttention-4 (FA4) Support

## Overview

This PR adds FlashAttention-4 support to torchtitan across all model architectures (Llama3, Llama4, Qwen3, DeepSeek V3).

FA4 is a drop-in replacement for SDPA with the same input/output interface. It provides throughput gains that scale with sequence length, particularly beneficial for long-context training.

## Usage

Install FA4:

```bash
pip install fa4
```

Set `attn_type = "fa4"` in your model flavor or config. For example, Qwen3 30B-A3B:

```python
"30B-A3B-fa4": Qwen3ModelArgs(
    ...
    attn_type="fa4",
)
```

No other changes are needed — FA4 handles causal masking and GQA automatically.

## Benchmark Results

**Qwen3 30B-A3B MoE, 8x NVIDIA B200 (180GB), bf16, TP=8, 20 steps on c4_test**

| Seq Len | Batch Size | SDPA TPS | FA4 TPS  | Speedup   | SDPA TFLOPS | FA4 TFLOPS |
|---------|------------|----------|----------|-----------|-------------|------------|
| 4k      | 2          | 6,611    | 6,612    | +0.0%     | —           | —          |
| 8k      | 2          | 7,718    | 7,872    | +2.0%     | 290.0       | 295.8      |
| 16k     | 1          | 7,146    | 7,257    | +1.6%     | 406.7       | 413.0      |
| 32k     | 1          | 6,400    | 6,685    | **+4.5%** | 611.6       | 638.8      |

Memory usage is identical between SDPA and FA4 at all sequence lengths.

FA4's advantage grows with sequence length — attention becomes a larger fraction of total compute at longer sequences, so a faster attention kernel has more impact.

## Context Parallel (CP) Support

FA4 supports context parallelism with 1/W compute per rank (only K and V are allgathered; Q stays local). Two sequence distribution strategies are available:

- **`round_robin`** (default): Head-tail balanced distribution. Rank r owns a head chunk `[r*half : (r+1)*half]` and a tail chunk `[S-(r+1)*half : S-r*half]`, where `half = seq_len // (2*W)`. Tokens within each shard are in globally ascending order, enabling load-balanced causal attention. Requires `seq_len % (2 * cp_degree) == 0`.
- **`striped`**: Interleaved distribution. Rank r owns tokens at global positions `{r, r+W, r+2W, ...}`. Near load-balanced for causal. Requires `seq_len % cp_degree == 0`.

Both modes allgather only K and V (not Q), reducing communication volume by ~1/3 vs. naive full-sequence allgather.

Configure via TOML:

```toml
[parallelism]
context_parallel_degree = 4
context_parallel_distribution = "round_robin"  # or "striped"

[model]
attn_type = "fa4"
```

### CP Benchmark Results

**Llama3 100M, 8x NVIDIA B200, bf16, CP=2, seq_len=32768, global_batch=4 (131K tokens), 100 steps on c4**

| Distribution | Avg TPS | Avg TFLOPS | Avg MFU  | Memory   | Step-100 Loss |
|--------------|---------|------------|----------|----------|---------------|
| `round_robin` | 53,650  | 913.1      | 40.6%    | 63.7 GiB | 2.586         |
| `striped`     | 52,744  | 897.7      | 39.9%    | 63.7 GiB | 2.022         |

- `round_robin` is ~1.7% faster in throughput. In the past-rank ring step it uses only the head half of K (`K[:S_half]`), reducing FLOPS vs. striped which uses the full K shard with `causal=True`.
- Memory is identical — both strategies allgather the same total K/V volume.
- Loss tracks closely at early steps (step 1: 12.408 vs 12.408; step 50: 2.686 vs 2.687) but diverges by step 100 due to different gradient dynamics from token distribution ordering. Both converge correctly — longer runs (1k+ steps) converge to similar final loss.

## Limitations

- **Requires `fa4` package:** The import is lazy (inside `forward()`), so users without FA4 installed won't see errors unless they set `attn_type="fa4"`.
- **CP `round_robin` requires `seq_len % (2 * cp_degree) == 0`** for the head-tail balanced split.
- **CP backward is approximate:** The online softmax merge treats FA4's returned LSE as a constant (FA4 marks it non-differentiable). This drops the second-order cross-term in the merge weights — standard practice for ring attention, negligible in practice.

## Implementation Details

- `FlashAttention4Wrapper` in `torchtitan/models/attention.py` handles the tensor layout conversion between torchtitan's `(batch, nheads, seqlen, headdim)` and FA4's `(batch, seqlen, nheads, headdim)` format.
- FA4 is wired into all model architectures via `case "fa4"` in `__init__` and shares the same forward branch as SDPA (`case "sdpa" | "fa4"`).
- `FA4ContextParallelWrapper` in `torchtitan/distributed/fa4_context_parallel.py` implements ring-over-gathered attention: K and V are allgathered once, then the W ring steps iterate over gathered shards with appropriate Q/K chunk selection to ensure each FA4 call uses only a standard triangular or non-causal mask (no offset-causal needed). Partial attention results are combined via online softmax (log-sum-exp rescaling).
