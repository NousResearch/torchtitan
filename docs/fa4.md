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

## Limitations

- **Context Parallel (CP):** FA4 is not compatible with CP. CP works by intercepting the attention kernel dispatch for ring-attention coordination. FA4 uses its own CUDA kernel that PyTorch's CP dispatcher doesn't support. FA4 + CP will raise `NotImplementedError`.
- **Requires `fa4` package:** The import is lazy (inside `forward()`), so users without FA4 installed won't see errors unless they set `attn_type="fa4"`.

## Implementation Details

- `FlashAttention4Wrapper` in `torchtitan/models/attention.py` handles the tensor layout conversion between torchtitan's `(batch, nheads, seqlen, headdim)` and FA4's `(batch, seqlen, nheads, headdim)` format.
- FA4 is wired into all model architectures via `case "fa4"` in `__init__` and shares the same forward branch as SDPA (`case "sdpa" | "fa4"`).
- CP guards in all model families explicitly block `fa4` alongside `varlen`.
