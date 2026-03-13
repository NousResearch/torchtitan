# Nemotron Super (NemotronH) -- Implementation Notes

Reference: `nvidia/Nemotron-H-120B-A12B`
HF ref code: `model/hf_ref.py` (auto-generated from `modular_nemotron_h.py`)

## Architecture Overview

Hybrid model with 3 block types interleaved via `hybrid_override_pattern`:
- **M** = Mamba2 SSM block (40 layers)
- **E** = MoE block (40 layers)
- **\*** = Attention block (8 layers)

Pattern: `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME`

Only 8 out of 88 layers are attention. The rest alternate between Mamba2 and MoE.


## Weird / Non-obvious Implementation Details

### 1. MoE experts are NON-GATED (relu^2, not SwiGLU)

Unlike Qwen3/DeepSeek/Mixtral, NemotronH experts use a simple 2-layer MLP:
```
out = down_proj(relu^2(up_proj(x)))
```
NOT the typical gated pattern:
```
out = down_proj(silu(gate_proj(x)) * up_proj(x))
```

This means:
- `NemotronHExperts` stores `up_proj` and `down_proj` only (no `gate_proj`/`w3`)
- Expert weights are 3D tensors: `(num_experts, out_dim, in_dim)`
- The activation is `relu^2` (square of relu), not silu/swish
- The torchtitan `MoE` class and `GroupedExperts` assume gated experts -- this will need
  a separate expert implementation or a `has_gate=False` path

### 2. Latent MoE projection

Before tokens hit the routed experts, they pass through a latent bottleneck:
```
x = fc1_latent_proj(x)    # dim -> moe_latent_size (4096 -> 1024)
x = experts(x)             # expert computation at latent dim
x = fc2_latent_proj(x)    # moe_latent_size -> dim (1024 -> 4096)
```
The expert `up_proj`/`down_proj` operate at `moe_latent_size` (1024), not `dim` (4096).
This massively reduces per-expert parameter count. If `moe_latent_size is None`, these
become `nn.Identity()`.

### 3. Routing uses sigmoid + correction bias, NOT softmax

The router computes `sigmoid(logits)` then adds `e_score_correction_bias` before top-k
selection. After selection, raw sigmoid scores (without bias) are gathered as weights.
This is different from both softmax routing (Mixtral) and the more common sigmoid
routing (DeepSeek V3 uses sigmoid too, but without the correction bias trick).

The correction bias is a registered buffer initialized to zeros -- it's populated from
the pretrained checkpoint and NOT learned during training (it's a buffer, not parameter).

### 4. 512 experts with top-22 is extreme

Most MoE models use 8-64 experts with top-1 or top-2 routing. NemotronH uses 512
experts with top-22, meaning each token activates 22/512 ~ 4.3% of experts. Combined
with `routed_scaling_factor=5.0`, this creates very different load balancing dynamics.

Expert parallel will be essential -- 512 experts is too many for a single device.

### 5. Single-norm block architecture

Each `NemotronHBlock` has only ONE norm (pre-norm before the mixer):
```python
hidden_states = self.norm(hidden_states)
hidden_states = self.mixer(hidden_states)  # mamba, attention, or moe
hidden_states = residual + hidden_states
```
There is NO separate FFN after attention blocks. Each block is purely one type.
This is different from standard transformers where each layer has attention + FFN.

### 6. Attention blocks have NO FFN sublayer

In the hybrid pattern, `*` blocks are pure attention -- no feed-forward after them.
The FFN computation is handled by the adjacent `E` (MoE) blocks. This means attention
layers are much smaller (just QKV projections + output proj) compared to standard
transformer layers.

### 7. Mamba blocks need special mask handling

Mamba layers use a different mask than attention layers:
```python
block_type_to_mask = {
    "mamba": mamba_mask,      # None if all-ones or cached forward
    "attention": causal_mask, # standard causal mask
    "moe": None,              # no mask needed
}
```
The `mamba_mask` is zeroed out (set to None) when:
- Cached forward (has_previous_state=True)
- All positions are attended (attention_mask is all 1s)

### 8. Confusing naming: n_groups vs n_group

- `n_groups = 8` -> Mamba SSM groups (B and C tensors are grouped)
- `n_group = 1` -> MoE routing groups (for grouped top-k expert selection)

These are completely different concepts with almost identical names.

### 9. CUDA stream workaround in NemotronHBlock

```python
if hidden_states.device.type == "cuda":
    stream_context = torch.cuda.stream(torch.cuda.default_stream(hidden_states.device))
```
Mamba kernels may launch on the default CUDA stream, causing race conditions with
PyTorch's current stream on multi-GPU. The block forces the default stream to avoid
reading uninitialized memory.

### 10. No QK-norm on attention

Unlike Qwen3 which uses RMSNorm on Q and K projections, NemotronH attention has NO
qk-norm. The attention is vanilla multi-head attention with RoPE and GQA (32 heads,
2 KV heads).

### 11. rope_theta = 10000 for 262k context

This is surprisingly low. Most 100k+ context models use much higher rope_theta values
(Llama3: 500k, Qwen: 1M+). It's possible they rely on the Mamba layers for long-range
dependencies and only use attention for local patterns. Worth investigating whether
this causes issues with long-context fine-tuning.

### 12. State dict mapping will be complex

The HF model uses flat layer indexing (`model.layers.0`, `model.layers.1`, ...) where
each layer can be mamba, attention, or MoE. The state dict adapter needs to handle:
- Mamba layers: `in_proj`, `conv1d`, `dt_bias`, `A_log`, `D`, `norm`, `out_proj`
- Attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MoE layers: `gate.weight`, `gate.e_score_correction_bias`, `experts.up_proj`,
  `experts.down_proj`, `shared_experts.*`, `fc1_latent_proj`, `fc2_latent_proj`
- Each block: `norm.weight` (single pre-norm)

### 13. Mamba requires external kernels

The fast path needs `causal-conv1d` and `mamba-ssm` packages. Without them, it falls
back to a pure PyTorch naive SSD implementation (the `torch_forward` method) which is
functional but much slower. The naive path is ~500 lines of manual chunk/scan logic.


## Discrepancies in original __init__.py draft (now fixed)

The original `nemotron_super_args` dict had several values that didn't match the HF config:

| Field             | HF Config | Old Dict | Notes                            |
|-------------------|-----------|----------|----------------------------------|
| n_layers          | 88        | 94       | Pattern has exactly 88 chars     |
| n_heads           | 32        | 64       |                                  |
| n_kv_heads        | 2         | 4        |                                  |
| rope_theta        | 10000     | 5000000  |                                  |
| num_experts_per_tok| 22       | 8        | Massive difference               |
| moe_inter_dim     | 2688      | 1536     |                                  |
| score_func        | sigmoid   | softmax  |                                  |

These have been corrected in the current config to match the HF reference.


## State dict adapter naming assumptions

The adapter (`state_dict_adapter.py`) maps between HF's 88 flat layers and our
40-block layout. Each block = (Mamba2, [Attention], MoE).

### Assumed torchtitan module names

These are the names the model implementation should use. If they change,
update the adapter maps (`_mamba_map`, `_attn_map`, `_moe_map`, `_global_map`).

```
tok_embeddings.weight

layers.{i}.mamba_norm.weight
layers.{i}.mamba.in_proj.{weight,bias}
layers.{i}.mamba.conv1d.{weight,bias}
layers.{i}.mamba.dt_bias
layers.{i}.mamba.A_log
layers.{i}.mamba.D
layers.{i}.mamba.norm.weight               # internal gated RMSNorm
layers.{i}.mamba.out_proj.{weight,bias}

layers.{i}.attn_norm.weight                # only if i in attn_layer_idxs
layers.{i}.attention.wq.weight
layers.{i}.attention.wk.weight
layers.{i}.attention.wv.weight
layers.{i}.attention.wo.weight

layers.{i}.moe_norm.weight
layers.{i}.moe.router.weight
layers.{i}.moe.router.e_score_correction_bias
layers.{i}.moe.experts.up_proj            # 3D param (num_experts, inter, in)
layers.{i}.moe.experts.down_proj           # 3D param (num_experts, in, inter)
layers.{i}.moe.shared_experts.up_proj.weight
layers.{i}.moe.shared_experts.down_proj.weight
layers.{i}.moe.latent_in.{weight,bias}    # fc1_latent_proj
layers.{i}.moe.latent_out.{weight,bias}   # fc2_latent_proj

norm.weight
output.weight
```

### HF flat layer index mapping

HF uses `model.layers.{flat_idx}` where flat_idx runs 0-87. For each of
our 40 blocks, the flat indices are assigned sequentially:

- Block without attention: flat M, flat E (2 flat layers)
- Block with attention: flat M, flat *, flat E (3 flat layers)

The block-level pre-norm (`model.layers.{f}.norm.weight`) maps to
`{sublayer}_norm.weight` based on which sublayer type owns that flat index.

### Open questions

- Do we need the `e_score_correction_bias` buffer? It's initialized to zeros
  and is a buffer (not parameter) in HF. May need special handling for DCP.
- Expert weights are raw 3D nn.Parameters in HF, not nn.Linear. Our model
  may wrap them differently depending on whether we use GroupedExperts or
  a custom NemotronH expert module.
- Mamba norm is `Zamba2RMSNormGated` (takes gate arg in forward). Our FLA-based
  mamba may handle this differently -- norm.weight mapping might need adjustment.
