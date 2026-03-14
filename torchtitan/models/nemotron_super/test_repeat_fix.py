"""Test that repeat vs repeat_interleave is the Mamba2 bug."""
import types
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
device = "cuda"

from fla.models.mamba2.modeling_mamba2 import Mamba2

fla_mamba = Mamba2(
    num_heads=8, head_dim=16, hidden_size=64,
    state_size=16, expand=2, n_groups=2,
    conv_kernel=4, use_conv_bias=True, hidden_act="silu",
    chunk_size=32, time_step_min=0.001, time_step_max=0.1,
    use_bias=False, norm_eps=1e-5,
).to(device).float()

x = torch.randn(1, 16, 64, device=device)

# 1. FLA as-is (uses repeat - WRONG for n_groups > 1)
with torch.no_grad():
    fla_out, _, _ = fla_mamba(x)

# 2. Patch repeat -> repeat_interleave and re-run
original_forward = fla_mamba.torch_forward.__func__

def patched_forward(self, input_states, last_state=None, use_cache=False, attention_mask=None):
    _orig = torch.Tensor.repeat
    def _fix(tensor, *sizes):
        if len(sizes) == 4 and sizes[0] == 1 and sizes[1] == 1 and sizes[3] == 1 and tensor.dim() == 4:
            return tensor.repeat_interleave(sizes[2], dim=2)
        return _orig(tensor, *sizes)
    torch.Tensor.repeat = _fix
    try:
        return original_forward(self, input_states, last_state, use_cache, attention_mask)
    finally:
        torch.Tensor.repeat = _orig

fla_mamba.torch_forward = types.MethodType(patched_forward, fla_mamba)

with torch.no_grad():
    fixed_out, _, _ = fla_mamba(x)

# 3. Compare
diff_orig = (fla_out - fixed_out).abs().max().item()
rel_orig = diff_orig / (fixed_out.abs().max().item() + 1e-10)

print(f"FLA (repeat) std:            {fla_out.std().item():.6f}")
print(f"Fixed (repeat_interleave) std: {fixed_out.std().item():.6f}")
print(f"Ratio: {fla_out.std().item() / fixed_out.std().item():.2f}x")
print(f"Max abs diff: {diff_orig:.2e}")
print(f"Max rel diff: {rel_orig:.1%}")
print()
if rel_orig > 0.05:
    print("CONFIRMED: repeat vs repeat_interleave is THE bug")
else:
    print("Not the bug, keep looking")
