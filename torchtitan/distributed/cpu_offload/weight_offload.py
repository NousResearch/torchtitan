# Copyright (c) 2025 Nous Research. All rights reserved.

"""Weight/parameter offloading via module hooks.

Offloads a module's parameters to CPU between forward passes and prefetches
them back to GPU before the next forward. Uses async H2D/D2H on dedicated
streams so prefetch overlaps with the previous module's computation.

Usage:
    offloader = TensorOffloader()
    hook = WeightOffloadHook(offloader)

    # Register on modules you want to offload
    hook.register(model.layer1)
    hook.register(model.layer2)

    # Training loop — weights auto-prefetch/offload around forward
    output = model(input)
    loss.backward()

    # Cleanup between iterations
    offloader.reset_pool()

Or use the convenience function:
    handles = offload_module_weights(model.experts, offloader)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .tensor_offloader import TensorOffloader, OffloadHandle


class WeightOffloadHook:
    """Manages weight offloading for registered modules.

    Installs pre-forward and post-forward hooks:
    - pre-forward: reload weights from CPU → GPU (async, overlapped)
    - post-forward: offload weights from GPU → CPU (async)

    Args:
        offloader: Shared TensorOffloader for async transfers.
        prefetch_next: If True, prefetch the next module's weights
            during the current module's forward (pipelining).
    """

    def __init__(self, offloader: TensorOffloader, prefetch_next: bool = True):
        self.offloader = offloader
        self.prefetch_next = prefetch_next
        self._handles: Dict[nn.Module, List[OffloadHandle]] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._registered_modules: List[nn.Module] = []
        self._module_index: Dict[nn.Module, int] = {}

    def register(self, module: nn.Module) -> None:
        """Register a module for weight offloading.

        Call this for each module whose weights should live on CPU
        between forward passes. Order of registration determines
        prefetch scheduling.
        """
        idx = len(self._registered_modules)
        self._registered_modules.append(module)
        self._module_index[module] = idx

        # Pre-forward: reload this module's weights to GPU
        h1 = module.register_forward_pre_hook(self._pre_forward_hook)
        # Post-forward: offload this module's weights to CPU
        h2 = module.register_forward_hook(self._post_forward_hook)
        self._hook_handles.extend([h1, h2])

    def initial_offload(self) -> None:
        """Offload all registered modules' weights to CPU.

        Call once after register() to move weights off GPU.
        """
        for module in self._registered_modules:
            handles = []
            for param in module.parameters(recurse=True):
                if param.device.type == "cuda":
                    handle = self.offloader.offload(param, release_storage=False)
                    handles.append((param, handle))
            self._handles[module] = handles
            # Now move params to CPU in-place
            module.to("cpu")
        self.offloader.sync_all()

    def _pre_forward_hook(self, module: nn.Module, args) -> None:
        """Reload weights from CPU → GPU before forward."""
        if module not in self._handles or not self._handles[module]:
            # First forward or weights already on GPU — just ensure on GPU
            if next(module.parameters()).device.type == "cpu":
                module.to("cuda", non_blocking=True)
                self.offloader.sync_all()
            return

        # Reload each parameter from its CPU handle
        handles = self._handles[module]
        for param, handle in handles:
            gpu_data = self.offloader.reload(handle)
            self.offloader.sync_reload()
            param.data = gpu_data
        self._handles[module] = []

    def _post_forward_hook(self, module: nn.Module, args, output) -> None:
        """Offload weights from GPU → CPU after forward."""
        handles = []
        for param in module.parameters(recurse=True):
            if param.device.type == "cuda":
                handle = self.offloader.offload(param, release_storage=False)
                handles.append((param, handle))
        self._handles[module] = handles

        # Move params to CPU to free GPU memory
        self.offloader.sync_offload()
        module.to("cpu")

        # Prefetch next module's weights
        if self.prefetch_next:
            idx = self._module_index.get(module, -1)
            next_idx = idx + 1
            if next_idx < len(self._registered_modules):
                next_mod = self._registered_modules[next_idx]
                if next_mod in self._handles and self._handles[next_mod]:
                    for param, handle in self._handles[next_mod]:
                        gpu_data = self.offloader.reload(handle)
                        param.data = gpu_data
                    self._handles[next_mod] = []

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._registered_modules.clear()
        self._module_index.clear()
        self._handles.clear()


def offload_module_weights(
    modules: nn.ModuleList,
    offloader: Optional[TensorOffloader] = None,
) -> Tuple[WeightOffloadHook, TensorOffloader]:
    """Convenience: register all modules in a ModuleList for weight offloading.

    Args:
        modules: ModuleList of modules to offload.
        offloader: Shared TensorOffloader (created if None).

    Returns:
        (hook, offloader) tuple. Call hook.remove_hooks() to clean up.
    """
    if offloader is None:
        offloader = TensorOffloader(pin_memory=True, use_pool=True)
    hook = WeightOffloadHook(offloader)
    for module in modules:
        hook.register(module)
    return hook, offloader
