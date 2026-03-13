# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Nemotron Super (NemotronH).

Maps between HF's flat layer indexing (88 layers, each one type) and our
block-based layout (40 layers, each = mamba + [attn] + moe).

See NOTES.md "State dict adapter naming assumptions" for the assumed
torchtitan module naming conventions.
"""

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import NemotronSuperModelArgs


class NemotronSuperStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: NemotronSuperModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args

        attn_set = set(model_args.attn_layer_idxs)

        # Build flat HF index -> (block_idx, sublayer_type) mapping
        # HF pattern: for each block, flat layers are M, [*], E
        self.flat_to_block: dict[int, tuple[int, str]] = {}
        self.block_to_flat: dict[int, dict[str, int]] = {}
        flat_idx = 0
        for block_idx in range(model_args.n_layers):
            self.block_to_flat[block_idx] = {}

            self.flat_to_block[flat_idx] = (block_idx, "mamba")
            self.block_to_flat[block_idx]["mamba"] = flat_idx
            flat_idx += 1

            if block_idx in attn_set:
                self.flat_to_block[flat_idx] = (block_idx, "attn")
                self.block_to_flat[block_idx]["attn"] = flat_idx
                flat_idx += 1

            self.flat_to_block[flat_idx] = (block_idx, "moe")
            self.block_to_flat[block_idx]["moe"] = flat_idx
            flat_idx += 1

        # HF key -> torchtitan key mappings, keyed by sublayer type.
        # {flat_layer} is the HF flat index, {block} is our block index.
        # None value means skip (don't load).

        self._mamba_map = {
            "model.layers.{flat}.norm.weight": "layers.{block}.mamba_norm.weight",
            "model.layers.{flat}.mixer.in_proj.weight": "layers.{block}.mamba.in_proj.weight",
            "model.layers.{flat}.mixer.in_proj.bias": "layers.{block}.mamba.in_proj.bias",
            "model.layers.{flat}.mixer.conv1d.weight": "layers.{block}.mamba.conv1d.weight",
            "model.layers.{flat}.mixer.conv1d.bias": "layers.{block}.mamba.conv1d.bias",
            "model.layers.{flat}.mixer.dt_bias": "layers.{block}.mamba.dt_bias",
            "model.layers.{flat}.mixer.A_log": "layers.{block}.mamba.A_log",
            "model.layers.{flat}.mixer.D": "layers.{block}.mamba.D",
            "model.layers.{flat}.mixer.norm.weight": "layers.{block}.mamba.norm.weight",
            "model.layers.{flat}.mixer.out_proj.weight": "layers.{block}.mamba.out_proj.weight",
            "model.layers.{flat}.mixer.out_proj.bias": "layers.{block}.mamba.out_proj.bias",
        }

        self._attn_map = {
            "model.layers.{flat}.norm.weight": "layers.{block}.attn_norm.weight",
            "model.layers.{flat}.mixer.q_proj.weight": "layers.{block}.attention.wq.weight",
            "model.layers.{flat}.mixer.k_proj.weight": "layers.{block}.attention.wk.weight",
            "model.layers.{flat}.mixer.v_proj.weight": "layers.{block}.attention.wv.weight",
            "model.layers.{flat}.mixer.o_proj.weight": "layers.{block}.attention.wo.weight",
        }

        self._moe_map = {
            "model.layers.{flat}.norm.weight": "layers.{block}.moe_norm.weight",
            "model.layers.{flat}.mixer.gate.weight": "layers.{block}.moe.router.weight",
            "model.layers.{flat}.mixer.gate.e_score_correction_bias": "layers.{block}.moe.router.e_score_correction_bias",
            "model.layers.{flat}.mixer.experts.up_proj": "layers.{block}.moe.experts.up_proj",
            "model.layers.{flat}.mixer.experts.down_proj": "layers.{block}.moe.experts.down_proj",
            "model.layers.{flat}.mixer.shared_experts.up_proj.weight": "layers.{block}.moe.shared_experts.up_proj.weight",
            "model.layers.{flat}.mixer.shared_experts.up_proj.bias": "layers.{block}.moe.shared_experts.up_proj.bias",
            "model.layers.{flat}.mixer.shared_experts.down_proj.weight": "layers.{block}.moe.shared_experts.down_proj.weight",
            "model.layers.{flat}.mixer.shared_experts.down_proj.bias": "layers.{block}.moe.shared_experts.down_proj.bias",
            "model.layers.{flat}.mixer.fc1_latent_proj.weight": "layers.{block}.moe.latent_in.weight",
            "model.layers.{flat}.mixer.fc1_latent_proj.bias": "layers.{block}.moe.latent_in.bias",
            "model.layers.{flat}.mixer.fc2_latent_proj.weight": "layers.{block}.moe.latent_out.weight",
            "model.layers.{flat}.mixer.fc2_latent_proj.bias": "layers.{block}.moe.latent_out.bias",
        }

        self._global_map = {
            "model.embeddings.weight": "tok_embeddings.weight",
            "model.norm_f.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Build the full from_hf lookup: hf_key_template -> (sublayer_type, titan_key_template)
        # We need this for the generic key matching in from_hf/to_hf.
        self._hf_to_titan: dict[str, str] = {}
        self._titan_to_hf: dict[str, str] = {}
        for sublayer, mapping in [("mamba", self._mamba_map), ("attn", self._attn_map), ("moe", self._moe_map)]:
            for hf_tmpl, titan_tmpl in mapping.items():
                if titan_tmpl is not None:
                    # Store with sublayer prefix for disambiguation
                    self._hf_to_titan[hf_tmpl] = titan_tmpl
                    self._titan_to_hf[titan_tmpl] = hf_tmpl

    def _hf_key_to_flat_idx(self, key: str) -> int | None:
        """Extract flat layer index from an HF key like 'model.layers.42.mixer.X'."""
        m = re.match(r"model\.layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def _titan_key_to_block_idx(self, key: str) -> int | None:
        """Extract block index from a titan key like 'layers.7.mamba.X'."""
        m = re.match(r"layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def _sublayer_from_titan_key(self, key: str) -> str | None:
        """Determine sublayer type from a titan key."""
        m = re.match(r"layers\.\d+\.(\w+)", key)
        if not m:
            return None
        prefix = m.group(1)
        if prefix in ("mamba", "mamba_norm"):
            return "mamba"
        elif prefix in ("attention", "attn_norm"):
            return "attn"
        elif prefix in ("moe", "moe_norm"):
            return "moe"
        return None

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        for hf_key, value in hf_state_dict.items():
            # Global keys (embeddings, final norm, lm_head)
            if hf_key in self._global_map:
                state_dict[self._global_map[hf_key]] = value
                continue

            # Layer keys - extract flat index
            flat_idx = self._hf_key_to_flat_idx(hf_key)
            if flat_idx is None:
                continue

            if flat_idx not in self.flat_to_block:
                continue

            block_idx, sublayer = self.flat_to_block[flat_idx]

            # Build the template key by replacing the flat index
            hf_tmpl = re.sub(r"(model\.layers\.)\d+\.", r"\g<1>{flat}.", hf_key)

            # Look up in the appropriate sublayer map
            sublayer_map = {"mamba": self._mamba_map, "attn": self._attn_map, "moe": self._moe_map}
            mapping = sublayer_map.get(sublayer, {})
            titan_tmpl = mapping.get(hf_tmpl)

            if titan_tmpl is None:
                # Key not in our mapping - skip
                continue

            titan_key = titan_tmpl.replace("{block}", str(block_idx))
            state_dict[titan_key] = value

        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        # Build reverse global map
        reverse_global = {v: k for k, v in self._global_map.items()}

        hf_state_dict = {}

        for titan_key, value in state_dict.items():
            # Global keys
            if titan_key in reverse_global:
                hf_state_dict[reverse_global[titan_key]] = value
                continue

            # Layer keys - extract block index and sublayer type
            block_idx = self._titan_key_to_block_idx(titan_key)
            if block_idx is None:
                continue

            sublayer = self._sublayer_from_titan_key(titan_key)
            if sublayer is None:
                continue

            # Get the flat HF index for this block + sublayer
            flat_indices = self.block_to_flat.get(block_idx, {})
            flat_idx = flat_indices.get(sublayer)
            if flat_idx is None:
                continue

            # Build template and look up reverse mapping
            titan_tmpl = re.sub(r"(layers\.)\d+\.", r"\g<1>{block}.", titan_key)

            sublayer_map = {"mamba": self._mamba_map, "attn": self._attn_map, "moe": self._moe_map}
            mapping = sublayer_map.get(sublayer, {})

            # Reverse lookup
            hf_tmpl = None
            for h, t in mapping.items():
                if t == titan_tmpl:
                    hf_tmpl = h
                    break

            if hf_tmpl is None:
                continue

            hf_key = hf_tmpl.replace("{flat}", str(flat_idx))
            hf_state_dict[hf_key] = value

        return hf_state_dict


# Alias for backward compat with existing __init__.py imports
Qwen3StateDictAdapter = NemotronSuperStateDictAdapter
