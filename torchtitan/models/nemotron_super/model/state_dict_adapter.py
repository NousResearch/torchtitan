# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Nemotron Super (NemotronH).

Maps between HF's flat layer indexing (88 layers, each one type) and our
layout (40 layers, each = mamba + [attn] + moe).

HF checkpoint uses:
  - backbone.layers.{flat_idx} prefix (not model.layers)
  - Per-expert weights: backbone.layers.{i}.mixer.experts.{e}.up_proj.weight
  - torchtitan uses 3D grouped tensors: layers.{i}.moe.experts.w1
"""

import re
from collections import defaultdict
from typing import Any

import torch

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
        self.num_experts = model_args.moe_args.num_experts

        attn_set = set(model_args.attn_layer_idxs)

        # Build flat HF index -> (layer_idx, sublayer_type) mapping
        self.flat_to_layer: dict[int, tuple[int, str]] = {}
        self.layer_to_flat: dict[int, dict[str, int]] = {}
        flat_idx = 0
        for layer_idx in range(model_args.n_layers):
            self.layer_to_flat[layer_idx] = {}

            self.flat_to_layer[flat_idx] = (layer_idx, "mamba")
            self.layer_to_flat[layer_idx]["mamba"] = flat_idx
            flat_idx += 1

            if layer_idx in attn_set:
                self.flat_to_layer[flat_idx] = (layer_idx, "attn")
                self.layer_to_flat[layer_idx]["attn"] = flat_idx
                flat_idx += 1

            self.flat_to_layer[flat_idx] = (layer_idx, "moe")
            self.layer_to_flat[layer_idx]["moe"] = flat_idx
            flat_idx += 1

        # Key maps: HF template -> titan template
        # {flat} = HF flat layer index, {layer} = our layer index
        # Per-expert keys handled separately (not in these maps)

        self._mamba_map = {
            "backbone.layers.{flat}.norm.weight": "layers.{layer}.mamba_norm.weight",
            "backbone.layers.{flat}.mixer.in_proj.weight": "layers.{layer}.mamba.in_proj.weight",
            "backbone.layers.{flat}.mixer.conv1d.weight": "layers.{layer}.mamba.conv1d.weight",
            "backbone.layers.{flat}.mixer.conv1d.bias": "layers.{layer}.mamba.conv1d.bias",
            "backbone.layers.{flat}.mixer.dt_bias": "layers.{layer}.mamba.dt_bias",
            "backbone.layers.{flat}.mixer.A_log": "layers.{layer}.mamba.A_log",
            "backbone.layers.{flat}.mixer.D": "layers.{layer}.mamba.D",
            "backbone.layers.{flat}.mixer.norm.weight": "layers.{layer}.mamba.norm.weight",
            "backbone.layers.{flat}.mixer.out_proj.weight": "layers.{layer}.mamba.out_proj.weight",
        }

        self._attn_map = {
            "backbone.layers.{flat}.norm.weight": "layers.{layer}.attn_norm.weight",
            "backbone.layers.{flat}.mixer.q_proj.weight": "layers.{layer}.attention.wq.weight",
            "backbone.layers.{flat}.mixer.k_proj.weight": "layers.{layer}.attention.wk.weight",
            "backbone.layers.{flat}.mixer.v_proj.weight": "layers.{layer}.attention.wv.weight",
            "backbone.layers.{flat}.mixer.o_proj.weight": "layers.{layer}.attention.wo.weight",
        }

        # Non-expert MoE keys (experts handled separately)
        self._moe_map = {
            "backbone.layers.{flat}.norm.weight": "layers.{layer}.ffn_norm.weight",
            "backbone.layers.{flat}.mixer.gate.weight": "layers.{layer}.moe.router.gate.weight",
            "backbone.layers.{flat}.mixer.shared_experts.up_proj.weight": "layers.{layer}.moe.shared_experts.w1.weight",
            "backbone.layers.{flat}.mixer.shared_experts.down_proj.weight": "layers.{layer}.moe.shared_experts.w2.weight",
            "backbone.layers.{flat}.mixer.fc1_latent_proj.weight": "layers.{layer}.moe.latent_in.weight",
            "backbone.layers.{flat}.mixer.fc2_latent_proj.weight": "layers.{layer}.moe.latent_out.weight",
            # HF e_score_correction_bias -> titan expert_bias (same math in router forward)
            "backbone.layers.{flat}.mixer.gate.e_score_correction_bias": "layers.{layer}.moe.expert_bias",
        }

        self._global_map = {
            "backbone.embeddings.weight": "tok_embeddings.weight",
            "backbone.norm_f.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # MTP keys: mtp.layers.0 = attention + fusion, mtp.layers.1 = MoE + final_norm
        # When num_nextn_predict_layers == 0, all MTP keys map to None (skipped)
        self._mtp_enabled = model_args.num_nextn_predict_layers > 0
        self._mtp_map = {
            # Fusion (on flat layer 0)
            "mtp.layers.0.enorm.weight": "mtp.enorm.weight",
            "mtp.layers.0.hnorm.weight": "mtp.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": "mtp.eh_proj.weight",
            # Attention (flat layer 0)
            "mtp.layers.0.norm.weight": "mtp.attn_norm.weight",
            "mtp.layers.0.mixer.q_proj.weight": "mtp.attention.wq.weight",
            "mtp.layers.0.mixer.k_proj.weight": "mtp.attention.wk.weight",
            "mtp.layers.0.mixer.v_proj.weight": "mtp.attention.wv.weight",
            "mtp.layers.0.mixer.o_proj.weight": "mtp.attention.wo.weight",
            # MoE (flat layer 1)
            "mtp.layers.1.norm.weight": "mtp.ffn_norm.weight",
            "mtp.layers.1.mixer.gate.weight": "mtp.moe.router.gate.weight",
            "mtp.layers.1.mixer.shared_experts.up_proj.weight": "mtp.moe.shared_experts.w1.weight",
            "mtp.layers.1.mixer.shared_experts.down_proj.weight": "mtp.moe.shared_experts.w2.weight",
            "mtp.layers.1.mixer.fc1_latent_proj.weight": "mtp.moe.latent_in.weight",
            "mtp.layers.1.mixer.fc2_latent_proj.weight": "mtp.moe.latent_out.weight",
            # Final norm (on flat layer 1)
            "mtp.layers.1.final_layernorm.weight": "mtp.final_layernorm.weight",
            # correction bias -> expert_bias
            "mtp.layers.1.mixer.gate.e_score_correction_bias": "mtp.moe.expert_bias",
        }

        # HF expert key pattern: {prefix}.mixer.experts.{expert_id}.{up_proj|down_proj}.weight
        # Maps to titan: {prefix}.moe.experts.{w1|w2}  (3D tensors)
        self._expert_hf_to_titan = {
            "up_proj": "w1",
            "down_proj": "w2",
        }
        self._expert_titan_to_hf = {v: k for k, v in self._expert_hf_to_titan.items()}

    def _hf_key_to_flat_idx(self, key: str) -> int | None:
        m = re.match(r"backbone\.layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def _is_expert_key(self, hf_key: str) -> bool:
        return ".mixer.experts." in hf_key and re.search(r"experts\.\d+\.", hf_key) is not None

    def _parse_expert_key(self, hf_key: str) -> tuple[str, int, int, str] | None:
        """Parse {prefix}.layers.{idx}.mixer.experts.{expert_id}.{proj}.weight
        Returns (prefix, flat_idx, expert_id, proj_name) or None."""
        m = re.match(
            r"(backbone|mtp)\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight",
            hf_key,
        )
        if m:
            return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        return None

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        # Buffer for per-expert weights: {(layer_idx, titan_name): {expert_id: tensor}}
        expert_buffer: dict[tuple[int, str], dict[int, Any]] = defaultdict(dict)

        for hf_key, value in hf_state_dict.items():
            # Global keys
            if hf_key in self._global_map:
                titan_key = self._global_map[hf_key]
                if titan_key is not None:
                    state_dict[titan_key] = value
                continue

            # MTP keys (non-expert)
            if hf_key.startswith("mtp.") and not self._is_expert_key(hf_key):
                if self._mtp_enabled:
                    titan_key = self._mtp_map.get(hf_key)
                    if titan_key is not None:
                        state_dict[titan_key] = value
                continue

            # Per-expert keys: buffer for stacking (backbone + mtp)
            if self._is_expert_key(hf_key):
                parsed = self._parse_expert_key(hf_key)
                if parsed is None:
                    continue
                prefix, flat_idx, expert_id, proj_name = parsed
                titan_name = self._expert_hf_to_titan[proj_name]

                if prefix == "backbone":
                    if flat_idx not in self.flat_to_layer:
                        continue
                    layer_idx, sublayer = self.flat_to_layer[flat_idx]
                    if sublayer != "moe":
                        continue
                    buf_key = (f"layers.{layer_idx}.moe.experts", titan_name)
                elif prefix == "mtp":
                    if not self._mtp_enabled:
                        continue
                    buf_key = ("mtp.moe.experts", titan_name)
                else:
                    continue
                expert_buffer[buf_key][expert_id] = value
                continue

            # Regular backbone layer keys
            flat_idx = self._hf_key_to_flat_idx(hf_key)
            if flat_idx is None or flat_idx not in self.flat_to_layer:
                continue

            layer_idx, sublayer = self.flat_to_layer[flat_idx]
            hf_tmpl = re.sub(r"(backbone\.layers\.)\d+\.", r"\g<1>{flat}.", hf_key)

            sublayer_map = {"mamba": self._mamba_map, "attn": self._attn_map, "moe": self._moe_map}
            mapping = sublayer_map.get(sublayer, {})
            titan_tmpl = mapping.get(hf_tmpl)

            if titan_tmpl is None:
                continue

            state_dict[titan_tmpl.replace("{layer}", str(layer_idx))] = value

        # Stack buffered expert weights into 3D tensors
        for (titan_prefix, titan_name), experts in expert_buffer.items():
            num_experts = max(experts.keys()) + 1
            stacked = torch.stack([experts[i] for i in range(num_experts)], dim=0)
            state_dict[f"{titan_prefix}.{titan_name}"] = stacked

        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        reverse_global = {v: k for k, v in self._global_map.items() if v is not None}
        reverse_mtp = {v: k for k, v in self._mtp_map.items() if v is not None}
        hf_state_dict = {}

        for titan_key, value in state_dict.items():
            # Global keys
            if titan_key in reverse_global:
                hf_state_dict[reverse_global[titan_key]] = value
                continue

            # MTP non-expert keys
            if titan_key in reverse_mtp:
                hf_state_dict[reverse_mtp[titan_key]] = value
                continue

            # Expert keys: split 3D tensor into per-expert weights
            expert_match = re.match(r"(layers\.(\d+)\.moe|mtp\.moe)\.experts\.(w\d+)", titan_key)
            if expert_match:
                full_prefix = expert_match.group(1)
                titan_name = expert_match.group(3)
                hf_proj = self._expert_titan_to_hf.get(titan_name)
                if hf_proj is None:
                    continue

                if full_prefix.startswith("layers."):
                    layer_idx = int(expert_match.group(2))
                    flat_idx = self.layer_to_flat.get(layer_idx, {}).get("moe")
                    if flat_idx is None:
                        continue
                    hf_prefix = f"backbone.layers.{flat_idx}"
                else:
                    # MTP experts are on flat layer 1 (the E in "*E")
                    hf_prefix = "mtp.layers.1"

                for expert_id in range(value.shape[0]):
                    hf_key = f"{hf_prefix}.mixer.experts.{expert_id}.{hf_proj}.weight"
                    hf_state_dict[hf_key] = value[expert_id]
                continue

            # Regular backbone layer keys
            m = re.match(r"layers\.(\d+)\.", titan_key)
            if m is None:
                continue
            layer_idx = int(m.group(1))

            prefix_match = re.match(r"layers\.\d+\.(\w+)", titan_key)
            if not prefix_match:
                continue
            prefix = prefix_match.group(1)
            if prefix in ("mamba", "mamba_norm"):
                sublayer = "mamba"
            elif prefix in ("attention", "attn_norm"):
                sublayer = "attn"
            elif prefix in ("moe", "ffn_norm"):
                sublayer = "moe"
            else:
                continue

            flat_indices = self.layer_to_flat.get(layer_idx, {})
            flat_idx = flat_indices.get(sublayer)
            if flat_idx is None:
                continue

            titan_tmpl = re.sub(r"(layers\.)\d+\.", r"\g<1>{layer}.", titan_key)
            sublayer_map = {"mamba": self._mamba_map, "attn": self._attn_map, "moe": self._moe_map}
            mapping = sublayer_map.get(sublayer, {})

            hf_tmpl = None
            for h, t in mapping.items():
                if t == titan_tmpl:
                    hf_tmpl = h
                    break

            if hf_tmpl is None:
                continue

            hf_state_dict[hf_tmpl.replace("{flat}", str(flat_idx))] = value

        return hf_state_dict
