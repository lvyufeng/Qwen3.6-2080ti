from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loader import TensorLoader
from reference_ops import linear, silu_mul
from weight_mapping import ExpertMapping, LanguageModelMapping, LinearTensor, ShardedTensor


@dataclass(frozen=True)
class MappedWeightStats:
    tensor_count: int
    bytes: int


class MappedWeights:
    def __init__(self, loader: TensorLoader, mapping: LanguageModelMapping, *, device: str | None = None) -> None:
        self.loader = loader
        self.mapping = mapping
        self.device = device
        self._cache: dict[str, Any] = {}
        self._tensors = _mapping_tensors(mapping)
        self.stats = MappedWeightStats(tensor_count=0, bytes=0)

    def preload(self) -> MappedWeightStats:
        for name in sorted(self._tensors):
            self.tensor(self._tensors[name])
        return self.stats

    def tensor(self, name: Any) -> Any:
        tensor_name, mapped = self._resolve_tensor(name)
        tensor = self._cache.get(tensor_name)
        if tensor is not None:
            return tensor
        tensor = self.loader.tensor_shard(mapped, device=self.device)
        self._cache[tensor_name] = tensor
        self.stats = MappedWeightStats(
            tensor_count=self.stats.tensor_count + 1,
            bytes=self.stats.bytes + mapped.nbytes,
        )
        return tensor

    def linear_weight(self, tensor: LinearTensor) -> tuple[Any, Any | None]:
        weight = self.tensor(tensor.weight)
        scale = self.tensor(tensor.scale) if tensor.scale is not None else None
        return weight, scale

    def linear(self, hidden_states: Any, tensor: LinearTensor) -> Any:
        weight, scale = self.linear_weight(tensor)
        return linear(hidden_states, weight, scale)

    def expert(self, hidden_states: Any, expert: ExpertMapping) -> Any:
        gate = self.linear(hidden_states, expert.gate_proj)
        up = self.linear(hidden_states, expert.up_proj)
        return self.linear(silu_mul(gate, up), expert.down_proj)

    def clear(self) -> None:
        self._cache.clear()
        self.stats = MappedWeightStats(tensor_count=0, bytes=0)

    def _resolve_tensor(self, name: Any) -> tuple[str, ShardedTensor]:
        mapped_arg = name if hasattr(name, "info") and hasattr(name, "shard") else None
        if hasattr(name, "name"):
            tensor_name = name.name
        else:
            tensor_name = name
        try:
            mapped = self._tensors[tensor_name]
        except KeyError as exc:
            raise KeyError(f"tensor is not mapped on this rank: {tensor_name}") from exc
        if mapped_arg is not None:
            mapped = mapped_arg
        return tensor_name, mapped


def _mapping_tensors(mapping: LanguageModelMapping) -> dict[str, ShardedTensor]:
    tensors: list[ShardedTensor] = [mapping.embed_tokens, mapping.final_norm, mapping.lm_head]
    for layer in mapping.layers:
        tensors.extend((layer.input_layernorm, layer.post_attention_layernorm))
        attention = layer.attention
        if layer.layer_type == "full_attention":
            linear_tensors = (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj)
            tensors.extend((attention.q_norm, attention.k_norm))
        else:
            linear_tensors = (attention.in_proj_qkv, attention.in_proj_z, attention.out_proj, attention.in_proj_a, attention.in_proj_b)
            tensors.extend((attention.conv1d_weight, attention.a_log, attention.dt_bias, attention.norm))
        for linear_tensor in linear_tensors:
            tensors.append(linear_tensor.weight)
            if linear_tensor.scale is not None:
                tensors.append(linear_tensor.scale)
        moe = layer.mlp
        tensors.extend((moe.gate, moe.shared_expert_gate))
        for expert in (*moe.experts, moe.shared_expert):
            for linear_tensor in (expert.gate_proj, expert.up_proj, expert.down_proj):
                tensors.append(linear_tensor.weight)
                if linear_tensor.scale is not None:
                    tensors.append(linear_tensor.scale)
    return {tensor.name: tensor for tensor in tensors}
