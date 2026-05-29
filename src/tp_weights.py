from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loader import TensorLoader
from weight_mapping import LanguageModelMapping, ShardedTensor


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
            self.tensor(name)
        return self.stats

    def tensor(self, name: str) -> Any:
        tensor = self._cache.get(name)
        if tensor is not None:
            return tensor
        try:
            mapped = self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor is not mapped on this rank: {name}") from exc
        tensor = self.loader.tensor_shard(mapped, device=self.device)
        self._cache[name] = tensor
        self.stats = MappedWeightStats(
            tensor_count=self.stats.tensor_count + 1,
            bytes=self.stats.bytes + mapped.nbytes,
        )
        return tensor

    def clear(self) -> None:
        self._cache.clear()
        self.stats = MappedWeightStats(tensor_count=0, bytes=0)


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
        for linear in linear_tensors:
            tensors.append(linear.weight)
            if linear.scale is not None:
                tensors.append(linear.scale)
        moe = layer.mlp
        tensors.extend((moe.gate, moe.shared_expert_gate))
        for expert in (*moe.experts, moe.shared_expert):
            for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
                tensors.append(linear.weight)
                if linear.scale is not None:
                    tensors.append(linear.scale)
    return {tensor.name: tensor for tensor in tensors}
