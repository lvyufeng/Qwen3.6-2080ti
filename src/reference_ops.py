from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loader import TensorLoader
from runtime_config import RuntimeConfig
from weight_mapping import ExpertMapping, LinearTensor, MoEMapping


@dataclass(frozen=True)
class TopKRouting:
    logits: Any
    scores: Any
    indices: Any


def embedding(input_ids: Any, weight: Any) -> Any:
    import torch.nn.functional as F

    return F.embedding(input_ids, weight)


def rms_norm(hidden_states: Any, weight: Any, eps: float) -> Any:
    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states.float() * variance.add(eps).rsqrt()
    return (normalized * (1.0 + weight.float())).to(hidden_states.dtype)


def silu_mul(gate: Any, up: Any) -> Any:
    import torch.nn.functional as F

    return F.silu(gate) * up


def topk_route(hidden_states: Any, gate_weight: Any, top_k: int) -> TopKRouting:
    import torch
    import torch.nn.functional as F

    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat.float(), gate_weight.float())
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    scores, indices = torch.topk(probabilities, top_k, dim=-1)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    return TopKRouting(logits=probabilities, scores=scores.to(probabilities.dtype), indices=indices)


def linear(hidden_states: Any, weight: Any, scale_inv: Any | None = None) -> Any:
    import torch.nn.functional as F

    return F.linear(hidden_states.float(), dequantize_fp8_weight(weight, scale_inv))


def dequantize_fp8_weight(weight: Any, scale_inv: Any | None = None) -> Any:
    if scale_inv is None:
        return weight.float()
    block = 128
    rows, cols = weight.shape
    expanded_scale = scale_inv.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return weight.float() * expanded_scale[:rows, :cols]


class ReferenceWeights:
    def __init__(self, loader: TensorLoader, *, device: str | None = None) -> None:
        self.loader = loader
        self.device = device

    def tensor(self, name: str) -> Any:
        return self.loader.tensor(name, device=self.device)

    def linear_weight(self, tensor: LinearTensor) -> tuple[Any, Any | None]:
        weight = self.tensor(tensor.weight.name)
        scale = self.tensor(tensor.scale.name) if tensor.scale is not None else None
        return weight, scale

    def linear(self, hidden_states: Any, tensor: LinearTensor) -> Any:
        weight, scale = self.linear_weight(tensor)
        return linear(hidden_states, weight, scale)

    def expert(self, hidden_states: Any, expert: ExpertMapping) -> Any:
        gate = self.linear(hidden_states, expert.gate_proj)
        up = self.linear(hidden_states, expert.up_proj)
        return self.linear(silu_mul(gate, up), expert.down_proj)

    def moe(self, hidden_states: Any, mapping: MoEMapping, config: RuntimeConfig) -> Any:
        import torch

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        routing = topk_route(flat, self.tensor(mapping.gate.name), config.moe.experts_per_token)
        output = torch.zeros_like(flat.float())
        for expert_index, expert in enumerate(mapping.experts):
            token_indices, topk_indices = torch.where(routing.indices == expert_index)
            if token_indices.numel() == 0:
                continue
            token_output = self.expert(flat[token_indices], expert)
            token_output = token_output * routing.scores[token_indices, topk_indices, None]
            output.index_add_(0, token_indices, token_output.float())
        shared = self.expert(flat, mapping.shared_expert)
        shared_gate = self.tensor(mapping.shared_expert_gate.name)
        output = output + torch.sigmoid(flat.float() @ shared_gate.float().t()) * shared.float()
        return output.reshape(original_shape).to(hidden_states.dtype)
