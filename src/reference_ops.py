from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loader import TensorLoader
from runtime_config import RuntimeConfig
from weight_mapping import ExpertMapping, FullAttentionMapping, LayerMapping, LinearTensor, MoEMapping


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

def rotary_embeddings(position_ids: Any, config: RuntimeConfig, *, device: Any = None, dtype: Any = None) -> tuple[Any, Any]:
    import torch

    rotary_dim = int(config.full_attention.head_dim * config.partial_rotary_factor)
    inv_freq = 1.0 / (
        config.rope_theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(position_ids.to(device=device, dtype=torch.float32).reshape(-1), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).reshape(*position_ids.shape, rotary_dim)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rotary_pos_emb(query: Any, key: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    return (
        _cat_last(q_rot * cos + _rotate_half(q_rot) * sin, q_pass),
        _cat_last(k_rot * cos + _rotate_half(k_rot) * sin, k_pass),
    )


def full_attention(hidden_states: Any, mapping: FullAttentionMapping, config: RuntimeConfig, weights: ReferenceWeights) -> Any:
    import torch

    batch, seq_len, _ = hidden_states.shape
    full = config.full_attention
    q_proj = weights.linear(hidden_states, mapping.q_proj).view(batch, seq_len, -1, full.head_dim * 2)
    query, gate = q_proj.chunk(2, dim=-1)
    gate = gate.reshape(batch, seq_len, full.attn_dim)
    key = weights.linear(hidden_states, mapping.k_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    value = weights.linear(hidden_states, mapping.v_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    query = rms_norm(query, weights.tensor(mapping.q_norm.name), config.rms_norm_eps).transpose(1, 2)
    key = rms_norm(key, weights.tensor(mapping.k_norm.name), config.rms_norm_eps).transpose(1, 2)
    value = value.transpose(1, 2)
    positions = torch.arange(seq_len, device=hidden_states.device).expand(batch, seq_len)
    cos, sin = rotary_embeddings(positions, config, device=hidden_states.device, dtype=query.dtype)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = _repeat_kv(key, full.num_heads // full.num_key_value_heads)
    value = _repeat_kv(value, full.num_heads // full.num_key_value_heads)
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * (full.head_dim**-0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs, value).transpose(1, 2).reshape(batch, seq_len, full.attn_dim)
    out = out * torch.sigmoid(gate)
    return weights.linear(out, mapping.o_proj).to(hidden_states.dtype)


def decoder_layer(hidden_states: Any, mapping: LayerMapping, config: RuntimeConfig, weights: ReferenceWeights) -> Any:
    if mapping.layer_type != "full_attention":
        raise ValueError(f"reference decoder layer only supports full_attention, got {mapping.layer_type}")
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm.name), config.rms_norm_eps)
    hidden_states = residual + full_attention(hidden_states, mapping.attention, config, weights)
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm.name), config.rms_norm_eps)
    return residual + weights.moe(hidden_states, mapping.mlp, config)


def _repeat_kv(hidden_states: Any, repeats: int) -> Any:
    if repeats == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


def _rotate_half(x: Any) -> Any:
    import torch

    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _cat_last(first: Any, second: Any) -> Any:
    import torch

    return torch.cat((first, second), dim=-1)


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
