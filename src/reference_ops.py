from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loader import TensorLoader
from runtime_config import RuntimeConfig
from weight_mapping import (
    ExpertMapping,
    FullAttentionMapping,
    LanguageModelMapping,
    LayerMapping,
    LinearAttentionMapping,
    LinearTensor,
    MoEMapping,
)


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


def gated_rms_norm(hidden_states: Any, weight: Any, gate: Any, eps: float) -> Any:
    import torch.nn.functional as F

    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states.float() * variance.add(eps).rsqrt()
    out = weight.float() * normalized.to(hidden_states.dtype)
    return (out * F.silu(gate.float())).to(hidden_states.dtype)


def l2_norm(hidden_states: Any, eps: float = 1e-6) -> Any:
    states = hidden_states.float()
    return states * (states * states).sum(dim=-1, keepdim=True).add(eps).rsqrt()


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


def linear_attention(
    hidden_states: Any,
    mapping: LinearAttentionMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
) -> Any:
    import torch
    import torch.nn.functional as F

    batch, seq_len, _ = hidden_states.shape
    linear_cfg = config.linear_attention
    mixed_qkv = weights.linear(hidden_states, mapping.in_proj_qkv).transpose(1, 2)
    conv_weight = weights.tensor(mapping.conv1d_weight.name).float()
    mixed_qkv = F.conv1d(
        mixed_qkv.float(),
        conv_weight,
        padding=linear_cfg.conv_kernel_dim - 1,
        groups=linear_cfg.qkv_dim,
    )[:, :, :seq_len]
    mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            linear_cfg.key_heads * linear_cfg.key_head_dim,
            linear_cfg.key_heads * linear_cfg.key_head_dim,
            linear_cfg.value_state_dim,
        ],
        dim=-1,
    )
    query = query.reshape(batch, seq_len, linear_cfg.key_heads, linear_cfg.key_head_dim)
    key = key.reshape(batch, seq_len, linear_cfg.key_heads, linear_cfg.key_head_dim)
    value = value.reshape(batch, seq_len, linear_cfg.value_heads, linear_cfg.value_head_dim)
    beta = torch.sigmoid(weights.linear(hidden_states, mapping.in_proj_b))
    a = weights.linear(hidden_states, mapping.in_proj_a)
    a_log = weights.tensor(mapping.a_log.name).float()
    dt_bias = weights.tensor(mapping.dt_bias.name).float()
    g = -a_log.exp() * F.softplus(a.float() + dt_bias)
    repeats = linear_cfg.value_heads // linear_cfg.key_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    core = recurrent_gated_delta_rule(query, key, value, g, beta).reshape(-1, linear_cfg.value_head_dim)
    z = weights.linear(hidden_states, mapping.in_proj_z).reshape(-1, linear_cfg.value_head_dim)
    core = gated_rms_norm(core, weights.tensor(mapping.norm.name), z, config.rms_norm_eps)
    core = core.reshape(batch, seq_len, linear_cfg.value_state_dim)
    return weights.linear(core, mapping.out_proj).to(hidden_states.dtype)


def recurrent_gated_delta_rule(query: Any, key: Any, value: Any, g: Any, beta: Any) -> Any:
    import torch

    initial_dtype = query.dtype
    query = l2_norm(query).transpose(1, 2).float()
    key = l2_norm(key).transpose(1, 2).float()
    value = value.transpose(1, 2).float()
    beta = beta.transpose(1, 2).float()
    g = g.transpose(1, 2).float()
    batch, heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    query = query * (key_dim**-0.5)
    state = torch.zeros(batch, heads, key_dim, value_dim, device=query.device, dtype=torch.float32)
    output = torch.zeros(batch, heads, seq_len, value_dim, device=query.device, dtype=torch.float32)
    for index in range(seq_len):
        q_t = query[:, :, index]
        k_t = key[:, :, index]
        v_t = value[:, :, index]
        state = state * g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(initial_dtype)


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
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm.name), config.rms_norm_eps)
    if mapping.layer_type == "full_attention":
        hidden_states = residual + full_attention(hidden_states, mapping.attention, config, weights)
    elif mapping.layer_type == "linear_attention":
        hidden_states = residual + linear_attention(hidden_states, mapping.attention, config, weights)
    else:
        raise ValueError(f"unsupported reference layer type: {mapping.layer_type}")
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm.name), config.rms_norm_eps)
    return residual + weights.moe(hidden_states, mapping.mlp, config)


def language_model(input_ids: Any, mapping: LanguageModelMapping, config: RuntimeConfig, weights: ReferenceWeights) -> Any:
    hidden_states = embedding(input_ids, weights.tensor(mapping.embed_tokens.name))
    for layer in mapping.layers:
        hidden_states = decoder_layer(hidden_states, layer, config, weights)
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm.name), config.rms_norm_eps)
    return weights.linear(hidden_states, LinearTensor(weight=mapping.lm_head, scale=None))


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


def linear(hidden_states: Any, weight: Any, scale_inv: Any | None = None, *, use_cuda_kernel: bool = True) -> Any:
    import torch
    import torch.nn.functional as F

    if (
        use_cuda_kernel
        and scale_inv is not None
        and getattr(hidden_states, "is_cuda", False)
        and getattr(weight, "is_cuda", False)
        and getattr(scale_inv, "is_cuda", False)
        and weight.dtype == torch.float8_e4m3fn
        and scale_inv.dtype == torch.bfloat16
        and hidden_states.shape[-1] % 128 == 0
        and weight.shape[0] % 128 == 0
    ):
        from fp8_cuda import fp8_e4m3_bf16_linear

        return fp8_e4m3_bf16_linear(hidden_states.float(), weight, scale_inv)
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
        for expert in mapping.experts:
            token_indices, topk_indices = torch.where(routing.indices == expert.index)
            if token_indices.numel() == 0:
                continue
            token_output = self.expert(flat[token_indices], expert)
            token_output = token_output * routing.scores[token_indices, topk_indices, None]
            output.index_add_(0, token_indices, token_output.float())
        if mapping.tp.adds_shared_expert:
            shared = self.expert(flat, mapping.shared_expert)
            shared_gate = self.tensor(mapping.shared_expert_gate.name)
            output = output + torch.sigmoid(flat.float() @ shared_gate.float().t()) * shared.float()
        return output.reshape(original_shape).to(hidden_states.dtype)
