from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from reference_ops import (
    ReferenceWeights,
    apply_rotary_pos_emb,
    gated_rms_norm,
    l2_norm,
    linear,
    rms_norm,
    rotary_embeddings,
    silu_mul,
    topk_route,
)
from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping, LayerMapping, LinearTensor, MoEMapping


class TpRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TpLaunchConfig:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise TpRuntimeError(f"tp world size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise TpRuntimeError(f"tp rank {self.rank} out of range for world size {self.world_size}")
        if self.local_rank < 0:
            raise TpRuntimeError(f"tp local rank must be >= 0, got {self.local_rank}")
        if self.backend not in ("nccl", "gloo"):
            raise TpRuntimeError(f"unsupported tp backend: {self.backend}")

    @classmethod
    def from_env(
        cls,
        *,
        world_size: int | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        backend: str = "nccl",
        init_method: str | None = None,
        device: str | None = None,
    ) -> TpLaunchConfig:
        resolved_world = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
        resolved_rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
        resolved_local = local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", str(resolved_rank)))
        return cls(
            world_size=resolved_world,
            rank=resolved_rank,
            local_rank=resolved_local,
            backend=backend,
            init_method=init_method,
            device=device,
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def torch_device(self) -> Any:
        import torch

        if self.device is not None:
            return torch.device(self.device)
        if self.backend == "nccl":
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")


class TpRuntime:
    def __init__(self, config: TpLaunchConfig) -> None:
        self.config = config
        self.device: Any | None = None
        self._owns_process_group = False

    def __enter__(self) -> TpRuntime:
        import torch
        import torch.distributed as dist

        self.device = self.config.torch_device()
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise TpRuntimeError("CUDA is required for NCCL TP runtime")
            torch.cuda.set_device(self.device)
        if self.config.is_distributed:
            if not dist.is_available():
                raise TpRuntimeError("torch.distributed is not available")
            if self.config.backend == "nccl" and not dist.is_nccl_available():
                raise TpRuntimeError("torch.distributed NCCL backend is not available")
            if self.config.backend == "gloo" and not dist.is_gloo_available():
                raise TpRuntimeError("torch.distributed gloo backend is not available")
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method or "env://",
                    rank=self.config.rank,
                    world_size=self.config.world_size,
                )
                self._owns_process_group = True
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._owns_process_group:
            return
        import torch.distributed as dist

        dist.destroy_process_group()
        self._owns_process_group = False

    def all_reduce_sum(self, tensor: Any) -> Any:
        if not self.config.is_distributed:
            return tensor
        import torch.distributed as dist

        if self.config.backend == "nccl" and tensor.device.type != "cuda":
            raise TpRuntimeError("NCCL all-reduce requires a CUDA tensor")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather_cat(self, tensor: Any, dim: int = -1) -> Any:
        if not self.config.is_distributed:
            return tensor
        import torch
        import torch.distributed as dist

        outputs = [torch.empty_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim=dim)

    def barrier(self) -> None:
        if not self.config.is_distributed:
            return
        import torch.distributed as dist

        dist.barrier()


def tp_decoder_layer(
    hidden_states: Any,
    mapping: LayerMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
) -> Any:
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm), config.rms_norm_eps)
    if mapping.layer_type == "full_attention":
        hidden_states = residual + tp_full_attention(hidden_states, mapping.attention, config, weights, runtime)
    elif mapping.layer_type == "linear_attention":
        hidden_states = residual + tp_linear_attention(hidden_states, mapping.attention, config, weights, runtime)
    else:
        raise ValueError(f"unsupported TP layer type: {mapping.layer_type}")
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm), config.rms_norm_eps)
    return residual + tp_moe(hidden_states, mapping.mlp, config, weights, runtime)


def tp_language_model(
    input_ids: Any,
    mapping: LanguageModelMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
) -> Any:
    hidden_states = tp_embedding(input_ids, mapping, weights, runtime)
    for layer in mapping.layers:
        hidden_states = tp_decoder_layer(hidden_states, layer, config, weights, runtime)
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm), config.rms_norm_eps)
    logits = weights.linear(hidden_states, LinearTensor(weight=mapping.lm_head, scale=None))
    return runtime.all_gather_cat(logits, dim=-1)


def tp_embedding(input_ids: Any, mapping: LanguageModelMapping, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    import torch
    import torch.nn.functional as F

    weight = weights.tensor(mapping.embed_tokens)
    shard = getattr(mapping.embed_tokens, "shard", None)
    start = getattr(shard, "start", 0) or 0
    size = weight.shape[0]
    local_ids = input_ids - start
    mask = (input_ids < start) | (input_ids >= start + size)
    local_ids = local_ids.masked_fill(mask, 0)
    out = F.embedding(local_ids, weight)
    out = out.masked_fill(mask.unsqueeze(-1), 0)
    return runtime.all_reduce_sum(out)


def tp_full_attention(hidden_states: Any, mapping: Any, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    import torch

    batch, seq_len, _ = hidden_states.shape
    full = config.full_attention
    local_heads = full.num_heads // runtime.config.world_size
    q_proj = weights.linear(hidden_states, mapping.q_proj).view(batch, seq_len, local_heads, full.head_dim * 2)
    query, gate = q_proj.chunk(2, dim=-1)
    gate = gate.reshape(batch, seq_len, local_heads * full.head_dim)
    key = weights.linear(hidden_states, mapping.k_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    value = weights.linear(hidden_states, mapping.v_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    query = rms_norm(query, weights.tensor(mapping.q_norm), config.rms_norm_eps).transpose(1, 2)
    key = rms_norm(key, weights.tensor(mapping.k_norm), config.rms_norm_eps).transpose(1, 2)
    value = value.transpose(1, 2)
    positions = torch.arange(seq_len, device=hidden_states.device).expand(batch, seq_len)
    cos, sin = rotary_embeddings(positions, config, device=hidden_states.device, dtype=query.dtype)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = _repeat_kv(key, full.num_heads // full.num_key_value_heads)
    value = _repeat_kv(value, full.num_heads // full.num_key_value_heads)
    head_start = runtime.config.rank * local_heads
    key = key[:, head_start : head_start + local_heads]
    value = value[:, head_start : head_start + local_heads]
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * (full.head_dim**-0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs, value).transpose(1, 2).reshape(batch, seq_len, local_heads * full.head_dim)
    out = out * torch.sigmoid(gate)
    partial = weights.linear(out, mapping.o_proj).to(hidden_states.dtype)
    return runtime.all_reduce_sum(partial)


def tp_linear_attention(hidden_states: Any, mapping: Any, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    import torch
    import torch.nn.functional as F

    batch, seq_len, _ = hidden_states.shape
    linear_cfg = config.linear_attention
    world = runtime.config.world_size
    local_key_heads = linear_cfg.key_heads // world
    local_value_heads = linear_cfg.value_heads // world
    local_key_dim = local_key_heads * linear_cfg.key_head_dim
    local_value_dim = local_value_heads * linear_cfg.value_head_dim
    local_qkv_dim = local_key_dim * 2 + local_value_dim
    mixed_qkv = weights.linear(hidden_states, mapping.in_proj_qkv).transpose(1, 2)
    conv_weight = weights.tensor(mapping.conv1d_weight).float()
    mixed_qkv = F.conv1d(
        mixed_qkv.float(),
        conv_weight,
        padding=linear_cfg.conv_kernel_dim - 1,
        groups=local_qkv_dim,
    )[:, :, :seq_len]
    mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1)
    query = query.reshape(batch, seq_len, local_key_heads, linear_cfg.key_head_dim)
    key = key.reshape(batch, seq_len, local_key_heads, linear_cfg.key_head_dim)
    value = value.reshape(batch, seq_len, local_value_heads, linear_cfg.value_head_dim)
    beta = torch.sigmoid(weights.linear(hidden_states, mapping.in_proj_b))
    a = weights.linear(hidden_states, mapping.in_proj_a)
    a_log = weights.tensor(mapping.a_log).float()
    dt_bias = weights.tensor(mapping.dt_bias).float()
    g = -a_log.exp() * F.softplus(a.float() + dt_bias)
    repeats = local_value_heads // local_key_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    core = _recurrent_gated_delta_rule(query, key, value, g, beta).reshape(-1, linear_cfg.value_head_dim)
    z = weights.linear(hidden_states, mapping.in_proj_z).reshape(-1, linear_cfg.value_head_dim)
    core = gated_rms_norm(core, weights.tensor(mapping.norm), z, config.rms_norm_eps)
    core = core.reshape(batch, seq_len, local_value_dim)
    partial = weights.linear(core, mapping.out_proj).to(hidden_states.dtype)
    return runtime.all_reduce_sum(partial)


def tp_moe(hidden_states: Any, mapping: MoEMapping, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    import torch

    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, original_shape[-1])
    routing = topk_route(flat, weights.tensor(mapping.gate), config.moe.experts_per_token)
    routed = torch.zeros_like(flat.float())
    for expert in mapping.experts:
        token_indices, topk_indices = torch.where(routing.indices == expert.index)
        if token_indices.numel() == 0:
            continue
        token_output = weights.expert(flat[token_indices], expert)
        token_output = token_output * routing.scores[token_indices, topk_indices, None]
        routed.index_add_(0, token_indices, token_output.float())
    routed = runtime.all_reduce_sum(routed)
    shared = weights.expert(flat, mapping.shared_expert)
    shared_gate = weights.tensor(mapping.shared_expert_gate)
    output = routed + torch.sigmoid(flat.float() @ shared_gate.float().t()) * shared.float()
    return output.reshape(original_shape).to(hidden_states.dtype)


def mapped_tensor_bytes(mapping: LanguageModelMapping) -> int:
    return sum(tensor.nbytes for tensor in _mapped_tensor_infos(mapping))


def _mapped_tensor_infos(mapping: LanguageModelMapping):
    tensors = [mapping.embed_tokens, mapping.final_norm, mapping.lm_head]
    for layer in mapping.layers:
        tensors.extend((layer.input_layernorm, layer.post_attention_layernorm))
        attention = layer.attention
        if layer.layer_type == "full_attention":
            tensors.extend((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight, attention.o_proj.weight, attention.q_norm, attention.k_norm))
            tensors.extend(t for t in (attention.q_proj.scale, attention.k_proj.scale, attention.v_proj.scale, attention.o_proj.scale) if t is not None)
        else:
            tensors.extend((attention.in_proj_qkv.weight, attention.in_proj_z.weight, attention.out_proj.weight, attention.in_proj_a.weight, attention.in_proj_b.weight, attention.conv1d_weight, attention.a_log, attention.dt_bias, attention.norm))
            tensors.extend(t for t in (attention.in_proj_qkv.scale, attention.in_proj_z.scale, attention.out_proj.scale) if t is not None)
        moe = layer.mlp
        tensors.extend((moe.gate, moe.shared_expert_gate))
        for expert in (*moe.experts, moe.shared_expert):
            for linear_tensor in (expert.gate_proj, expert.up_proj, expert.down_proj):
                tensors.append(linear_tensor.weight)
                if linear_tensor.scale is not None:
                    tensors.append(linear_tensor.scale)
    return tensors


def _repeat_kv(hidden_states: Any, repeats: int) -> Any:
    if repeats == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


def _recurrent_gated_delta_rule(query: Any, key: Any, value: Any, g: Any, beta: Any) -> Any:
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
