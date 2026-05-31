from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from reference_ops import (
    ReferenceWeights,
    TopKRouting,
    apply_rotary_pos_emb,
    gated_rms_norm,
    l2_norm,
    linear,
    rms_norm,
    rotary_embeddings,
    silu_mul,
    topk_route,
)
from decode_state import DecodeState, FullAttentionCache, LinearAttentionCache, batch_kv_tensors
from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping, LayerMapping, LinearTensor, MoEMapping, ShardedTensor


class TpRuntimeError(RuntimeError):
    pass


_NATIVE_MOE_EXPERT_MAX_GROUP_TOKENS = 8


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
    *,
    cache: FullAttentionCache | LinearAttentionCache | None = None,
    position_offset: int = 0,
) -> Any:
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm), config.rms_norm_eps)
    if mapping.layer_type == "full_attention":
        hidden_states = residual + tp_full_attention(hidden_states, mapping.attention, config, weights, runtime, cache=cache, position_offset=position_offset)
    elif mapping.layer_type == "linear_attention":
        hidden_states = residual + tp_linear_attention(hidden_states, mapping.attention, config, weights, runtime, cache=cache)
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
    local_logits = tp_language_model_local_logits(input_ids, mapping, config, weights, runtime)
    return runtime.all_gather_cat(local_logits, dim=-1)


def tp_language_model_local_logits(
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
    return weights.linear(hidden_states, LinearTensor(weight=mapping.lm_head, scale=None))


def tp_decode_step(
    input_ids: Any,
    mapping: LanguageModelMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    state: DecodeState,
) -> Any:
    local_logits = tp_decode_step_local_logits(input_ids, mapping, config, weights, runtime, state)
    return runtime.all_gather_cat(local_logits, dim=-1)


def tp_decode_step_local_logits(
    input_ids: Any,
    mapping: LanguageModelMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    state: DecodeState,
) -> Any:
    hidden_states = tp_embedding(input_ids, mapping, weights, runtime)
    position_offset = state.position_offset
    for layer, layer_cache in zip(mapping.layers, state.layers, strict=True):
        hidden_states = tp_decoder_layer(hidden_states, layer, config, weights, runtime, cache=layer_cache, position_offset=position_offset)
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm), config.rms_norm_eps)
    logits = weights.linear(hidden_states, LinearTensor(weight=mapping.lm_head, scale=None))
    state.advance(input_ids.shape[1])
    return logits


def tp_decode_step_batch(
    input_ids: Any,
    mapping: LanguageModelMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    states: Sequence[DecodeState],
) -> Any:
    """Batched decode step: input_ids (B, 1), one DecodeState per row.

    Runs a single forward pass for all B requests. Each request has its own
    position_offset and KV caches at potentially different fill levels.
    Returns local logits (B, 1, vocab_local).
    """
    import torch

    batch = len(states)
    if batch == 0:
        raise TpRuntimeError("tp_decode_step_batch requires at least one state")
    if batch == 1:
        # Single-request fast path: delegate to avoid overhead
        logits = tp_decode_step_local_logits(input_ids, mapping, config, weights, runtime, states[0])
        return logits

    hidden_states = tp_embedding(input_ids, mapping, weights, runtime)
    position_offsets = torch.tensor(
        [s.position_offset for s in states], device=hidden_states.device, dtype=torch.long
    )

    for layer_idx, layer in enumerate(mapping.layers):
        caches = [states[r].layers[layer_idx] for r in range(batch)]
        hidden_states = _tp_decoder_layer_batch(
            hidden_states, layer, config, weights, runtime,
            caches=caches, position_offsets=position_offsets,
        )

    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm), config.rms_norm_eps)
    logits = weights.linear(hidden_states, LinearTensor(weight=mapping.lm_head, scale=None))
    for s in states:
        s.advance(1)
    return logits


def tp_local_argmax(logits: Any, lm_head: ShardedTensor) -> tuple[Any, Any]:
    import torch

    local_values, local_indices = torch.max(logits.float(), dim=-1)
    shard = getattr(lm_head, "shard", None)
    start = getattr(shard, "start", 0) or 0
    return local_values, local_indices.to(torch.long) + start


def tp_greedy_next_token(logits: Any, lm_head: ShardedTensor, runtime: TpRuntime) -> Any:
    import torch

    local_values, local_tokens = tp_local_argmax(logits[:, -1], lm_head)
    if not runtime.config.is_distributed:
        return local_tokens
    packed = torch.stack((local_values.float(), local_tokens.float()), dim=-1)
    gathered = runtime.all_gather_cat(packed.unsqueeze(0), dim=0)
    best_rank = torch.argmax(gathered[..., 0], dim=0)
    batch_indices = torch.arange(gathered.shape[1], device=gathered.device)
    return gathered[best_rank, batch_indices, 1].to(torch.long)


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


def tp_full_attention(hidden_states: Any, mapping: Any, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime, *, cache: FullAttentionCache | None = None, position_offset: int = 0) -> Any:
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
    positions = torch.arange(position_offset, position_offset + seq_len, device=hidden_states.device).expand(batch, seq_len)
    cos, sin = rotary_embeddings(positions, config, device=hidden_states.device, dtype=query.dtype)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = _repeat_kv(key, full.num_heads // full.num_key_value_heads)
    value = _repeat_kv(value, full.num_heads // full.num_key_value_heads)
    head_start = runtime.config.rank * local_heads
    key = key[:, head_start : head_start + local_heads]
    value = value[:, head_start : head_start + local_heads]
    if cache is not None:
        key, value = cache.append(key, value)
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * (full.head_dim**-0.5)
    key_positions = torch.arange(key.shape[2], device=hidden_states.device)
    query_positions = torch.arange(position_offset, position_offset + seq_len, device=hidden_states.device)
    mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs, value).transpose(1, 2).reshape(batch, seq_len, local_heads * full.head_dim)
    out = out * torch.sigmoid(gate)
    partial = weights.linear(out, mapping.o_proj).to(hidden_states.dtype)
    return runtime.all_reduce_sum(partial)


def tp_linear_attention(hidden_states: Any, mapping: Any, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime, *, cache: LinearAttentionCache | None = None) -> Any:
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
    kernel = linear_cfg.conv_kernel_dim
    conv_input = mixed_qkv.float()
    if cache is None:
        mixed_qkv = F.conv1d(conv_input, conv_weight, padding=kernel - 1, groups=local_qkv_dim)[:, :, :seq_len]
    else:
        if cache.conv_tail is None:
            cache.conv_tail = torch.zeros(batch, local_qkv_dim, kernel - 1, device=conv_input.device, dtype=torch.float32)
        padded = torch.cat((cache.conv_tail, conv_input), dim=2)
        cache.conv_tail = padded[:, :, -(kernel - 1):] if kernel > 1 else cache.conv_tail
        mixed_qkv = F.conv1d(padded, conv_weight, padding=0, groups=local_qkv_dim)
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
    initial_state = None if cache is None else cache.state
    core = _recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=initial_state, return_state=cache is not None)
    if cache is not None:
        core, cache.state = core
    core = core.reshape(-1, linear_cfg.value_head_dim)
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
    stats = getattr(weights, "dispatch_stats", None)
    if stats is not None:
        stats.moe_calls += 1
        stats.moe_assignments += int(routing.indices.numel())
    if config.moe.packed_expert_dispatch:
        if stats is not None:
            stats.moe_packed_calls += 1
        routed = _tp_moe_packed_local_experts(flat, routing, mapping, config, weights)
    else:
        if stats is not None:
            stats.moe_loop_calls += 1
        routed = _tp_moe_per_expert_loop(flat, routing, mapping, weights)
    routed = runtime.all_reduce_sum(routed)
    shared = weights.expert(flat, mapping.shared_expert)
    shared_gate = weights.tensor(mapping.shared_expert_gate)
    output = routed + torch.sigmoid(flat.float() @ shared_gate.float().t()) * shared.float()
    return output.reshape(original_shape).to(hidden_states.dtype)


def _tp_moe_per_expert_loop(flat: Any, routing: TopKRouting, mapping: MoEMapping, weights: ReferenceWeights) -> Any:
    import torch

    routed = torch.zeros_like(flat.float())
    for expert in mapping.experts:
        token_indices, topk_indices = torch.where(routing.indices == expert.index)
        if token_indices.numel() == 0:
            continue
        token_output = weights.expert(flat[token_indices], expert)
        token_output = token_output * routing.scores[token_indices, topk_indices, None]
        routed.index_add_(0, token_indices, token_output.float())
    return routed


def _tp_moe_packed_local_experts(flat: Any, routing: TopKRouting, mapping: MoEMapping, config: RuntimeConfig, weights: ReferenceWeights) -> Any:
    import torch

    routed = torch.zeros_like(flat.float())
    stats = getattr(weights, "dispatch_stats", None)
    token_count = int(flat.shape[0])
    token_ids = torch.arange(token_count, device=flat.device)
    assignment_tokens = token_ids[:, None].expand_as(routing.indices).reshape(-1)
    assignment_experts = routing.indices.reshape(-1)
    assignment_scores = routing.scores.reshape(-1)
    local_mask = (assignment_experts >= mapping.expert_start) & (assignment_experts < mapping.expert_end)
    local_tokens = assignment_tokens[local_mask]
    local_assignment_count = int(local_tokens.numel())
    if stats is not None:
        stats.moe_local_assignments += local_assignment_count
    if local_assignment_count == 0:
        if stats is not None:
            stats.moe_empty_local_dispatches += 1
        return routed

    local_experts = assignment_experts[local_mask]
    local_scores = assignment_scores[local_mask]
    order = torch.argsort(local_experts)
    packed_tokens = local_tokens[order]
    packed_experts = local_experts[order]
    packed_scores = local_scores[order]
    packed_hidden = flat.index_select(0, packed_tokens)
    unique_experts, counts = torch.unique_consecutive(packed_experts, return_counts=True)
    if stats is not None:
        stats.moe_active_expert_groups += int(unique_experts.numel())
        if counts.numel() > 0:
            stats.moe_max_group_tokens = max(stats.moe_max_group_tokens, int(counts.max().item()))
    expert_by_index = {expert.index: expert for expert in mapping.experts}
    offset = 0
    for expert_id_tensor, count_tensor in zip(unique_experts, counts, strict=True):
        expert_id = int(expert_id_tensor.item())
        count = int(count_tensor.item())
        try:
            expert = expert_by_index[expert_id]
        except KeyError as exc:
            raise RuntimeError(f"routing selected unmapped local expert {expert_id}") from exc
        end = offset + count
        hidden_chunk = packed_hidden[offset:end]
        token_chunk = packed_tokens[offset:end]
        score_chunk = packed_scores[offset:end]
        token_output = _tp_moe_expert_group(hidden_chunk, expert, config, weights)
        token_output = token_output * score_chunk[:, None]
        routed.index_add_(0, token_chunk, token_output.float())
        offset = end
    return routed


def _tp_moe_expert_group(hidden_chunk: Any, expert: Any, config: RuntimeConfig, weights: ReferenceWeights) -> Any:
    stats = getattr(weights, "dispatch_stats", None)
    group_tokens = int(hidden_chunk.shape[0]) if hasattr(hidden_chunk, "shape") and len(hidden_chunk.shape) > 0 else 0
    if stats is not None:
        stats.moe_native_expert_calls += 1
        stats.moe_native_expert_max_group_tokens = max(stats.moe_native_expert_max_group_tokens, group_tokens)
    if not config.moe.native_fused_expert_dispatch:
        _record_native_expert_fallback(stats, "disabled")
        return weights.expert(hidden_chunk, expert)

    eligible, reason, tensors = _tp_moe_native_expert_eligibility(hidden_chunk, expert, weights)
    if not eligible:
        _record_native_expert_fallback(stats, reason)
        return weights.expert(hidden_chunk, expert)
    if stats is not None:
        stats.moe_native_expert_eligible += 1
    try:
        from fp8_cuda import fp8_e4m3_bf16_moe_expert

        output = fp8_e4m3_bf16_moe_expert(hidden_chunk.float(), *tensors)
    except RuntimeError:
        _record_native_expert_fallback(stats, "exception")
        return weights.expert(hidden_chunk, expert)
    if stats is not None:
        stats.moe_native_expert_hits += 1
    return output


def _tp_moe_native_expert_eligibility(hidden_chunk: Any, expert: Any, weights: ReferenceWeights) -> tuple[bool, str, tuple[Any, ...]]:
    import torch

    if len(getattr(hidden_chunk, "shape", ())) != 2:
        return False, "shape", ()
    group_tokens = int(hidden_chunk.shape[0])
    hidden_size = int(hidden_chunk.shape[1])
    if group_tokens < 1 or group_tokens > _NATIVE_MOE_EXPERT_MAX_GROUP_TOKENS:
        return False, "group_tokens", ()
    if not getattr(hidden_chunk, "is_cuda", False):
        return False, "device", ()
    gate_weight, gate_scale = weights.linear_weight(expert.gate_proj)
    up_weight, up_scale = weights.linear_weight(expert.up_proj)
    down_weight, down_scale = weights.linear_weight(expert.down_proj)
    tensors = (gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale)
    if any(tensor is None for tensor in (gate_scale, up_scale, down_scale)):
        return False, "missing_scale", ()
    if not all(getattr(tensor, "is_cuda", False) for tensor in tensors):
        return False, "device", ()
    if gate_weight.dtype != torch.float8_e4m3fn or up_weight.dtype != torch.float8_e4m3fn or down_weight.dtype != torch.float8_e4m3fn:
        return False, "dtype", ()
    if gate_scale.dtype != torch.bfloat16 or up_scale.dtype != torch.bfloat16 or down_scale.dtype != torch.bfloat16:
        return False, "dtype", ()
    if len(gate_weight.shape) != 2 or len(up_weight.shape) != 2 or len(down_weight.shape) != 2:
        return False, "shape", ()
    intermediate_size = int(gate_weight.shape[0])
    if int(gate_weight.shape[1]) != hidden_size:
        return False, "shape", ()
    if tuple(up_weight.shape) != tuple(gate_weight.shape):
        return False, "shape", ()
    if tuple(down_weight.shape) != (hidden_size, intermediate_size):
        return False, "shape", ()
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        return False, "shape", ()
    if tuple(gate_scale.shape) != (intermediate_size // 128, hidden_size // 128):
        return False, "shape", ()
    if tuple(up_scale.shape) != (intermediate_size // 128, hidden_size // 128):
        return False, "shape", ()
    if tuple(down_scale.shape) != (hidden_size // 128, intermediate_size // 128):
        return False, "shape", ()
    return True, "", tensors


def _record_native_expert_fallback(stats: Any, reason: str) -> None:
    if stats is None:
        return
    stats.moe_native_expert_fallbacks += 1
    if reason == "disabled":
        stats.moe_native_expert_fallback_disabled += 1
    elif reason == "missing_scale":
        stats.moe_native_expert_fallback_missing_scale += 1
    elif reason == "dtype":
        stats.moe_native_expert_fallback_dtype += 1
    elif reason == "device":
        stats.moe_native_expert_fallback_device += 1
    elif reason == "group_tokens":
        stats.moe_native_expert_fallback_group_tokens += 1
    elif reason == "exception":
        stats.moe_native_expert_fallback_exception += 1
    else:
        stats.moe_native_expert_fallback_shape += 1


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


def _tp_decoder_layer_batch(
    hidden_states: Any,
    mapping: LayerMapping,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    caches: list[FullAttentionCache | LinearAttentionCache],
    position_offsets: Any,
) -> Any:
    """Batched decoder layer: hidden_states (B, 1, D), per-row caches and positions."""
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm), config.rms_norm_eps)
    if mapping.layer_type == "full_attention":
        hidden_states = residual + _tp_full_attention_batch(
            hidden_states, mapping.attention, config, weights, runtime,
            caches=caches, position_offsets=position_offsets,
        )
    elif mapping.layer_type == "linear_attention":
        hidden_states = residual + _tp_linear_attention_batch(
            hidden_states, mapping.attention, config, weights, runtime,
            caches=caches,
        )
    else:
        raise ValueError(f"unsupported TP layer type: {mapping.layer_type}")
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm), config.rms_norm_eps)
    return residual + tp_moe(hidden_states, mapping.mlp, config, weights, runtime)


def _tp_full_attention_batch(
    hidden_states: Any,
    mapping: Any,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    caches: list[FullAttentionCache],
    position_offsets: Any,
) -> Any:
    """Batched full attention for decode: hidden_states (B, 1, D), per-row KV caches."""
    import torch

    batch, seq_len, _ = hidden_states.shape  # seq_len == 1 for decode
    full = config.full_attention
    local_heads = full.num_heads // runtime.config.world_size

    # Projections — shared computation across batch
    q_proj = weights.linear(hidden_states, mapping.q_proj).view(batch, seq_len, local_heads, full.head_dim * 2)
    query, gate = q_proj.chunk(2, dim=-1)
    gate = gate.reshape(batch, seq_len, local_heads * full.head_dim)
    key = weights.linear(hidden_states, mapping.k_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    value = weights.linear(hidden_states, mapping.v_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    query = rms_norm(query, weights.tensor(mapping.q_norm), config.rms_norm_eps).transpose(1, 2)
    key = rms_norm(key, weights.tensor(mapping.k_norm), config.rms_norm_eps).transpose(1, 2)
    value = value.transpose(1, 2)

    # Per-row rotary positions: position_offsets is (B,), seq_len=1
    positions = position_offsets.unsqueeze(1)  # (B, 1)
    cos, sin = rotary_embeddings(positions, config, device=hidden_states.device, dtype=query.dtype)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # KV head repeat and TP shard selection
    key = _repeat_kv(key, full.num_heads // full.num_key_value_heads)
    value = _repeat_kv(value, full.num_heads // full.num_key_value_heads)
    head_start = runtime.config.rank * local_heads
    key = key[:, head_start: head_start + local_heads]
    value = value[:, head_start: head_start + local_heads]

    # Per-request cache append: each row has batch=1 in its cache
    # key/value are (B, local_heads, 1, head_dim) — split per row
    for r in range(batch):
        caches[r].append(key[r: r + 1], value[r: r + 1])

    # Gather KV across requests with padding
    batched_keys, batched_values, valid_mask = batch_kv_tensors(caches)
    # batched_keys: (B, local_heads, max_valid, head_dim)
    # valid_mask: (B, max_valid)

    # Attention scores
    scores = torch.matmul(query.float(), batched_keys.float().transpose(2, 3)) * (full.head_dim ** -0.5)
    # scores: (B, local_heads, 1, max_valid)

    # Causal mask: for decode (seq_len=1), query position is position_offsets[r].
    # Key positions are 0..max_valid-1. Mask where key_pos > query_pos OR not valid.
    max_valid = batched_keys.shape[2]
    key_positions = torch.arange(max_valid, device=hidden_states.device)  # (max_valid,)
    query_pos = position_offsets.unsqueeze(1)  # (B, 1)
    causal_mask = key_positions.unsqueeze(0) > query_pos  # (B, max_valid)
    padding_mask = ~valid_mask  # (B, max_valid)
    combined_mask = (causal_mask | padding_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, max_valid)
    scores = scores.masked_fill(combined_mask, torch.finfo(scores.dtype).min)

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs, batched_values).transpose(1, 2).reshape(batch, seq_len, local_heads * full.head_dim)
    out = out * torch.sigmoid(gate)
    partial = weights.linear(out, mapping.o_proj).to(hidden_states.dtype)
    return runtime.all_reduce_sum(partial)


def _tp_linear_attention_batch(
    hidden_states: Any,
    mapping: Any,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    caches: list[LinearAttentionCache],
) -> Any:
    """Batched linear attention: stack per-request recurrent state, run, scatter back."""
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
    kernel = linear_cfg.conv_kernel_dim
    conv_input = mixed_qkv.float()

    # Per-request conv tail handling: stack tails into batch, run conv, scatter back
    # Each cache has conv_tail (1, local_qkv_dim, kernel-1) or None
    for r in range(batch):
        cache = caches[r]
        if cache.conv_tail is None:
            cache.conv_tail = torch.zeros(
                1, local_qkv_dim, kernel - 1, device=conv_input.device, dtype=torch.float32
            )

    # Stack conv tails: (B, local_qkv_dim, kernel-1)
    stacked_tails = torch.cat([cache.conv_tail for cache in caches], dim=0)
    padded = torch.cat((stacked_tails, conv_input), dim=2)

    # Update conv tails
    for r in range(batch):
        caches[r].conv_tail = padded[r: r + 1, :, -(kernel - 1):] if kernel > 1 else caches[r].conv_tail

    mixed_qkv = F.conv1d(padded, conv_weight, padding=0, groups=local_qkv_dim)
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

    # Stack recurrent states: each cache.state is (1, heads, key_dim, value_dim) or None
    initial_states = []
    for cache in caches:
        initial_states.append(cache.state)

    # If all None, pass None; otherwise stack with zeros for None entries
    if all(s is None for s in initial_states):
        stacked_initial = None
    else:
        parts = []
        for s in initial_states:
            if s is None:
                parts.append(torch.zeros(
                    1, local_value_heads, linear_cfg.key_head_dim, linear_cfg.value_head_dim,
                    device=hidden_states.device, dtype=torch.float32,
                ))
            else:
                parts.append(s)
        stacked_initial = torch.cat(parts, dim=0)

    core = _recurrent_gated_delta_rule(
        query, key, value, g, beta,
        initial_state=stacked_initial, return_state=True,
    )
    core, new_state = core

    # Scatter state back to per-request caches
    for r in range(batch):
        caches[r].state = new_state[r: r + 1]

    core = core.reshape(-1, linear_cfg.value_head_dim)
    z = weights.linear(hidden_states, mapping.in_proj_z).reshape(-1, linear_cfg.value_head_dim)
    core = gated_rms_norm(core, weights.tensor(mapping.norm), z, config.rms_norm_eps)
    core = core.reshape(batch, seq_len, local_value_dim)
    partial = weights.linear(core, mapping.out_proj).to(hidden_states.dtype)
    return runtime.all_reduce_sum(partial)


def _repeat_kv(hidden_states: Any, repeats: int) -> Any:
    if repeats == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


def _recurrent_gated_delta_rule(query: Any, key: Any, value: Any, g: Any, beta: Any, *, initial_state: Any = None, return_state: bool = False) -> Any:
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
    if initial_state is None:
        state = torch.zeros(batch, heads, key_dim, value_dim, device=query.device, dtype=torch.float32)
    else:
        state = initial_state.to(device=query.device, dtype=torch.float32)
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
    out = output.transpose(1, 2).contiguous().to(initial_dtype)
    if return_state:
        return out, state
    return out
