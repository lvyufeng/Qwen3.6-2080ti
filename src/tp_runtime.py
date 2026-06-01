from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

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
from decode_state import DecodeState, FullAttentionCache, LinearAttentionCache, batch_kv_tensors, batch_page_tables
from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping, LayerMapping, LinearTensor, MoEMapping, ShardedTensor


class TpRuntimeError(RuntimeError):
    pass


_NATIVE_MOE_EXPERT_MAX_GROUP_TOKENS = 8
_NATIVE_MOE_ASSIGNMENT_MIN_ASSIGNMENTS = 4096


@dataclass(frozen=True)
class RuntimeProfileConfig:
    enabled: bool = False
    sync_cuda: bool = False
    layer_detail: bool = False
    collective_detail: bool = True


@dataclass
class ProfileScopeStats:
    calls: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    min_seconds: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    bytes: int = 0

    def record(self, seconds: float, *, input_tokens: int = 0, output_tokens: int = 0, bytes: int = 0) -> None:
        self.calls += 1
        self.total_seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)
        self.min_seconds = seconds if self.min_seconds is None else min(self.min_seconds, seconds)
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.bytes += int(bytes)

    def to_dict(self) -> dict[str, int | float | None]:
        avg_seconds = self.total_seconds / self.calls if self.calls else 0.0
        return {
            "calls": self.calls,
            "total_seconds": self.total_seconds,
            "avg_seconds": avg_seconds,
            "max_seconds": self.max_seconds,
            "min_seconds": self.min_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "bytes": self.bytes,
        }


@dataclass
class PagedAttentionDispatchStats:
    calls: int = 0
    eligible: int = 0
    dense_fallbacks: int = 0
    native_hits: int = 0
    fallback_disabled: int = 0
    fallback_no_kernel: int = 0
    fallback_prefill_or_multitoken: int = 0
    fallback_cpu: int = 0
    fallback_per_request_pools: int = 0
    fallback_shape: int = 0

    def snapshot(self) -> PagedAttentionDispatchStats:
        return PagedAttentionDispatchStats(**vars(self))

    def to_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class RuntimeProfileStats:
    enabled: bool = False
    sync_cuda: bool = False
    scopes: dict[str, ProfileScopeStats] = field(default_factory=dict)

    def record(self, name: str, seconds: float, *, input_tokens: int = 0, output_tokens: int = 0, bytes: int = 0) -> None:
        scope = self.scopes.setdefault(name, ProfileScopeStats())
        scope.record(seconds, input_tokens=input_tokens, output_tokens=output_tokens, bytes=bytes)

    def reset(self) -> None:
        self.scopes.clear()

    def snapshot(self) -> RuntimeProfileStats:
        return RuntimeProfileStats(
            enabled=self.enabled,
            sync_cuda=self.sync_cuda,
            scopes={name: ProfileScopeStats(**vars(scope)) for name, scope in self.scopes.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sync_cuda": self.sync_cuda,
            "scopes": {name: scope.to_dict() for name, scope in sorted(self.scopes.items())},
        }


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
        self.profile_config = RuntimeProfileConfig()
        self.profile_stats = RuntimeProfileStats()
        self.paged_attention_stats = PagedAttentionDispatchStats()

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

    def configure_profiling(self, config: RuntimeProfileConfig | None = None) -> None:
        self.profile_config = config or RuntimeProfileConfig()
        self.profile_stats = RuntimeProfileStats(
            enabled=self.profile_config.enabled,
            sync_cuda=self.profile_config.sync_cuda,
        )
        self.paged_attention_stats = PagedAttentionDispatchStats()

    def reset_profile(self) -> None:
        self.profile_stats = RuntimeProfileStats(
            enabled=self.profile_config.enabled,
            sync_cuda=self.profile_config.sync_cuda,
        )
        self.paged_attention_stats = PagedAttentionDispatchStats()

    @contextmanager
    def profile_scope(
        self,
        name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        bytes: int = 0,
    ) -> Iterator[None]:
        if not self.profile_config.enabled:
            yield
            return
        self._profile_sync_device()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._profile_sync_device()
            self.profile_stats.record(
                name,
                max(0.0, time.perf_counter() - start),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                bytes=bytes,
            )

    def _profile_sync_device(self) -> None:
        if not self.profile_config.sync_cuda or getattr(self.device, "type", None) != "cuda":
            return
        import torch

        torch.cuda.synchronize(self.device)

    def all_reduce_sum(self, tensor: Any) -> Any:
        byte_count = _tensor_bytes(tensor) if self.profile_config.collective_detail else 0
        with self.profile_scope("collective.all_reduce_sum", bytes=byte_count):
            if not self.config.is_distributed:
                return tensor
            import torch.distributed as dist

            if self.config.backend == "nccl" and tensor.device.type != "cuda":
                raise TpRuntimeError("NCCL all-reduce requires a CUDA tensor")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor

    def all_gather_cat(self, tensor: Any, dim: int = -1) -> Any:
        byte_count = _tensor_bytes(tensor) if self.profile_config.collective_detail else 0
        with self.profile_scope("collective.all_gather_cat", bytes=byte_count):
            if not self.config.is_distributed:
                return tensor
            import torch
            import torch.distributed as dist

            outputs = [torch.empty_like(tensor) for _ in range(self.config.world_size)]
            dist.all_gather(outputs, tensor)
            return torch.cat(outputs, dim=dim)

    def barrier(self) -> None:
        with self.profile_scope("collective.barrier"):
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
    layer_scope = _layer_scope(runtime, mapping.index, "total")
    with runtime.profile_scope(layer_scope):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm), config.rms_norm_eps)
        if mapping.layer_type == "full_attention":
            with runtime.profile_scope(_layer_scope(runtime, mapping.index, "full_attention.total")):
                hidden_states = residual + tp_full_attention(
                    hidden_states, mapping.attention, config, weights, runtime, cache=cache, position_offset=position_offset
                )
        elif mapping.layer_type == "linear_attention":
            with runtime.profile_scope(_layer_scope(runtime, mapping.index, "linear_attention.total")):
                hidden_states = residual + tp_linear_attention(hidden_states, mapping.attention, config, weights, runtime, cache=cache)
        else:
            raise ValueError(f"unsupported TP layer type: {mapping.layer_type}")
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm), config.rms_norm_eps)
        with runtime.profile_scope(_layer_scope(runtime, mapping.index, "moe.total")):
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
    with runtime.profile_scope("embedding", input_tokens=int(input_ids.numel())):
        hidden_states = tp_embedding(input_ids, mapping, weights, runtime)
    position_offset = state.position_offset
    with runtime.profile_scope("layers_total"):
        for layer, layer_cache in zip(mapping.layers, state.layers, strict=True):
            hidden_states = tp_decoder_layer(hidden_states, layer, config, weights, runtime, cache=layer_cache, position_offset=position_offset)
    with runtime.profile_scope("final_norm"):
        hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm), config.rms_norm_eps)
    with runtime.profile_scope("lm_head"):
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

    with runtime.profile_scope("batch.embedding", input_tokens=int(input_ids.numel())):
        hidden_states = tp_embedding(input_ids, mapping, weights, runtime)
    position_offsets = torch.tensor(
        [s.position_offset for s in states], device=hidden_states.device, dtype=torch.long
    )

    with runtime.profile_scope("batch.layers_total"):
        for layer_idx, layer in enumerate(mapping.layers):
            caches = [states[r].layers[layer_idx] for r in range(batch)]
            hidden_states = _tp_decoder_layer_batch(
                hidden_states, layer, config, weights, runtime,
                caches=caches, position_offsets=position_offsets,
            )

    with runtime.profile_scope("batch.final_norm"):
        hidden_states = rms_norm(hidden_states, weights.tensor(mapping.final_norm), config.rms_norm_eps)
    with runtime.profile_scope("batch.lm_head"):
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


def tp_greedy_next_tokens(logits: Any, lm_head: ShardedTensor, runtime: TpRuntime) -> Any:
    import torch

    local_values, local_tokens = tp_local_argmax(logits[:, -1], lm_head)
    if not runtime.config.is_distributed:
        return local_tokens
    packed = torch.stack((local_values.float(), local_tokens.float()), dim=-1)
    gathered = runtime.all_gather_cat(packed.unsqueeze(0), dim=0)
    best_rank = torch.argmax(gathered[..., 0], dim=0)
    batch_indices = torch.arange(gathered.shape[1], device=gathered.device)
    return gathered[best_rank, batch_indices, 1].to(torch.long)


def tp_greedy_next_token(logits: Any, lm_head: ShardedTensor, runtime: TpRuntime) -> Any:
    return tp_greedy_next_tokens(logits, lm_head, runtime)


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
    with runtime.profile_scope("full_attention.qkv_proj"):
        q_proj = weights.linear(hidden_states, mapping.q_proj).view(batch, seq_len, local_heads, full.head_dim * 2)
        query, gate = q_proj.chunk(2, dim=-1)
        gate = gate.reshape(batch, seq_len, local_heads * full.head_dim)
        key = weights.linear(hidden_states, mapping.k_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
        value = weights.linear(hidden_states, mapping.v_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    with runtime.profile_scope("full_attention.norm_rope"):
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
        with runtime.profile_scope("full_attention.cache_append"):
            key, value = cache.append(key, value)
        return _tp_full_attention_paged_or_dense(
            query,
            gate,
            key,
            value,
            hidden_states,
            mapping,
            config,
            weights,
            runtime,
            cache=cache,
            position_offset=position_offset,
        )
    return _tp_full_attention_dense_from_kv(
        query,
        gate,
        key,
        value,
        hidden_states,
        mapping,
        config,
        weights,
        runtime,
        position_offset=position_offset,
    )


def _tp_full_attention_paged_or_dense(
    query: Any,
    gate: Any,
    key: Any,
    value: Any,
    hidden_states: Any,
    mapping: Any,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    cache: FullAttentionCache,
    position_offset: int,
) -> Any:
    if config.full_attention.paged_kv_metadata:
        with runtime.profile_scope("full_attention.page_metadata"):
            cache.page_metadata()
    with runtime.profile_scope("full_attention.paged_dispatch"):
        _record_paged_attention_dispatch(
            runtime,
            config,
            query,
            key,
            value,
            seq_len=int(query.shape[2]),
            batched=False,
        )
    with runtime.profile_scope("full_attention.dense_fallback"):
        return _tp_full_attention_dense_from_kv(
            query,
            gate,
            key,
            value,
            hidden_states,
            mapping,
            config,
            weights,
            runtime,
            position_offset=position_offset,
        )


def _tp_full_attention_dense_from_kv(
    query: Any,
    gate: Any,
    key: Any,
    value: Any,
    hidden_states: Any,
    mapping: Any,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    position_offset: int,
) -> Any:
    import torch

    batch, seq_len, _ = hidden_states.shape
    full = config.full_attention
    local_heads = full.num_heads // runtime.config.world_size
    with runtime.profile_scope("full_attention.scores"):
        scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * (full.head_dim**-0.5)
        key_positions = torch.arange(key.shape[2], device=hidden_states.device)
        query_positions = torch.arange(position_offset, position_offset + seq_len, device=hidden_states.device)
        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    with runtime.profile_scope("full_attention.softmax"):
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    with runtime.profile_scope("full_attention.value"):
        out = torch.matmul(probs, value).transpose(1, 2).reshape(batch, seq_len, local_heads * full.head_dim)
        out = out * torch.sigmoid(gate)
    with runtime.profile_scope("full_attention.output_proj"):
        partial = weights.linear(out, mapping.o_proj).to(hidden_states.dtype)
    with runtime.profile_scope("full_attention.all_reduce"):
        return runtime.all_reduce_sum(partial)


def _record_paged_attention_dispatch(
    runtime: TpRuntime,
    config: RuntimeConfig,
    query: Any,
    key: Any,
    value: Any,
    *,
    seq_len: int,
    batched: bool,
) -> None:
    stats = runtime.paged_attention_stats
    stats.calls += 1
    reason = _paged_attention_fallback_reason(config, query, key, value, seq_len=seq_len, batched=batched)
    if reason == "no_kernel":
        stats.eligible += 1
    stats.dense_fallbacks += 1
    if reason == "disabled":
        stats.fallback_disabled += 1
    elif reason == "cpu":
        stats.fallback_cpu += 1
    elif reason == "prefill_or_multitoken":
        stats.fallback_prefill_or_multitoken += 1
    elif reason == "per_request_pools":
        stats.fallback_per_request_pools += 1
    elif reason == "shape":
        stats.fallback_shape += 1
    else:
        stats.fallback_no_kernel += 1


def _paged_attention_fallback_reason(
    config: RuntimeConfig,
    query: Any,
    key: Any,
    value: Any,
    *,
    seq_len: int,
    batched: bool,
) -> str:
    if not config.full_attention.paged_kv_metadata or not config.full_attention.native_paged_attention:
        return "disabled"
    if not getattr(query, "is_cuda", False) or not getattr(key, "is_cuda", False) or not getattr(value, "is_cuda", False):
        return "cpu"
    if seq_len != 1:
        return "prefill_or_multitoken"
    if batched:
        return "per_request_pools"
    if len(getattr(query, "shape", ())) != 4 or len(getattr(key, "shape", ())) != 4 or len(getattr(value, "shape", ())) != 4:
        return "shape"
    if int(query.shape[0]) != int(key.shape[0]) or int(query.shape[0]) != int(value.shape[0]):
        return "shape"
    if int(query.shape[1]) != int(key.shape[1]) or int(query.shape[1]) != int(value.shape[1]):
        return "shape"
    if int(query.shape[3]) != int(key.shape[3]) or int(query.shape[3]) != int(value.shape[3]):
        return "shape"
    return "no_kernel"


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
    with runtime.profile_scope("linear_attention.qkv_conv"):
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
    with runtime.profile_scope("linear_attention.projections"):
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
    with runtime.profile_scope("linear_attention.recurrent_core"):
        core = _recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=initial_state, return_state=cache is not None)
    if cache is not None:
        core, cache.state = core
    with runtime.profile_scope("linear_attention.output_norm"):
        core = core.reshape(-1, linear_cfg.value_head_dim)
        z = weights.linear(hidden_states, mapping.in_proj_z).reshape(-1, linear_cfg.value_head_dim)
        core = gated_rms_norm(core, weights.tensor(mapping.norm), z, config.rms_norm_eps)
        core = core.reshape(batch, seq_len, local_value_dim)
    with runtime.profile_scope("linear_attention.output_proj"):
        partial = weights.linear(core, mapping.out_proj).to(hidden_states.dtype)
    with runtime.profile_scope("linear_attention.all_reduce"):
        return runtime.all_reduce_sum(partial)


def tp_moe(hidden_states: Any, mapping: MoEMapping, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    import torch

    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, original_shape[-1])
    with runtime.profile_scope("moe.route"):
        routing = topk_route(flat, weights.tensor(mapping.gate), config.moe.experts_per_token)
    stats = getattr(weights, "dispatch_stats", None)
    if stats is not None:
        stats.moe_calls += 1
        stats.moe_assignments += int(routing.indices.numel())
    with runtime.profile_scope("moe.dispatch"):
        if config.moe.packed_expert_dispatch:
            if stats is not None:
                stats.moe_packed_calls += 1
            routed = _tp_moe_packed_local_experts(flat, routing, mapping, config, weights, runtime)
        else:
            if stats is not None:
                stats.moe_loop_calls += 1
            routed = _tp_moe_per_expert_loop(flat, routing, mapping, weights)
    with runtime.profile_scope("moe.all_reduce"):
        routed = runtime.all_reduce_sum(routed)
    with runtime.profile_scope("moe.shared_expert"):
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


def _tp_moe_packed_local_experts(flat: Any, routing: TopKRouting, mapping: MoEMapping, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    import torch

    routed = torch.zeros_like(flat.float())
    stats = getattr(weights, "dispatch_stats", None)
    local_assignment_count, packed_tokens, packed_scores, unique_experts, counts = _tp_moe_packed_assignment_plan(
        routing,
        mapping,
        runtime,
        stats,
    )
    if stats is not None:
        stats.moe_local_assignments += local_assignment_count
    if local_assignment_count == 0:
        if stats is not None:
            stats.moe_empty_local_dispatches += 1
        return routed

    with runtime.profile_scope("moe.packed.gather_hidden"):
        packed_hidden = flat.index_select(0, packed_tokens)
    unique_expert_ids = unique_experts.tolist()
    group_counts = counts.tolist()
    if stats is not None:
        stats.moe_active_expert_groups += len(unique_expert_ids)
        if group_counts:
            stats.moe_max_group_tokens = max(stats.moe_max_group_tokens, max(group_counts))
            for count in group_counts:
                if count == 1:
                    stats.moe_group_size_1 += 1
                elif count <= 4:
                    stats.moe_group_size_2_to_4 += 1
                elif count <= 8:
                    stats.moe_group_size_5_to_8 += 1
                else:
                    stats.moe_group_size_over_8 += 1
    expert_by_index = _tp_moe_expert_by_index(mapping)
    packed_output = torch.empty((local_assignment_count, flat.shape[-1]), device=flat.device, dtype=torch.float32)
    offset = 0
    with runtime.profile_scope("moe.packed.group_loop"):
        for expert_id, count in zip(unique_expert_ids, group_counts, strict=True):
            try:
                expert = expert_by_index[expert_id]
            except KeyError as exc:
                raise RuntimeError(f"routing selected unmapped local expert {expert_id}") from exc
            end = offset + count
            hidden_chunk = packed_hidden[offset:end]
            score_chunk = packed_scores[offset:end]
            token_output = _tp_moe_expert_group(hidden_chunk, expert, config, weights, runtime)
            with runtime.profile_scope("moe.packed.score_apply"):
                packed_output[offset:end] = token_output.float() * score_chunk[:, None]
            offset = end
    with runtime.profile_scope("moe.packed.scatter"):
        routed.index_add_(0, packed_tokens, packed_output)
    if stats is not None:
        stats.moe_packed_index_add_calls += 1
        stats.moe_packed_single_scatter_calls += 1
    return routed


def _tp_moe_packed_assignment_plan(
    routing: TopKRouting,
    mapping: MoEMapping,
    runtime: TpRuntime,
    stats: Any,
) -> tuple[int, Any, Any, Any, Any]:
    eligible, reason = _tp_moe_native_assignment_eligibility(routing, mapping)
    if stats is not None:
        stats.moe_native_assignment_calls += 1
    if eligible:
        if stats is not None:
            stats.moe_native_assignment_eligible += 1
        native_fn = _native_moe_assignment_fn()
        if native_fn is not None:
            try:
                with runtime.profile_scope("moe.packed.local_plan"):
                    packed_tokens, packed_scores, unique_experts, counts = native_fn(
                        routing.indices,
                        routing.scores,
                        mapping.expert_start,
                        mapping.expert_end,
                    )
                if stats is not None:
                    stats.moe_native_assignment_hits += 1
                return int(packed_tokens.numel()), packed_tokens, packed_scores, unique_experts, counts
            except RuntimeError:
                _record_native_assignment_fallback(stats, "exception")
        else:
            _record_native_assignment_fallback(stats, "exception")
    else:
        _record_native_assignment_fallback(stats, reason)
    return _tp_moe_packed_assignment_plan_torch(routing, mapping, runtime)


def _tp_moe_packed_assignment_plan_torch(routing: TopKRouting, mapping: MoEMapping, runtime: TpRuntime) -> tuple[int, Any, Any, Any, Any]:
    import torch

    with runtime.profile_scope("moe.packed.assignments"):
        token_count = int(routing.indices.shape[0])
        token_ids = torch.arange(token_count, device=routing.indices.device)
        assignment_tokens = token_ids[:, None].expand_as(routing.indices).reshape(-1)
        assignment_experts = routing.indices.reshape(-1)
        assignment_scores = routing.scores.reshape(-1)
    with runtime.profile_scope("moe.packed.local_filter"):
        local_mask = (assignment_experts >= mapping.expert_start) & (assignment_experts < mapping.expert_end)
        local_tokens = assignment_tokens[local_mask]
        local_assignment_count = int(local_tokens.numel())
    if local_assignment_count == 0:
        empty_experts = routing.indices.new_empty((0,))
        empty_counts = routing.indices.new_empty((0,))
        empty_scores = routing.scores.new_empty((0,))
        return 0, local_tokens, empty_scores, empty_experts, empty_counts

    local_experts = assignment_experts[local_mask]
    local_scores = assignment_scores[local_mask]
    with runtime.profile_scope("moe.packed.sort"):
        order = torch.argsort(local_experts)
        packed_tokens = local_tokens[order]
        packed_experts = local_experts[order]
        packed_scores = local_scores[order]
        unique_experts, counts = torch.unique_consecutive(packed_experts, return_counts=True)
    return local_assignment_count, packed_tokens, packed_scores, unique_experts, counts


def _tp_moe_native_assignment_eligibility(routing: TopKRouting, mapping: MoEMapping) -> tuple[bool, str]:
    import torch

    indices = routing.indices
    scores = routing.scores
    if not getattr(indices, "is_cuda", False) or not getattr(scores, "is_cuda", False):
        return False, "device"
    if getattr(indices, "dtype", None) != torch.long or getattr(scores, "dtype", None) != torch.float32:
        return False, "dtype"
    if len(getattr(indices, "shape", ())) != 2 or len(getattr(scores, "shape", ())) != 2:
        return False, "shape"
    if tuple(indices.shape) != tuple(scores.shape):
        return False, "shape"
    if int(mapping.expert_end) <= int(mapping.expert_start):
        return False, "shape"
    if int(indices.numel()) < _NATIVE_MOE_ASSIGNMENT_MIN_ASSIGNMENTS:
        return False, "small"
    return True, ""


def _record_native_assignment_fallback(stats: Any, reason: str) -> None:
    if stats is None:
        return
    stats.moe_native_assignment_fallbacks += 1
    if reason == "small":
        stats.moe_native_assignment_fallback_small += 1
    elif reason == "device":
        stats.moe_native_assignment_fallback_device += 1
    elif reason == "dtype":
        stats.moe_native_assignment_fallback_dtype += 1
    elif reason == "exception":
        stats.moe_native_assignment_fallback_exception += 1
    else:
        stats.moe_native_assignment_fallback_shape += 1


def _tp_moe_expert_by_index(mapping: MoEMapping) -> dict[int, Any]:
    cached = getattr(mapping, "_expert_by_index_cache", None)
    if cached is not None:
        return cached
    expert_by_index = {expert.index: expert for expert in mapping.experts}
    try:
        object.__setattr__(mapping, "_expert_by_index_cache", expert_by_index)
    except (AttributeError, TypeError):
        pass
    return expert_by_index


_NATIVE_MOE_ASSIGNMENT_FN: Any = None
_NATIVE_MOE_ASSIGNMENT_IMPORT_FAILED = False


def _native_moe_assignment_fn() -> Any:
    global _NATIVE_MOE_ASSIGNMENT_FN, _NATIVE_MOE_ASSIGNMENT_IMPORT_FAILED
    if _NATIVE_MOE_ASSIGNMENT_FN is not None:
        return _NATIVE_MOE_ASSIGNMENT_FN
    if _NATIVE_MOE_ASSIGNMENT_IMPORT_FAILED:
        return None
    try:
        from fp8_cuda import moe_packed_local_assignments
    except Exception:
        _NATIVE_MOE_ASSIGNMENT_IMPORT_FAILED = True
        return None
    _NATIVE_MOE_ASSIGNMENT_FN = moe_packed_local_assignments
    return _NATIVE_MOE_ASSIGNMENT_FN


_NATIVE_MOE_EXPERT_FN: Any = None
_NATIVE_MOE_EXPERT_IMPORT_FAILED = False


def _native_moe_expert_fn() -> Any:
    global _NATIVE_MOE_EXPERT_FN, _NATIVE_MOE_EXPERT_IMPORT_FAILED
    if _NATIVE_MOE_EXPERT_FN is not None:
        return _NATIVE_MOE_EXPERT_FN
    if _NATIVE_MOE_EXPERT_IMPORT_FAILED:
        return None
    try:
        from fp8_cuda import fp8_e4m3_bf16_moe_expert
    except Exception:
        _NATIVE_MOE_EXPERT_IMPORT_FAILED = True
        return None
    _NATIVE_MOE_EXPERT_FN = fp8_e4m3_bf16_moe_expert
    return _NATIVE_MOE_EXPERT_FN


def _tp_moe_expert_group(hidden_chunk: Any, expert: Any, config: RuntimeConfig, weights: ReferenceWeights, runtime: TpRuntime) -> Any:
    stats = getattr(weights, "dispatch_stats", None)
    group_tokens = int(hidden_chunk.shape[0]) if hasattr(hidden_chunk, "shape") and len(hidden_chunk.shape) > 0 else 0
    if stats is not None:
        stats.moe_native_expert_calls += 1
        stats.moe_native_expert_max_group_tokens = max(stats.moe_native_expert_max_group_tokens, group_tokens)
    if not config.moe.native_fused_expert_dispatch:
        _record_native_expert_fallback(stats, "disabled")
        with runtime.profile_scope("moe.fallback_expert"):
            return weights.expert(hidden_chunk, expert)

    eligible, reason, tensors = _tp_moe_native_expert_eligibility(hidden_chunk, expert, weights)
    if not eligible:
        _record_native_expert_fallback(stats, reason)
        with runtime.profile_scope("moe.fallback_expert"):
            return weights.expert(hidden_chunk, expert)
    native_fn = _native_moe_expert_fn()
    if native_fn is None:
        _record_native_expert_fallback(stats, "exception")
        with runtime.profile_scope("moe.fallback_expert"):
            return weights.expert(hidden_chunk, expert)
    if stats is not None:
        stats.moe_native_expert_eligible += 1
    try:
        with runtime.profile_scope("moe.native_expert"):
            output = native_fn(hidden_chunk.float(), *tensors)
    except RuntimeError:
        _record_native_expert_fallback(stats, "exception")
        with runtime.profile_scope("moe.fallback_expert"):
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
    with runtime.profile_scope(_layer_scope(runtime, mapping.index, "total")):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, weights.tensor(mapping.input_layernorm), config.rms_norm_eps)
        if mapping.layer_type == "full_attention":
            with runtime.profile_scope(_layer_scope(runtime, mapping.index, "full_attention.total")):
                hidden_states = residual + _tp_full_attention_batch(
                    hidden_states, mapping.attention, config, weights, runtime,
                    caches=caches, position_offsets=position_offsets,
                )
        elif mapping.layer_type == "linear_attention":
            with runtime.profile_scope(_layer_scope(runtime, mapping.index, "linear_attention.total")):
                hidden_states = residual + _tp_linear_attention_batch(
                    hidden_states, mapping.attention, config, weights, runtime,
                    caches=caches,
                )
        else:
            raise ValueError(f"unsupported TP layer type: {mapping.layer_type}")
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, weights.tensor(mapping.post_attention_layernorm), config.rms_norm_eps)
        with runtime.profile_scope(_layer_scope(runtime, mapping.index, "moe.total")):
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
    with runtime.profile_scope("batch.full_attention.qkv_proj"):
        q_proj = weights.linear(hidden_states, mapping.q_proj).view(batch, seq_len, local_heads, full.head_dim * 2)
        query, gate = q_proj.chunk(2, dim=-1)
        gate = gate.reshape(batch, seq_len, local_heads * full.head_dim)
        key = weights.linear(hidden_states, mapping.k_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
        value = weights.linear(hidden_states, mapping.v_proj).view(batch, seq_len, full.num_key_value_heads, full.head_dim)
    with runtime.profile_scope("batch.full_attention.norm_rope"):
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
    with runtime.profile_scope("batch.full_attention.cache_append"):
        for r in range(batch):
            caches[r].append(key[r: r + 1], value[r: r + 1])

    return _tp_full_attention_batch_paged_or_dense(
        query,
        gate,
        hidden_states,
        mapping,
        config,
        weights,
        runtime,
        caches=caches,
        position_offsets=position_offsets,
    )


def _tp_full_attention_batch_paged_or_dense(
    query: Any,
    gate: Any,
    hidden_states: Any,
    mapping: Any,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    caches: list[FullAttentionCache],
    position_offsets: Any,
) -> Any:
    page_tables = None
    if config.full_attention.paged_kv_metadata:
        with runtime.profile_scope("batch.full_attention.page_metadata"):
            page_tables = batch_page_tables(caches)
    with runtime.profile_scope("batch.full_attention.paged_dispatch"):
        first_cache = next(cache for cache in caches if cache.key_blocks is not None and cache.value_blocks is not None)
        _record_paged_attention_dispatch(
            runtime,
            config,
            query,
            first_cache.key_blocks,
            first_cache.value_blocks,
            seq_len=int(query.shape[2]),
            batched=True,
        )
    with runtime.profile_scope("batch.full_attention.dense_fallback"):
        return _tp_full_attention_batch_dense_from_kv(
            query,
            gate,
            hidden_states,
            mapping,
            config,
            weights,
            runtime,
            caches=caches,
            position_offsets=position_offsets,
            valid_mask=None if page_tables is None else page_tables.valid_mask,
        )


def _tp_full_attention_batch_dense_from_kv(
    query: Any,
    gate: Any,
    hidden_states: Any,
    mapping: Any,
    config: RuntimeConfig,
    weights: ReferenceWeights,
    runtime: TpRuntime,
    *,
    caches: list[FullAttentionCache],
    position_offsets: Any,
    valid_mask: Any | None = None,
) -> Any:
    import torch

    batch, seq_len, _ = hidden_states.shape
    full = config.full_attention
    local_heads = full.num_heads // runtime.config.world_size

    # Gather KV across requests with padding
    with runtime.profile_scope("batch.full_attention.batch_kv_tensors"):
        batched_keys, batched_values, dense_valid_mask = batch_kv_tensors(caches)
    if valid_mask is None:
        valid_mask = dense_valid_mask
    # batched_keys: (B, local_heads, max_valid, head_dim)
    # valid_mask: (B, max_valid)

    # Attention scores
    with runtime.profile_scope("batch.full_attention.scores"):
        scores = torch.matmul(query.float(), batched_keys.float().transpose(2, 3)) * (full.head_dim ** -0.5)
        # scores: (B, local_heads, 1, max_valid)

    # Causal mask: for decode (seq_len=1), query position is position_offsets[r].
    # Key positions are 0..max_valid-1. Mask where key_pos > query_pos OR not valid.
    with runtime.profile_scope("batch.full_attention.mask"):
        max_valid = batched_keys.shape[2]
        key_positions = torch.arange(max_valid, device=hidden_states.device)  # (max_valid,)
        query_pos = position_offsets.unsqueeze(1)  # (B, 1)
        causal_mask = key_positions.unsqueeze(0) > query_pos  # (B, max_valid)
        padding_mask = ~valid_mask  # (B, max_valid)
        combined_mask = (causal_mask | padding_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, max_valid)
        scores = scores.masked_fill(combined_mask, torch.finfo(scores.dtype).min)

    with runtime.profile_scope("batch.full_attention.softmax"):
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    with runtime.profile_scope("batch.full_attention.value"):
        out = torch.matmul(probs, batched_values).transpose(1, 2).reshape(batch, seq_len, local_heads * full.head_dim)
        out = out * torch.sigmoid(gate)
    with runtime.profile_scope("batch.full_attention.output_proj"):
        partial = weights.linear(out, mapping.o_proj).to(hidden_states.dtype)
    with runtime.profile_scope("batch.full_attention.all_reduce"):
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

    with runtime.profile_scope("linear_attention.qkv_conv"):
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

    with runtime.profile_scope("linear_attention.projections"):
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

    with runtime.profile_scope("linear_attention.recurrent_core"):
        core = _recurrent_gated_delta_rule(
            query, key, value, g, beta,
            initial_state=stacked_initial, return_state=True,
        )
    core, new_state = core

    # Scatter state back to per-request caches
    for r in range(batch):
        caches[r].state = new_state[r: r + 1]

    with runtime.profile_scope("linear_attention.output_norm"):
        core = core.reshape(-1, linear_cfg.value_head_dim)
        z = weights.linear(hidden_states, mapping.in_proj_z).reshape(-1, linear_cfg.value_head_dim)
        core = gated_rms_norm(core, weights.tensor(mapping.norm), z, config.rms_norm_eps)
        core = core.reshape(batch, seq_len, local_value_dim)
    with runtime.profile_scope("linear_attention.output_proj"):
        partial = weights.linear(core, mapping.out_proj).to(hidden_states.dtype)
    with runtime.profile_scope("linear_attention.all_reduce"):
        return runtime.all_reduce_sum(partial)


def _layer_scope(runtime: TpRuntime, layer_index: int, name: str) -> str:
    if runtime.profile_config.layer_detail:
        return f"layer.{layer_index}.{name}"
    if name == "total":
        return "layer.total"
    return name


def _tensor_bytes(tensor: Any) -> int:
    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return 0
    return int(numel() * element_size())


def _repeat_kv(hidden_states: Any, repeats: int) -> Any:
    if repeats == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


def _recurrent_gated_delta_rule(query: Any, key: Any, value: Any, g: Any, beta: Any, *, initial_state: Any = None, return_state: bool = False) -> Any:
    if getattr(query, "is_cuda", False):
        try:
            return _recurrent_gated_delta_rule_native(
                query, key, value, g, beta, initial_state=initial_state, return_state=return_state
            )
        except RuntimeError:
            pass
    return _recurrent_gated_delta_rule_torch(query, key, value, g, beta, initial_state=initial_state, return_state=return_state)


def _recurrent_gated_delta_rule_torch(query: Any, key: Any, value: Any, g: Any, beta: Any, *, initial_state: Any = None, return_state: bool = False) -> Any:
    import torch

    initial_dtype = query.dtype
    query, key, value, g, beta, state = _prepare_recurrent_gated_delta_inputs(query, key, value, g, beta, initial_state)
    batch, heads, seq_len, _ = key.shape
    value_dim = value.shape[-1]
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


def _recurrent_gated_delta_rule_native(query: Any, key: Any, value: Any, g: Any, beta: Any, *, initial_state: Any = None, return_state: bool = False) -> Any:
    initial_dtype = query.dtype
    query, key, value, g, beta, state = _prepare_recurrent_gated_delta_inputs(query, key, value, g, beta, initial_state)
    from fp8_cuda import linear_attention_recurrent_core

    output, final_state = linear_attention_recurrent_core(query, key, value, g, beta, state)
    out = output.transpose(1, 2).contiguous().to(initial_dtype)
    if return_state:
        return out, final_state
    return out


def _prepare_recurrent_gated_delta_inputs(query: Any, key: Any, value: Any, g: Any, beta: Any, initial_state: Any = None) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    query = l2_norm(query).transpose(1, 2).float().contiguous()
    key = l2_norm(key).transpose(1, 2).float().contiguous()
    value = value.transpose(1, 2).float().contiguous()
    beta = beta.transpose(1, 2).float().contiguous()
    g = g.transpose(1, 2).float().contiguous()
    batch, heads, _, key_dim = key.shape
    value_dim = value.shape[-1]
    query = (query * (key_dim**-0.5)).contiguous()
    if initial_state is None:
        state = torch.zeros(batch, heads, key_dim, value_dim, device=query.device, dtype=torch.float32)
    else:
        state = initial_state.to(device=query.device, dtype=torch.float32).contiguous()
    return query, key, value, g, beta, state
