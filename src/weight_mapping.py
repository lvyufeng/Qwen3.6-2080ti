from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from checkpoint import Manifest, TensorInfo
from runtime_config import ConfigError, LayerType, RuntimeConfig, parse_runtime_config
from tensor_parallel import TensorParallel


class MappingError(RuntimeError):
    pass


ShardRule = Literal[
    "replicated",
    "column_parallel",
    "row_parallel",
    "parallel_embedding",
    "parallel_head",
    "packed_qkv_column_parallel",
    "packed_conv1d_channel_parallel",
    "vector_column_parallel",
    "expert_owned",
]


@dataclass(frozen=True)
class PackedSegment:
    start: int
    size: int


@dataclass(frozen=True)
class TensorShard:
    rule: ShardRule
    dim: int | None = None
    start: int | None = None
    size: int | None = None
    local_shape: tuple[int, ...] | None = None
    segments: tuple[PackedSegment, ...] = ()

    @classmethod
    def replicated(cls, shape: tuple[int, ...]) -> TensorShard:
        return cls(rule="replicated", local_shape=shape)

    @classmethod
    def dim_shard(cls, rule: ShardRule, shape: tuple[int, ...], dim: int, tp: TensorParallel) -> TensorShard:
        start, size = tp.shard_range(shape[dim])
        local = list(shape)
        local[dim] = size
        return cls(rule=rule, dim=dim, start=start, size=size, local_shape=tuple(local))

    @classmethod
    def packed_dim0(cls, rule: ShardRule, shape: tuple[int, ...], segments: tuple[int, ...], tp: TensorParallel) -> TensorShard:
        local_segments: list[PackedSegment] = []
        offset = 0
        local_total = 0
        for segment_size in segments:
            start, size = tp.shard_range(segment_size)
            local_segments.append(PackedSegment(offset + start, size))
            offset += segment_size
            local_total += size
        local = (local_total, *shape[1:])
        return cls(rule=rule, dim=0, local_shape=local, segments=tuple(local_segments))


@dataclass(frozen=True)
class ShardedTensor:
    info: TensorInfo
    shard: TensorShard

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.shard.local_shape or self.info.shape

    @property
    def dtype(self) -> str:
        return self.info.dtype

    @property
    def nbytes(self) -> int:
        if self.shard.local_shape is None:
            return self.info.nbytes
        item_size = self.info.nbytes // self.info.numel
        numel = 1
        for dim in self.shard.local_shape:
            numel *= dim
        return numel * item_size


@dataclass(frozen=True)
class LinearTensor:
    weight: ShardedTensor
    scale: ShardedTensor | None


@dataclass(frozen=True)
class ExpertMapping:
    index: int
    gate_proj: LinearTensor
    up_proj: LinearTensor
    down_proj: LinearTensor


@dataclass(frozen=True)
class MoEMapping:
    gate: ShardedTensor
    experts: tuple[ExpertMapping, ...]
    shared_expert: ExpertMapping
    shared_expert_gate: ShardedTensor
    expert_start: int
    expert_end: int
    num_experts: int
    tp: TensorParallel


@dataclass(frozen=True)
class LinearAttentionMapping:
    in_proj_qkv: LinearTensor
    in_proj_z: LinearTensor
    out_proj: LinearTensor
    in_proj_a: LinearTensor
    in_proj_b: LinearTensor
    conv1d_weight: ShardedTensor
    a_log: ShardedTensor
    dt_bias: ShardedTensor
    norm: ShardedTensor


@dataclass(frozen=True)
class FullAttentionMapping:
    q_proj: LinearTensor
    k_proj: LinearTensor
    v_proj: LinearTensor
    o_proj: LinearTensor
    q_norm: ShardedTensor
    k_norm: ShardedTensor


@dataclass(frozen=True)
class LayerMapping:
    index: int
    layer_type: LayerType
    input_layernorm: ShardedTensor
    attention: LinearAttentionMapping | FullAttentionMapping
    post_attention_layernorm: ShardedTensor
    mlp: MoEMapping


@dataclass(frozen=True)
class LanguageModelMapping:
    model_dir: Path
    embed_tokens: ShardedTensor
    final_norm: ShardedTensor
    lm_head: ShardedTensor
    layers: tuple[LayerMapping, ...]
    mapped_tensor_names: frozenset[str]
    ignored_tensor_names: frozenset[str]
    unmapped_language_tensor_names: tuple[str, ...]

    @property
    def linear_attention_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.layer_type == "linear_attention")

    @property
    def full_attention_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.layer_type == "full_attention")

    @property
    def experts_per_layer(self) -> int:
        return len(self.layers[0].mlp.experts) if self.layers else 0

    @property
    def routed_experts(self) -> int:
        return sum(len(layer.mlp.experts) for layer in self.layers)


class _Builder:
    def __init__(self, manifest: Manifest, config: RuntimeConfig, tp: TensorParallel) -> None:
        self.manifest = manifest
        self.config = config
        self.tp = tp
        self.mapped: set[str] = set()
        self.remote: set[str] = set()

    def claim_remote(self, name: str) -> None:
        if name not in self.manifest.tensors:
            raise MappingError(f"missing tensor: {name}")
        self.remote.add(name)
        scale = self.manifest.scale_of.get(name)
        if scale is not None:
            self.remote.add(scale)

    def tensor(
        self,
        name: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
        shard: TensorShard | None = None,
    ) -> ShardedTensor:
        try:
            tensor = self.manifest.tensors[name]
        except KeyError as exc:
            raise MappingError(f"missing tensor: {name}") from exc
        self.mapped.add(name)
        if shape is not None and tensor.shape != shape:
            raise MappingError(f"{name}: expected shape {shape}, got {tensor.shape}")
        if dtype is not None and tensor.dtype != dtype:
            raise MappingError(f"{name}: expected dtype {dtype}, got {tensor.dtype}")
        return ShardedTensor(info=tensor, shard=shard or TensorShard.replicated(tensor.shape))

    def linear(
        self,
        name: str,
        *,
        shape: tuple[int, int] | None = None,
        fp8: bool | None = None,
        shard: TensorShard | None = None,
    ) -> LinearTensor:
        weight = self.tensor(name, shape=shape, shard=shard)
        if len(weight.info.shape) != 2:
            raise MappingError(f"{name}: expected rank-2 linear weight, got {weight.info.shape}")
        if fp8 is True and not weight.info.is_fp8:
            raise MappingError(f"{name}: expected FP8 weight, got {weight.info.dtype}")
        if fp8 is False and weight.info.is_fp8:
            raise MappingError(f"{name}: expected non-FP8 weight, got {weight.info.dtype}")
        if not weight.info.is_fp8:
            return LinearTensor(weight=weight, scale=None)
        try:
            scale_name = self.manifest.scale_of[name]
        except KeyError as exc:
            raise MappingError(f"{name}: missing FP8 scale tensor") from exc
        scale_info = self.manifest.tensors[scale_name]
        expected_scale_shape = self.config.fp8_scale_shape(weight.info.shape)
        if scale_info.shape != expected_scale_shape:
            raise MappingError(f"{scale_name}: expected shape {expected_scale_shape}, got {scale_info.shape}")
        self.mapped.add(scale_name)
        scale = ShardedTensor(info=scale_info, shard=_scale_shard_for(weight.shard, self.config.fp8_block_size, scale_info.shape))
        return LinearTensor(weight=weight, scale=scale)



def build_language_model_mapping(
    manifest: Manifest,
    *,
    strict: bool = True,
    tensor_parallel: TensorParallel | None = None,
) -> LanguageModelMapping:
    try:
        config = parse_runtime_config(manifest.config)
    except ConfigError as exc:
        raise MappingError(str(exc)) from exc
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    tp = tensor_parallel or TensorParallel(world_size=1, rank=0)
    tp.local_expert_count(config.moe.num_experts)

    builder = _Builder(manifest, config, tp)
    embed_tokens = builder.tensor(
        "model.language_model.embed_tokens.weight",
        shape=(vocab_size, hidden_size),
        dtype="BF16",
        shard=TensorShard.dim_shard("parallel_embedding", (vocab_size, hidden_size), 0, tp),
    )
    final_norm = builder.tensor(
        "model.language_model.norm.weight",
        shape=(hidden_size,),
        dtype="BF16",
    )
    lm_head = builder.tensor(
        "lm_head.weight",
        shape=(vocab_size, hidden_size),
        dtype="BF16",
        shard=TensorShard.dim_shard("parallel_head", (vocab_size, hidden_size), 0, tp),
    )
    layers = tuple(_map_layer(builder, config, index, layer_type) for index, layer_type in enumerate(config.layer_types))
    mapped = frozenset(builder.mapped)
    claimed = mapped | frozenset(builder.remote)
    unmapped_language = tuple(sorted(name for name in manifest.tensors if _is_language_tensor(name) and name not in claimed))
    if strict and unmapped_language:
        examples = ", ".join(unmapped_language[:8])
        raise MappingError(f"unmapped language tensors: {examples}")
    return LanguageModelMapping(
        model_dir=manifest.model_dir,
        embed_tokens=embed_tokens,
        final_norm=final_norm,
        lm_head=lm_head,
        layers=layers,
        mapped_tensor_names=mapped,
        ignored_tensor_names=frozenset(name for name in manifest.tensors if name not in mapped),
        unmapped_language_tensor_names=unmapped_language,
    )


def _map_layer(builder: _Builder, config: RuntimeConfig, index: int, layer_type: LayerType) -> LayerMapping:
    hidden_size = config.hidden_size
    prefix = f"model.language_model.layers.{index}."
    input_layernorm = builder.tensor(prefix + "input_layernorm.weight", shape=(hidden_size,), dtype="BF16")
    if layer_type == "linear_attention":
        attention: LinearAttentionMapping | FullAttentionMapping = _map_linear_attention(builder, config, prefix)
    elif layer_type == "full_attention":
        attention = _map_full_attention(builder, config, prefix)
    else:
        raise MappingError(f"unsupported layer type at {index}: {layer_type}")
    return LayerMapping(
        index=index,
        layer_type=layer_type,
        input_layernorm=input_layernorm,
        attention=attention,
        post_attention_layernorm=builder.tensor(
            prefix + "post_attention_layernorm.weight",
            shape=(hidden_size,),
            dtype="BF16",
        ),
        mlp=_map_moe(builder, config, prefix + "mlp."),
    )


def _map_linear_attention(builder: _Builder, config: RuntimeConfig, prefix: str) -> LinearAttentionMapping:
    hidden_size = config.hidden_size
    linear = config.linear_attention
    p = prefix + "linear_attn."
    qkv_segments = (
        linear.key_heads * linear.key_head_dim,
        linear.key_heads * linear.key_head_dim,
        linear.value_state_dim,
    )
    return LinearAttentionMapping(
        in_proj_qkv=builder.linear(
            p + "in_proj_qkv.weight",
            shape=(linear.qkv_dim, hidden_size),
            fp8=True,
            shard=TensorShard.packed_dim0("packed_qkv_column_parallel", (linear.qkv_dim, hidden_size), qkv_segments, builder.tp),
        ),
        in_proj_z=builder.linear(
            p + "in_proj_z.weight",
            shape=(linear.value_state_dim, hidden_size),
            fp8=True,
            shard=TensorShard.dim_shard("column_parallel", (linear.value_state_dim, hidden_size), 0, builder.tp),
        ),
        out_proj=builder.linear(
            p + "out_proj.weight",
            shape=(hidden_size, linear.value_state_dim),
            fp8=True,
            shard=TensorShard.dim_shard("row_parallel", (hidden_size, linear.value_state_dim), 1, builder.tp),
        ),
        in_proj_a=builder.linear(
            p + "in_proj_a.weight",
            shape=(linear.value_heads, hidden_size),
            fp8=False,
            shard=TensorShard.dim_shard("column_parallel", (linear.value_heads, hidden_size), 0, builder.tp),
        ),
        in_proj_b=builder.linear(
            p + "in_proj_b.weight",
            shape=(linear.value_heads, hidden_size),
            fp8=False,
            shard=TensorShard.dim_shard("column_parallel", (linear.value_heads, hidden_size), 0, builder.tp),
        ),
        conv1d_weight=builder.tensor(
            p + "conv1d.weight",
            shape=(linear.qkv_dim, 1, linear.conv_kernel_dim),
            dtype="BF16",
            shard=TensorShard.packed_dim0("packed_conv1d_channel_parallel", (linear.qkv_dim, 1, linear.conv_kernel_dim), qkv_segments, builder.tp),
        ),
        a_log=builder.tensor(
            p + "A_log",
            shape=(linear.value_heads,),
            dtype="BF16",
            shard=TensorShard.dim_shard("vector_column_parallel", (linear.value_heads,), 0, builder.tp),
        ),
        dt_bias=builder.tensor(
            p + "dt_bias",
            shape=(linear.value_heads,),
            dtype="BF16",
            shard=TensorShard.dim_shard("vector_column_parallel", (linear.value_heads,), 0, builder.tp),
        ),
        norm=builder.tensor(p + "norm.weight", shape=(linear.value_head_dim,), dtype="BF16"),
    )


def _map_full_attention(builder: _Builder, config: RuntimeConfig, prefix: str) -> FullAttentionMapping:
    hidden_size = config.hidden_size
    full = config.full_attention
    p = prefix + "self_attn."
    return FullAttentionMapping(
        q_proj=builder.linear(
            p + "q_proj.weight",
            shape=(full.q_dim, hidden_size),
            fp8=True,
            shard=TensorShard.dim_shard("column_parallel", (full.q_dim, hidden_size), 0, builder.tp),
        ),
        k_proj=builder.linear(p + "k_proj.weight", shape=(full.kv_dim, hidden_size), fp8=True),
        v_proj=builder.linear(p + "v_proj.weight", shape=(full.kv_dim, hidden_size), fp8=True),
        o_proj=builder.linear(
            p + "o_proj.weight",
            shape=(hidden_size, full.attn_dim),
            fp8=True,
            shard=TensorShard.dim_shard("row_parallel", (hidden_size, full.attn_dim), 1, builder.tp),
        ),
        q_norm=builder.tensor(p + "q_norm.weight", shape=(full.head_dim,), dtype="BF16"),
        k_norm=builder.tensor(p + "k_norm.weight", shape=(full.head_dim,), dtype="BF16"),
    )


def _map_moe(builder: _Builder, config: RuntimeConfig, prefix: str) -> MoEMapping:
    hidden_size = config.hidden_size
    moe = config.moe
    expert_start, expert_end = builder.tp.expert_range(moe.num_experts)
    experts: list[ExpertMapping] = []
    for i in range(moe.num_experts):
        expert_prefix = f"{prefix}experts.{i}."
        if expert_start <= i < expert_end:
            experts.append(_map_expert(builder, expert_prefix, i, hidden_size, moe.intermediate_size, expert_owned=True))
        else:
            _claim_remote_expert(builder, expert_prefix)
    return MoEMapping(
        gate=builder.tensor(prefix + "gate.weight", shape=(moe.num_experts, hidden_size), dtype="BF16"),
        experts=tuple(experts),
        shared_expert=_map_expert(builder, prefix + "shared_expert.", -1, hidden_size, moe.shared_intermediate_size, expert_owned=False),
        shared_expert_gate=builder.tensor(prefix + "shared_expert_gate.weight", shape=(1, hidden_size), dtype="BF16"),
        expert_start=expert_start,
        expert_end=expert_end,
        num_experts=moe.num_experts,
        tp=builder.tp,
    )


def _map_expert(
    builder: _Builder,
    prefix: str,
    index: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    expert_owned: bool,
) -> ExpertMapping:
    rule: ShardRule = "expert_owned" if expert_owned else "replicated"
    shard = TensorShard.replicated((intermediate_size, hidden_size)) if not expert_owned else TensorShard(rule=rule, local_shape=(intermediate_size, hidden_size))
    down_shard = TensorShard.replicated((hidden_size, intermediate_size)) if not expert_owned else TensorShard(rule=rule, local_shape=(hidden_size, intermediate_size))
    return ExpertMapping(
        index=index,
        gate_proj=builder.linear(prefix + "gate_proj.weight", shape=(intermediate_size, hidden_size), fp8=True, shard=shard),
        up_proj=builder.linear(prefix + "up_proj.weight", shape=(intermediate_size, hidden_size), fp8=True, shard=shard),
        down_proj=builder.linear(prefix + "down_proj.weight", shape=(hidden_size, intermediate_size), fp8=True, shard=down_shard),
    )


def _claim_remote_expert(builder: _Builder, prefix: str) -> None:
    builder.claim_remote(prefix + "gate_proj.weight")
    builder.claim_remote(prefix + "up_proj.weight")
    builder.claim_remote(prefix + "down_proj.weight")


def _scale_shard_for(weight_shard: TensorShard, block_size: int, scale_shape: tuple[int, ...]) -> TensorShard:
    if weight_shard.rule in {"replicated", "expert_owned"}:
        return TensorShard(rule=weight_shard.rule, local_shape=scale_shape)
    if weight_shard.rule in {"column_parallel", "parallel_embedding", "parallel_head"}:
        assert weight_shard.dim == 0 and weight_shard.start is not None and weight_shard.size is not None
        start, size = _block_range(weight_shard.start, weight_shard.size, block_size)
        return TensorShard(
            rule=weight_shard.rule,
            dim=0,
            start=start,
            size=size,
            local_shape=(size, scale_shape[1]),
        )
    if weight_shard.rule == "row_parallel":
        assert weight_shard.dim == 1 and weight_shard.start is not None and weight_shard.size is not None
        start, size = _block_range(weight_shard.start, weight_shard.size, block_size)
        return TensorShard(
            rule=weight_shard.rule,
            dim=1,
            start=start,
            size=size,
            local_shape=(scale_shape[0], size),
        )
    if weight_shard.rule == "packed_qkv_column_parallel":
        scale_segments = tuple(PackedSegment(*_block_range(segment.start, segment.size, block_size)) for segment in weight_shard.segments)
        local_rows = sum(segment.size for segment in scale_segments)
        return TensorShard(rule=weight_shard.rule, dim=0, local_shape=(local_rows, scale_shape[1]), segments=scale_segments)
    raise MappingError(f"unsupported FP8 scale shard rule: {weight_shard.rule}")


def _block_range(start: int, size: int, block_size: int) -> tuple[int, int]:
    block_start = start // block_size
    block_end = (start + size + block_size - 1) // block_size
    return block_start, block_end - block_start


def _is_language_tensor(name: str) -> bool:
    return name == "lm_head.weight" or name.startswith("model.language_model.")
