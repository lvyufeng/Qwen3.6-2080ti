from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from checkpoint import Manifest, TensorInfo
from runtime_config import ConfigError, LayerType, RuntimeConfig, parse_runtime_config


class MappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearTensor:
    weight: TensorInfo
    scale: TensorInfo | None


@dataclass(frozen=True)
class ExpertMapping:
    gate_proj: LinearTensor
    up_proj: LinearTensor
    down_proj: LinearTensor


@dataclass(frozen=True)
class MoEMapping:
    gate: TensorInfo
    experts: tuple[ExpertMapping, ...]
    shared_expert: ExpertMapping
    shared_expert_gate: TensorInfo


@dataclass(frozen=True)
class LinearAttentionMapping:
    in_proj_qkv: LinearTensor
    in_proj_z: LinearTensor
    out_proj: LinearTensor
    in_proj_a: LinearTensor
    in_proj_b: LinearTensor
    conv1d_weight: TensorInfo
    a_log: TensorInfo
    dt_bias: TensorInfo
    norm: TensorInfo


@dataclass(frozen=True)
class FullAttentionMapping:
    q_proj: LinearTensor
    k_proj: LinearTensor
    v_proj: LinearTensor
    o_proj: LinearTensor
    q_norm: TensorInfo
    k_norm: TensorInfo


@dataclass(frozen=True)
class LayerMapping:
    index: int
    layer_type: LayerType
    input_layernorm: TensorInfo
    attention: LinearAttentionMapping | FullAttentionMapping
    post_attention_layernorm: TensorInfo
    mlp: MoEMapping


@dataclass(frozen=True)
class LanguageModelMapping:
    model_dir: Path
    embed_tokens: TensorInfo
    final_norm: TensorInfo
    lm_head: TensorInfo
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
    def __init__(self, manifest: Manifest, config: RuntimeConfig) -> None:
        self.manifest = manifest
        self.config = config
        self.mapped: set[str] = set()

    def tensor(
        self,
        name: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> TensorInfo:
        try:
            tensor = self.manifest.tensors[name]
        except KeyError as exc:
            raise MappingError(f"missing tensor: {name}") from exc
        self.mapped.add(name)
        if shape is not None and tensor.shape != shape:
            raise MappingError(f"{name}: expected shape {shape}, got {tensor.shape}")
        if dtype is not None and tensor.dtype != dtype:
            raise MappingError(f"{name}: expected dtype {dtype}, got {tensor.dtype}")
        return tensor

    def linear(
        self,
        name: str,
        *,
        shape: tuple[int, int] | None = None,
        fp8: bool | None = None,
    ) -> LinearTensor:
        weight = self.tensor(name, shape=shape)
        if len(weight.shape) != 2:
            raise MappingError(f"{name}: expected rank-2 linear weight, got {weight.shape}")
        if fp8 is True and not weight.is_fp8:
            raise MappingError(f"{name}: expected FP8 weight, got {weight.dtype}")
        if fp8 is False and weight.is_fp8:
            raise MappingError(f"{name}: expected non-FP8 weight, got {weight.dtype}")
        if not weight.is_fp8:
            return LinearTensor(weight=weight, scale=None)
        try:
            scale_name = self.manifest.scale_of[name]
        except KeyError as exc:
            raise MappingError(f"{name}: missing FP8 scale tensor") from exc
        scale = self.tensor(scale_name)
        expected_scale_shape = self.config.fp8_scale_shape(weight.shape)
        if scale.shape != expected_scale_shape:
            raise MappingError(f"{scale.name}: expected shape {expected_scale_shape}, got {scale.shape}")
        return LinearTensor(weight=weight, scale=scale)


def build_language_model_mapping(manifest: Manifest, *, strict: bool = True) -> LanguageModelMapping:
    try:
        config = parse_runtime_config(manifest.config)
    except ConfigError as exc:
        raise MappingError(str(exc)) from exc
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    builder = _Builder(manifest, config)
    embed_tokens = builder.tensor(
        "model.language_model.embed_tokens.weight",
        shape=(vocab_size, hidden_size),
        dtype="BF16",
    )
    final_norm = builder.tensor("model.language_model.norm.weight", shape=(hidden_size,), dtype="BF16")
    lm_head = builder.tensor("lm_head.weight", shape=(vocab_size, hidden_size), dtype="BF16")
    layers = tuple(_map_layer(builder, config, index, layer_type) for index, layer_type in enumerate(config.layer_types))
    mapped = frozenset(builder.mapped)
    unmapped_language = tuple(sorted(name for name in manifest.tensors if _is_language_tensor(name) and name not in mapped))
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
    return LinearAttentionMapping(
        in_proj_qkv=builder.linear(p + "in_proj_qkv.weight", shape=(linear.qkv_dim, hidden_size), fp8=True),
        in_proj_z=builder.linear(p + "in_proj_z.weight", shape=(linear.value_state_dim, hidden_size), fp8=True),
        out_proj=builder.linear(p + "out_proj.weight", shape=(hidden_size, linear.value_state_dim), fp8=True),
        in_proj_a=builder.linear(p + "in_proj_a.weight", shape=(linear.value_heads, hidden_size), fp8=False),
        in_proj_b=builder.linear(p + "in_proj_b.weight", shape=(linear.value_heads, hidden_size), fp8=False),
        conv1d_weight=builder.tensor(p + "conv1d.weight", shape=(linear.qkv_dim, 1, linear.conv_kernel_dim), dtype="BF16"),
        a_log=builder.tensor(p + "A_log", shape=(linear.value_heads,), dtype="BF16"),
        dt_bias=builder.tensor(p + "dt_bias", shape=(linear.value_heads,), dtype="BF16"),
        norm=builder.tensor(p + "norm.weight", shape=(linear.value_head_dim,), dtype="BF16"),
    )


def _map_full_attention(builder: _Builder, config: RuntimeConfig, prefix: str) -> FullAttentionMapping:
    hidden_size = config.hidden_size
    full = config.full_attention
    p = prefix + "self_attn."
    return FullAttentionMapping(
        q_proj=builder.linear(p + "q_proj.weight", shape=(full.q_dim, hidden_size), fp8=True),
        k_proj=builder.linear(p + "k_proj.weight", shape=(full.kv_dim, hidden_size), fp8=True),
        v_proj=builder.linear(p + "v_proj.weight", shape=(full.kv_dim, hidden_size), fp8=True),
        o_proj=builder.linear(p + "o_proj.weight", shape=(hidden_size, full.attn_dim), fp8=True),
        q_norm=builder.tensor(p + "q_norm.weight", shape=(full.head_dim,), dtype="BF16"),
        k_norm=builder.tensor(p + "k_norm.weight", shape=(full.head_dim,), dtype="BF16"),
    )


def _map_moe(builder: _Builder, config: RuntimeConfig, prefix: str) -> MoEMapping:
    hidden_size = config.hidden_size
    moe = config.moe
    return MoEMapping(
        gate=builder.tensor(prefix + "gate.weight", shape=(moe.num_experts, hidden_size), dtype="BF16"),
        experts=tuple(
            _map_expert(builder, f"{prefix}experts.{i}.", hidden_size, moe.intermediate_size)
            for i in range(moe.num_experts)
        ),
        shared_expert=_map_expert(builder, prefix + "shared_expert.", hidden_size, moe.shared_intermediate_size),
        shared_expert_gate=builder.tensor(prefix + "shared_expert_gate.weight", shape=(1, hidden_size), dtype="BF16"),
    )


def _map_expert(builder: _Builder, prefix: str, hidden_size: int, intermediate_size: int) -> ExpertMapping:
    return ExpertMapping(
        gate_proj=builder.linear(prefix + "gate_proj.weight", shape=(intermediate_size, hidden_size), fp8=True),
        up_proj=builder.linear(prefix + "up_proj.weight", shape=(intermediate_size, hidden_size), fp8=True),
        down_proj=builder.linear(prefix + "down_proj.weight", shape=(hidden_size, intermediate_size), fp8=True),
    )


def _is_language_tensor(name: str) -> bool:
    return name == "lm_head.weight" or name.startswith("model.language_model.")
