from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

LayerType = Literal["linear_attention", "full_attention"]

FP8_BLOCK_SIZE = 128


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearAttentionConfig:
    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int

    @property
    def qkv_dim(self) -> int:
        return self.key_heads * self.key_head_dim * 2 + self.value_heads * self.value_head_dim

    @property
    def value_state_dim(self) -> int:
        return self.value_heads * self.value_head_dim


@dataclass(frozen=True)
class FullAttentionConfig:
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    output_gate: bool

    @property
    def attn_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.attn_dim * (2 if self.output_gate else 1)


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int
    experts_per_token: int
    intermediate_size: int
    shared_intermediate_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    model_type: str
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    layer_types: tuple[LayerType, ...]
    linear_attention: LinearAttentionConfig
    full_attention: FullAttentionConfig
    moe: MoEConfig
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    fp8_block_size: int = FP8_BLOCK_SIZE

    @property
    def linear_attention_layers(self) -> int:
        return sum(1 for layer_type in self.layer_types if layer_type == "linear_attention")

    @property
    def full_attention_layers(self) -> int:
        return sum(1 for layer_type in self.layer_types if layer_type == "full_attention")

    def fp8_scale_shape(self, weight_shape: tuple[int, int]) -> tuple[int, int]:
        return (_ceil_div(weight_shape[0], self.fp8_block_size), _ceil_div(weight_shape[1], self.fp8_block_size))


def parse_runtime_config(config: dict[str, Any]) -> RuntimeConfig:
    text_config = _text_config(config)
    num_layers = _required_int(text_config, "num_hidden_layers")
    rope_parameters = text_config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise ConfigError("missing object text_config.rope_parameters")
    return RuntimeConfig(
        model_type=_required_str(text_config, "model_type"),
        hidden_size=_required_int(text_config, "hidden_size"),
        vocab_size=_required_int(text_config, "vocab_size"),
        num_hidden_layers=num_layers,
        layer_types=_layer_types(text_config, num_layers),
        linear_attention=LinearAttentionConfig(
            key_heads=_required_int(text_config, "linear_num_key_heads"),
            value_heads=_required_int(text_config, "linear_num_value_heads"),
            key_head_dim=_required_int(text_config, "linear_key_head_dim"),
            value_head_dim=_required_int(text_config, "linear_value_head_dim"),
            conv_kernel_dim=_required_int(text_config, "linear_conv_kernel_dim"),
        ),
        full_attention=FullAttentionConfig(
            num_heads=_required_int(text_config, "num_attention_heads"),
            num_key_value_heads=_required_int(text_config, "num_key_value_heads"),
            head_dim=_required_int(text_config, "head_dim"),
            output_gate=_required_bool(text_config, "attn_output_gate"),
        ),
        moe=MoEConfig(
            num_experts=_required_int(text_config, "num_experts"),
            experts_per_token=_required_int(text_config, "num_experts_per_tok"),
            intermediate_size=_required_int(text_config, "moe_intermediate_size"),
            shared_intermediate_size=_required_int(text_config, "shared_expert_intermediate_size"),
        ),
        max_position_embeddings=_required_int(text_config, "max_position_embeddings"),
        rms_norm_eps=_required_float(text_config, "rms_norm_eps"),
        rope_theta=_required_float(rope_parameters, "rope_theta", prefix="text_config.rope_parameters"),
        partial_rotary_factor=_required_float(text_config, "partial_rotary_factor"),
    )


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if text_config is None:
        return config
    if not isinstance(text_config, dict):
        raise ConfigError("expected object text_config")
    return text_config


def _layer_types(config: dict[str, Any], num_layers: int) -> tuple[LayerType, ...]:
    raw = config.get("layer_types")
    if not isinstance(raw, list) or len(raw) != num_layers:
        raise ConfigError(f"expected text_config.layer_types list with {num_layers} entries")
    layer_types: list[LayerType] = []
    for index, value in enumerate(raw):
        if value not in ("linear_attention", "full_attention"):
            raise ConfigError(f"unsupported layer type at {index}: {value!r}")
        layer_types.append(value)
    return tuple(layer_types)


def _required_bool(config: dict[str, Any], key: str, *, prefix: str = "text_config") -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"missing boolean {prefix}.{key}")
    return value


def _required_float(config: dict[str, Any], key: str, *, prefix: str = "text_config") -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"missing number {prefix}.{key}")
    return float(value)


def _required_int(config: dict[str, Any], key: str, *, prefix: str = "text_config") -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"missing integer {prefix}.{key}")
    return value


def _required_str(config: dict[str, Any], key: str, *, prefix: str = "text_config") -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"missing string {prefix}.{key}")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
