from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping


@dataclass
class FullAttentionCache:
    """Per-layer KV cache for full attention.

    Stores post-rotary key/value in head-major layout ``(batch, heads, seq, head_dim)`` —
    the exact layout used right before the attention score matmul in ``full_attention``.
    """

    key: Any | None = None
    value: Any | None = None

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[2]

    def append(self, key: Any, value: Any) -> tuple[Any, Any]:
        import torch

        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
        return self.key, self.value


@dataclass
class LinearAttentionCache:
    """Per-layer recurrent + conv state for gated-delta linear attention.

    ``state`` is the recurrent memory ``(batch, heads, key_dim, value_dim)``.
    ``conv_tail`` holds the last ``conv_kernel_dim - 1`` conv inputs in channel-major
    layout ``(batch, channels, conv_kernel_dim - 1)`` to match the ``F.conv1d`` input.
    Both are created lazily on the first step.
    """

    state: Any | None = None
    conv_tail: Any | None = None


LayerCache = FullAttentionCache | LinearAttentionCache


@dataclass
class DecodeState:
    """Per-layer decode caches plus the number of tokens already consumed."""

    layers: list[LayerCache]
    position_offset: int = 0

    @classmethod
    def empty(cls, mapping: LanguageModelMapping, config: RuntimeConfig | None = None) -> DecodeState:
        del config  # caches allocate lazily; only the per-layer type matters
        layers: list[LayerCache] = []
        for layer in mapping.layers:
            if layer.layer_type == "full_attention":
                layers.append(FullAttentionCache())
            else:
                layers.append(LinearAttentionCache())
        return cls(layers=layers, position_offset=0)

    def advance(self, tokens: int) -> None:
        self.position_offset += tokens
