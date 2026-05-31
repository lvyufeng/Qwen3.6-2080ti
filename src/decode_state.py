from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping


@dataclass
class FullAttentionCache:
    """Per-layer KV cache for full attention.

    ``key``/``value`` are preallocated buffers in head-major layout
    ``(batch, heads, capacity, head_dim)`` — the exact layout used right before the
    attention score matmul in ``full_attention`` — and ``valid`` is the write pointer
    counting how many of those ``capacity`` slots hold real tokens. ``append`` writes new
    tokens in place and returns a view of the valid region, so the per-token ``torch.cat``
    copy is gone: a decode of ``N`` tokens costs amortized ``O(N)`` instead of ``O(N²)``.

    ``capacity_hint`` lets the first allocation cover the whole generation up front (see
    ``DecodeState.empty``); if it is too small or absent the buffer grows geometrically.
    """

    key: Any | None = None
    value: Any | None = None
    valid: int = 0
    capacity_hint: int | None = None

    @property
    def length(self) -> int:
        return self.valid

    def append(self, key: Any, value: Any) -> tuple[Any, Any]:
        import torch

        batch, heads, seq, head_dim = key.shape
        needed = self.valid + seq
        if self.key is None:
            capacity = max(needed, self.capacity_hint or 0)
            self.key = torch.empty((batch, heads, capacity, head_dim), dtype=key.dtype, device=key.device)
            self.value = torch.empty((batch, heads, capacity, head_dim), dtype=value.dtype, device=value.device)
        elif needed > self.key.shape[2]:
            capacity = max(needed, self.key.shape[2] * 2)
            self.key = _grow(self.key, capacity, self.valid)
            self.value = _grow(self.value, capacity, self.valid)
        self.key[:, :, self.valid : needed] = key
        self.value[:, :, self.valid : needed] = value
        self.valid = needed
        return self.key[:, :, :needed], self.value[:, :, :needed]


def _grow(buffer: Any, capacity: int, valid: int) -> Any:
    import torch

    batch, heads, _old_capacity, head_dim = buffer.shape
    grown = torch.empty((batch, heads, capacity, head_dim), dtype=buffer.dtype, device=buffer.device)
    grown[:, :, :valid] = buffer[:, :, :valid]
    return grown


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
    def empty(
        cls,
        mapping: LanguageModelMapping,
        config: RuntimeConfig | None = None,
        max_seq_len: int | None = None,
    ) -> DecodeState:
        del config  # caches allocate lazily; only the per-layer type matters
        layers: list[LayerCache] = []
        for layer in mapping.layers:
            if layer.layer_type == "full_attention":
                layers.append(FullAttentionCache(capacity_hint=max_seq_len))
            else:
                layers.append(LinearAttentionCache())
        return cls(layers=layers, position_offset=0)

    def advance(self, tokens: int) -> None:
        self.position_offset += tokens
