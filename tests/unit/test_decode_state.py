from __future__ import annotations

from types import SimpleNamespace

import torch

from decode_state import DecodeState, FullAttentionCache, LinearAttentionCache


def _kv(step: int, *, batch: int = 1, heads: int = 2, seq: int = 1, head_dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct, reproducible key/value tensors so equivalence checks are meaningful."""
    base = float(step * 1000)
    count = batch * heads * seq * head_dim
    key = (base + torch.arange(count, dtype=torch.float32)).reshape(batch, heads, seq, head_dim)
    value = (base + 0.5 + torch.arange(count, dtype=torch.float32)).reshape(batch, heads, seq, head_dim)
    return key, value


def _mapping(layer_types: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        layers=tuple(SimpleNamespace(index=index, layer_type=layer_type) for index, layer_type in enumerate(layer_types))
    )


def test_append_returns_views_equal_to_manual_concat() -> None:
    cache = FullAttentionCache()
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for step in range(5):
        key, value = _kv(step)
        keys.append(key)
        values.append(value)
        out_key, out_value = cache.append(key, value)
        torch.testing.assert_close(out_key, torch.cat(keys, dim=2))
        torch.testing.assert_close(out_value, torch.cat(values, dim=2))


def test_append_handles_prefill_chunk_then_single_steps() -> None:
    cache = FullAttentionCache()
    prefill_key, prefill_value = _kv(0, seq=3)
    keys = [prefill_key]
    values = [prefill_value]
    cache.append(prefill_key, prefill_value)

    for step in range(1, 4):
        key, value = _kv(step, seq=1)
        keys.append(key)
        values.append(value)
        out_key, out_value = cache.append(key, value)

    torch.testing.assert_close(out_key, torch.cat(keys, dim=2))
    torch.testing.assert_close(out_value, torch.cat(values, dim=2))


def test_length_tracks_total_appended_tokens() -> None:
    cache = FullAttentionCache()
    assert cache.length == 0

    cache.append(*_kv(0, seq=3))
    assert cache.length == 3

    cache.append(*_kv(1, seq=1))
    cache.append(*_kv(2, seq=1))
    assert cache.length == 5


def test_capacity_hint_allocates_once_without_reallocation() -> None:
    cache = FullAttentionCache(capacity_hint=8)

    cache.append(*_kv(0, seq=3))
    buffer_ptr = cache.key.data_ptr()
    value_ptr = cache.value.data_ptr()
    assert cache.key.shape[2] == 8

    for step in range(1, 5):
        cache.append(*_kv(step, seq=1))

    # The buffer object is preallocated once and written in place — never replaced.
    assert cache.key.data_ptr() == buffer_ptr
    assert cache.value.data_ptr() == value_ptr
    assert cache.key.shape[2] == 8
    assert cache.length == 7


def test_growth_beyond_hint_preserves_contents() -> None:
    cache = FullAttentionCache(capacity_hint=2)
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for step in range(6):
        key, value = _kv(step)
        keys.append(key)
        values.append(value)
        out_key, out_value = cache.append(key, value)

    # Grew past the (deliberately too-small) hint, yet contents still match a full concat.
    assert cache.key.shape[2] >= cache.length
    torch.testing.assert_close(out_key, torch.cat(keys, dim=2))
    torch.testing.assert_close(out_value, torch.cat(values, dim=2))


def test_growth_without_hint_preserves_contents() -> None:
    cache = FullAttentionCache()
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for step in range(4):
        key, value = _kv(step)
        keys.append(key)
        values.append(value)
        out_key, out_value = cache.append(key, value)

    torch.testing.assert_close(out_key, torch.cat(keys, dim=2))
    torch.testing.assert_close(out_value, torch.cat(values, dim=2))
    assert cache.length == 4


def test_append_preserves_dtype_and_device() -> None:
    cache = FullAttentionCache(capacity_hint=4)
    key = torch.zeros((1, 2, 1, 4), dtype=torch.float64)
    value = torch.ones((1, 2, 1, 4), dtype=torch.float64)

    out_key, out_value = cache.append(key, value)

    assert out_key.dtype == torch.float64
    assert out_value.dtype == torch.float64
    assert out_key.device == key.device


def test_decode_state_empty_threads_hint_to_full_attention_only() -> None:
    mapping = _mapping(("full_attention", "linear_attention", "full_attention"))

    state = DecodeState.empty(mapping, None, max_seq_len=16)

    full_caches = [layer for layer in state.layers if isinstance(layer, FullAttentionCache)]
    linear_caches = [layer for layer in state.layers if isinstance(layer, LinearAttentionCache)]
    assert len(full_caches) == 2
    assert len(linear_caches) == 1
    assert all(cache.capacity_hint == 16 for cache in full_caches)
    assert state.position_offset == 0


def test_decode_state_empty_without_hint_leaves_capacity_unset() -> None:
    mapping = _mapping(("full_attention",))

    state = DecodeState.empty(mapping)

    full_cache = state.layers[0]
    assert isinstance(full_cache, FullAttentionCache)
    assert full_cache.capacity_hint is None
    assert full_cache.key is None


def test_linear_attention_cache_remains_lazily_allocated() -> None:
    cache = LinearAttentionCache()

    assert cache.state is None
    assert cache.conv_tail is None
