from __future__ import annotations

from types import SimpleNamespace

import torch

from decode_state import DEFAULT_KV_BLOCK_SIZE, DecodeState, FullAttentionCache, LinearAttentionCache, batch_kv_tensors


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
    cache = FullAttentionCache(block_size=2)
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for step in range(5):
        key, value = _kv(step)
        keys.append(key)
        values.append(value)
        out_key, out_value = cache.append(key, value)
        torch.testing.assert_close(out_key, torch.cat(keys, dim=2))
        torch.testing.assert_close(out_value, torch.cat(values, dim=2))


def test_append_handles_prefill_chunk_then_single_steps_across_blocks() -> None:
    cache = FullAttentionCache(block_size=2)
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
    assert cache.block_table == [0, 1, 2, 3]
    assert cache.capacity == 8


def test_length_tracks_total_appended_tokens_not_allocated_capacity() -> None:
    cache = FullAttentionCache(capacity_hint=8, block_size=4)
    assert cache.length == 0
    assert cache.capacity == 0

    cache.append(*_kv(0, seq=3))
    assert cache.length == 3
    assert cache.capacity == 8

    cache.append(*_kv(1, seq=1))
    cache.append(*_kv(2, seq=1))
    assert cache.length == 5
    assert cache.capacity == 8


def test_capacity_hint_allocates_expected_contiguous_blocks() -> None:
    cache = FullAttentionCache(capacity_hint=8, block_size=4)

    out_key, out_value = cache.append(*_kv(0, seq=3))

    assert cache.pool.block_count == 2
    assert cache.block_table == [0, 1]
    assert cache.pool.free_blocks == []
    assert cache.key_blocks.shape[2] == 2
    assert cache.value_blocks.shape[2] == 2
    assert out_key.shape == (1, 2, 3, 4)
    assert out_value.shape == (1, 2, 3, 4)


def test_contiguous_fast_path_returns_view_of_block_storage() -> None:
    cache = FullAttentionCache(capacity_hint=8, block_size=4)

    out_key, out_value = cache.append(*_kv(0, seq=3))
    key_ptr = cache.key_blocks.data_ptr()
    value_ptr = cache.value_blocks.data_ptr()

    assert out_key.data_ptr() == key_ptr
    assert out_value.data_ptr() == value_ptr
    assert out_key._base is not None
    assert out_value._base is not None

    out_key, out_value = cache.append(*_kv(1, seq=2))
    assert cache.key_blocks.data_ptr() == key_ptr
    assert cache.value_blocks.data_ptr() == value_ptr
    assert out_key.data_ptr() == key_ptr
    assert out_value.data_ptr() == value_ptr
    assert out_key.shape[2] == 5


def test_growth_beyond_hint_allocates_blocks_and_preserves_contents() -> None:
    cache = FullAttentionCache(capacity_hint=2, block_size=2)
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for step in range(6):
        key, value = _kv(step)
        keys.append(key)
        values.append(value)
        out_key, out_value = cache.append(key, value)

    assert cache.pool.block_count >= len(cache.block_table)
    assert cache.capacity >= cache.length
    torch.testing.assert_close(out_key, torch.cat(keys, dim=2))
    torch.testing.assert_close(out_value, torch.cat(values, dim=2))


def test_release_clears_logical_state_and_reuses_blocks() -> None:
    cache = FullAttentionCache(capacity_hint=4, block_size=2)
    cache.append(*_kv(0, seq=3))
    released_blocks = list(cache.block_table)
    key_storage_ptr = cache.key_blocks.data_ptr()

    cache.release()

    assert cache.length == 0
    assert cache.block_table == []
    assert cache.pool.free_blocks == released_blocks

    out_key, _out_value = cache.append(*_kv(1, seq=2))
    assert cache.block_table == released_blocks
    assert cache.key_blocks.data_ptr() == key_storage_ptr
    torch.testing.assert_close(out_key, _kv(1, seq=2)[0])


def test_full_attention_cache_stats_track_append_and_contiguous_views() -> None:
    cache = FullAttentionCache(capacity_hint=4, block_size=2)

    cache.append(*_kv(0, seq=3))
    cache.as_tensors()

    stats = cache.stats()
    assert stats.append_calls == 1
    assert stats.appended_tokens == 3
    assert stats.growth_events == 1
    assert stats.contiguous_view_calls == 2
    assert stats.gather_view_calls == 0
    assert stats.valid_tokens == 3
    assert stats.capacity_tokens == 4
    assert stats.estimated_total_bytes > 0



def test_non_contiguous_block_table_gather_matches_logical_order() -> None:
    cache = FullAttentionCache(capacity_hint=6, block_size=2)
    first_key, first_value = _kv(0, seq=2)
    second_key, second_value = _kv(1, seq=2)
    third_key, third_value = _kv(2, seq=2)
    cache.append(first_key, first_value)
    cache.append(second_key, second_value)
    cache.append(third_key, third_value)

    cache.block_table = [2, 0, 1]
    cache.pool.key_blocks[:, :, 2] = first_key
    cache.pool.value_blocks[:, :, 2] = first_value
    cache.pool.key_blocks[:, :, 0] = second_key
    cache.pool.value_blocks[:, :, 0] = second_value
    cache.pool.key_blocks[:, :, 1] = third_key
    cache.pool.value_blocks[:, :, 1] = third_value

    out_key, out_value = cache.as_tensors()

    torch.testing.assert_close(out_key, torch.cat([first_key, second_key, third_key], dim=2))
    torch.testing.assert_close(out_value, torch.cat([first_value, second_value, third_value], dim=2))
    stats = cache.stats()
    assert stats.gather_view_calls == 1
    assert stats.gather_copied_tokens == 6
    assert stats.non_contiguous is True


def test_append_preserves_dtype_and_device() -> None:
    cache = FullAttentionCache(capacity_hint=4)
    key = torch.zeros((1, 2, 1, 4), dtype=torch.float64)
    value = torch.ones((1, 2, 1, 4), dtype=torch.float64)

    out_key, out_value = cache.append(key, value)

    assert out_key.dtype == torch.float64
    assert out_value.dtype == torch.float64
    assert out_key.device == key.device


def test_decode_state_empty_threads_hint_and_block_size_to_full_attention_only() -> None:
    mapping = _mapping(("full_attention", "linear_attention", "full_attention"))

    state = DecodeState.empty(mapping, None, max_seq_len=16, kv_block_size=4)

    full_caches = [layer for layer in state.layers if isinstance(layer, FullAttentionCache)]
    linear_caches = [layer for layer in state.layers if isinstance(layer, LinearAttentionCache)]
    assert len(full_caches) == 2
    assert len(linear_caches) == 1
    assert all(cache.capacity_hint == 16 for cache in full_caches)
    assert all(cache.block_size == 4 for cache in full_caches)
    assert state.position_offset == 0


def test_decode_state_empty_without_hint_leaves_pool_unallocated() -> None:
    mapping = _mapping(("full_attention",))

    state = DecodeState.empty(mapping)

    full_cache = state.layers[0]
    assert isinstance(full_cache, FullAttentionCache)
    assert full_cache.capacity_hint is None
    assert full_cache.block_size == DEFAULT_KV_BLOCK_SIZE
    assert full_cache.key_blocks is None


def test_decode_state_kv_stats_aggregates_full_attention_layers() -> None:
    mapping = _mapping(("full_attention", "linear_attention", "full_attention"))
    state = DecodeState.empty(mapping, None, max_seq_len=4, kv_block_size=2)
    for layer in state.layers:
        if isinstance(layer, FullAttentionCache):
            layer.append(*_kv(layer.block_size, seq=2))

    stats = state.kv_stats()

    assert stats.full_attention_layers == 2
    assert stats.block_size == 2
    assert stats.valid_tokens_total == 4
    assert stats.capacity_tokens_total == 8
    assert stats.append_calls_total == 2
    assert stats.contiguous_view_calls_total == 2
    assert stats.estimated_total_bytes > 0
    assert stats.max_valid_tokens == 2



def test_decode_state_release_releases_full_attention_caches_only() -> None:
    mapping = _mapping(("full_attention", "linear_attention", "full_attention"))
    state = DecodeState.empty(mapping, None, max_seq_len=4, kv_block_size=2)
    for layer in state.layers:
        if isinstance(layer, FullAttentionCache):
            layer.append(*_kv(layer.block_size, seq=2))

    state.release()

    for layer in state.layers:
        if isinstance(layer, FullAttentionCache):
            assert layer.length == 0
            assert layer.block_table == []
            assert layer.pool.free_blocks == [0, 1]
        else:
            assert isinstance(layer, LinearAttentionCache)
            assert layer.state is None
            assert layer.conv_tail is None


def test_linear_attention_cache_remains_lazily_allocated() -> None:
    cache = LinearAttentionCache()

    assert cache.state is None
    assert cache.conv_tail is None


def test_batch_kv_tensors_pads_and_masks_different_valid_lengths() -> None:
    """batch_kv_tensors stacks KV from caches with different fill levels."""
    cache_a = FullAttentionCache(block_size=4)
    cache_b = FullAttentionCache(block_size=4)

    # Cache A has 3 tokens, cache B has 5 tokens
    for step in range(3):
        cache_a.append(*_kv(step, heads=2, head_dim=4))
    for step in range(5):
        cache_b.append(*_kv(step + 10, heads=2, head_dim=4))

    keys, values, valid_mask = batch_kv_tensors([cache_a, cache_b])

    # Shape: (2, heads=2, max_valid=5, head_dim=4)
    assert keys.shape == (2, 2, 5, 4)
    assert values.shape == (2, 2, 5, 4)
    assert valid_mask.shape == (2, 5)

    # Mask correctness
    assert valid_mask[0].tolist() == [True, True, True, False, False]
    assert valid_mask[1].tolist() == [True, True, True, True, True]

    # Data correctness: cache_a's 3 tokens match
    expected_k_a, expected_v_a = cache_a.as_tensors()
    torch.testing.assert_close(keys[0:1, :, :3], expected_k_a)
    torch.testing.assert_close(values[0:1, :, :3], expected_v_a)

    # Padding is zero
    assert (keys[0, :, 3:] == 0).all()
    assert (values[0, :, 3:] == 0).all()

    # Cache B's 5 tokens match
    expected_k_b, expected_v_b = cache_b.as_tensors()
    torch.testing.assert_close(keys[1:2, :, :5], expected_k_b)
    torch.testing.assert_close(values[1:2, :, :5], expected_v_b)


def test_batch_kv_tensors_single_cache() -> None:
    """batch_kv_tensors works with a single cache (degenerate batch)."""
    cache = FullAttentionCache(block_size=2)
    cache.append(*_kv(0, heads=2, head_dim=4))
    cache.append(*_kv(1, heads=2, head_dim=4))

    keys, values, valid_mask = batch_kv_tensors([cache])

    assert keys.shape == (1, 2, 2, 4)
    assert valid_mask.shape == (1, 2)
    assert valid_mask[0].tolist() == [True, True]
    expected_k, expected_v = cache.as_tensors()
    torch.testing.assert_close(keys[0:1], expected_k)
    torch.testing.assert_close(values[0:1], expected_v)
