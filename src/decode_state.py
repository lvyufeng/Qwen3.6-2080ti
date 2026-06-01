from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from runtime_config import RuntimeConfig
from weight_mapping import LanguageModelMapping


DEFAULT_KV_BLOCK_SIZE = 16


@dataclass
class PagedKVBlockPool:
    """Per-cache fixed-size block pool for full-attention KV.

    Blocks are stored in ``(batch, heads, blocks, block_size, head_dim)`` order so a
    contiguous physical block table can be flattened into the current attention layout
    without a copy. The pool is deliberately per-cache for this foundation phase; a shared
    service-wide pool comes with multi-request batching.
    """

    block_size: int
    key_blocks: Any | None = None
    value_blocks: Any | None = None
    free_blocks: list[int] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return 0 if self.key_blocks is None else self.key_blocks.shape[2]

    def ensure_initialized(self, key: Any, value: Any, blocks: int) -> None:
        import torch

        if blocks <= 0:
            return
        batch, heads, _seq, head_dim = key.shape
        if self.key_blocks is None:
            self.key_blocks = torch.empty(
                (batch, heads, blocks, self.block_size, head_dim), dtype=key.dtype, device=key.device
            )
            self.value_blocks = torch.empty(
                (batch, heads, blocks, self.block_size, head_dim), dtype=value.dtype, device=value.device
            )
            self.free_blocks = list(range(blocks))

    def add_blocks(self, key: Any, value: Any, blocks: int) -> None:
        import torch

        if blocks <= 0:
            return
        if self.key_blocks is None:
            self.ensure_initialized(key, value, blocks)
            return
        batch, heads, _seq, head_dim = key.shape
        old_blocks = self.block_count
        new_blocks = old_blocks + blocks
        grown_key = torch.empty(
            (batch, heads, new_blocks, self.block_size, head_dim), dtype=self.key_blocks.dtype, device=self.key_blocks.device
        )
        grown_value = torch.empty(
            (batch, heads, new_blocks, self.block_size, head_dim),
            dtype=self.value_blocks.dtype,
            device=self.value_blocks.device,
        )
        grown_key[:, :, :old_blocks] = self.key_blocks
        grown_value[:, :, :old_blocks] = self.value_blocks
        self.key_blocks = grown_key
        self.value_blocks = grown_value
        self.free_blocks.extend(range(old_blocks, new_blocks))

    def allocate(self, count: int, key: Any, value: Any) -> list[int]:
        self.ensure_initialized(key, value, count)
        if count > len(self.free_blocks):
            self.add_blocks(key, value, count - len(self.free_blocks))
        allocated = self.free_blocks[:count]
        del self.free_blocks[:count]
        return allocated

    def release(self, blocks: list[int]) -> None:
        if not blocks:
            return
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()


@dataclass(frozen=True)
class FullAttentionCacheStats:
    block_size: int
    valid_tokens: int
    capacity_tokens: int
    logical_blocks: int
    allocated_blocks: int
    free_blocks: int
    append_calls: int
    appended_tokens: int
    growth_events: int
    release_calls: int
    released_blocks: int
    contiguous_view_calls: int
    gather_view_calls: int
    gather_copied_tokens: int
    non_contiguous: bool
    estimated_key_bytes: int
    estimated_value_bytes: int
    estimated_total_bytes: int

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "block_size": self.block_size,
            "valid_tokens": self.valid_tokens,
            "capacity_tokens": self.capacity_tokens,
            "logical_blocks": self.logical_blocks,
            "allocated_blocks": self.allocated_blocks,
            "free_blocks": self.free_blocks,
            "append_calls": self.append_calls,
            "appended_tokens": self.appended_tokens,
            "growth_events": self.growth_events,
            "release_calls": self.release_calls,
            "released_blocks": self.released_blocks,
            "contiguous_view_calls": self.contiguous_view_calls,
            "gather_view_calls": self.gather_view_calls,
            "gather_copied_tokens": self.gather_copied_tokens,
            "non_contiguous": self.non_contiguous,
            "estimated_key_bytes": self.estimated_key_bytes,
            "estimated_value_bytes": self.estimated_value_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
        }


@dataclass(frozen=True)
class KVCacheStats:
    full_attention_layers: int
    block_size: int | None
    valid_tokens_total: int
    capacity_tokens_total: int
    logical_blocks_total: int
    allocated_blocks_total: int
    free_blocks_total: int
    append_calls_total: int
    appended_tokens_total: int
    growth_events_total: int
    release_calls_total: int
    released_blocks_total: int
    contiguous_view_calls_total: int
    gather_view_calls_total: int
    gather_copied_tokens_total: int
    non_contiguous_layers: int
    estimated_total_bytes: int
    max_valid_tokens: int
    max_capacity_tokens: int

    def to_dict(self) -> dict[str, int | None]:
        return {
            "full_attention_layers": self.full_attention_layers,
            "block_size": self.block_size,
            "valid_tokens_total": self.valid_tokens_total,
            "capacity_tokens_total": self.capacity_tokens_total,
            "logical_blocks_total": self.logical_blocks_total,
            "allocated_blocks_total": self.allocated_blocks_total,
            "free_blocks_total": self.free_blocks_total,
            "append_calls_total": self.append_calls_total,
            "appended_tokens_total": self.appended_tokens_total,
            "growth_events_total": self.growth_events_total,
            "release_calls_total": self.release_calls_total,
            "released_blocks_total": self.released_blocks_total,
            "contiguous_view_calls_total": self.contiguous_view_calls_total,
            "gather_view_calls_total": self.gather_view_calls_total,
            "gather_copied_tokens_total": self.gather_copied_tokens_total,
            "non_contiguous_layers": self.non_contiguous_layers,
            "estimated_total_bytes": self.estimated_total_bytes,
            "max_valid_tokens": self.max_valid_tokens,
            "max_capacity_tokens": self.max_capacity_tokens,
        }


@dataclass
class FullAttentionCache:
    """Paged per-layer KV cache for full attention.

    The public contract stays the same as the Phase M cache: ``append`` accepts new
    post-rotary key/value tensors in head-major layout ``(batch, heads, seq, head_dim)`` and
    returns the full valid KV region in that same layout. Internally, KV is stored in fixed
    blocks and ``block_table`` maps logical token blocks to physical pool blocks. A
    contiguous physical table uses a cheap view; non-contiguous tables use a gather path that
    is tested now and becomes the future batching path.
    """

    capacity_hint: int | None = None
    block_size: int = DEFAULT_KV_BLOCK_SIZE
    valid: int = 0
    block_table: list[int] = field(default_factory=list)
    pool: PagedKVBlockPool = field(init=False)
    append_calls: int = 0
    appended_tokens: int = 0
    growth_events: int = 0
    release_calls: int = 0
    released_blocks: int = 0
    contiguous_view_calls: int = 0
    gather_view_calls: int = 0
    gather_copied_tokens: int = 0

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.pool = PagedKVBlockPool(self.block_size)

    @property
    def key_blocks(self) -> Any | None:
        return self.pool.key_blocks

    @property
    def value_blocks(self) -> Any | None:
        return self.pool.value_blocks

    @property
    def length(self) -> int:
        return self.valid

    @property
    def capacity(self) -> int:
        return len(self.block_table) * self.block_size

    def append(self, key: Any, value: Any) -> tuple[Any, Any]:
        batch, heads, seq, head_dim = key.shape
        self.append_calls += 1
        self.appended_tokens += int(seq)
        needed = self.valid + seq
        self._ensure_logical_capacity(needed, key, value)
        source_offset = 0
        write_pos = self.valid
        while source_offset < seq:
            logical_block = write_pos // self.block_size
            block_offset = write_pos % self.block_size
            take = min(seq - source_offset, self.block_size - block_offset)
            physical_block = self.block_table[logical_block]
            self.pool.key_blocks[:, :, physical_block, block_offset : block_offset + take] = key[
                :, :, source_offset : source_offset + take
            ]
            self.pool.value_blocks[:, :, physical_block, block_offset : block_offset + take] = value[
                :, :, source_offset : source_offset + take
            ]
            source_offset += take
            write_pos += take
        self.valid = needed
        return self.as_tensors()

    def as_tensors(self) -> tuple[Any, Any]:
        if self.valid == 0:
            raise ValueError("FullAttentionCache has no valid tokens")
        if self._is_contiguous_table():
            self.contiguous_view_calls += 1
            return self._contiguous_view(self.pool.key_blocks), self._contiguous_view(self.pool.value_blocks)
        self.gather_view_calls += 1
        self.gather_copied_tokens += self.valid
        return self._gather_blocks(self.pool.key_blocks), self._gather_blocks(self.pool.value_blocks)

    def release(self) -> None:
        self.release_calls += 1
        self.released_blocks += len(self.block_table)
        self.pool.release(self.block_table)
        self.block_table.clear()
        self.valid = 0

    def stats(self) -> FullAttentionCacheStats:
        key_bytes = _tensor_nbytes(self.pool.key_blocks)
        value_bytes = _tensor_nbytes(self.pool.value_blocks)
        return FullAttentionCacheStats(
            block_size=self.block_size,
            valid_tokens=self.valid,
            capacity_tokens=self.capacity,
            logical_blocks=len(self.block_table),
            allocated_blocks=self.pool.block_count,
            free_blocks=len(self.pool.free_blocks),
            append_calls=self.append_calls,
            appended_tokens=self.appended_tokens,
            growth_events=self.growth_events,
            release_calls=self.release_calls,
            released_blocks=self.released_blocks,
            contiguous_view_calls=self.contiguous_view_calls,
            gather_view_calls=self.gather_view_calls,
            gather_copied_tokens=self.gather_copied_tokens,
            non_contiguous=not self._is_contiguous_table(),
            estimated_key_bytes=key_bytes,
            estimated_value_bytes=value_bytes,
            estimated_total_bytes=key_bytes + value_bytes,
        )

    def _ensure_logical_capacity(self, needed: int, key: Any, value: Any) -> None:
        if needed <= self.capacity:
            return
        target_tokens = max(needed, self.capacity_hint or 0, max(self.block_size, self.capacity * 2))
        target_blocks = _ceil_div(target_tokens, self.block_size)
        missing = target_blocks - len(self.block_table)
        if missing <= 0:
            return
        self.growth_events += 1
        self.block_table.extend(self.pool.allocate(missing, key, value))

    def _is_contiguous_table(self) -> bool:
        if not self.block_table:
            return True
        start = self.block_table[0]
        return self.block_table == list(range(start, start + len(self.block_table)))

    def _contiguous_view(self, blocks: Any) -> Any:
        logical_blocks = len(self.block_table)
        batch = blocks.shape[0]
        heads = blocks.shape[1]
        head_dim = blocks.shape[4]
        start = self.block_table[0]
        flat = blocks.as_strided(
            (batch, heads, logical_blocks * self.block_size, head_dim),
            (blocks.stride(0), blocks.stride(1), blocks.stride(3), blocks.stride(4)),
            storage_offset=blocks.storage_offset() + start * blocks.stride(2),
        )
        return flat[:, :, : self.valid]

    def _gather_blocks(self, blocks: Any) -> Any:
        import torch

        index = torch.tensor(self.block_table, dtype=torch.long, device=blocks.device)
        gathered = blocks.index_select(2, index)
        batch = gathered.shape[0]
        heads = gathered.shape[1]
        head_dim = gathered.shape[4]
        flat = gathered.reshape(batch, heads, len(self.block_table) * self.block_size, head_dim)
        return flat[:, :, : self.valid]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _tensor_nbytes(tensor: Any | None) -> int:
    if tensor is None:
        return 0
    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return int(tensor.numel() * tensor.element_size())


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
        *,
        kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    ) -> DecodeState:
        del config  # caches allocate lazily; only the per-layer type matters
        layers: list[LayerCache] = []
        for layer in mapping.layers:
            if layer.layer_type == "full_attention":
                layers.append(FullAttentionCache(capacity_hint=max_seq_len, block_size=kv_block_size))
            else:
                layers.append(LinearAttentionCache())
        return cls(layers=layers, position_offset=0)

    def advance(self, tokens: int) -> None:
        self.position_offset += tokens

    def release(self) -> None:
        for layer in self.layers:
            if isinstance(layer, FullAttentionCache):
                layer.release()

    def kv_stats(self) -> KVCacheStats:
        full_stats = [layer.stats() for layer in self.layers if isinstance(layer, FullAttentionCache)]
        block_sizes = {stats.block_size for stats in full_stats}
        block_size = next(iter(block_sizes)) if len(block_sizes) == 1 else None
        return KVCacheStats(
            full_attention_layers=len(full_stats),
            block_size=block_size,
            valid_tokens_total=sum(stats.valid_tokens for stats in full_stats),
            capacity_tokens_total=sum(stats.capacity_tokens for stats in full_stats),
            logical_blocks_total=sum(stats.logical_blocks for stats in full_stats),
            allocated_blocks_total=sum(stats.allocated_blocks for stats in full_stats),
            free_blocks_total=sum(stats.free_blocks for stats in full_stats),
            append_calls_total=sum(stats.append_calls for stats in full_stats),
            appended_tokens_total=sum(stats.appended_tokens for stats in full_stats),
            growth_events_total=sum(stats.growth_events for stats in full_stats),
            release_calls_total=sum(stats.release_calls for stats in full_stats),
            released_blocks_total=sum(stats.released_blocks for stats in full_stats),
            contiguous_view_calls_total=sum(stats.contiguous_view_calls for stats in full_stats),
            gather_view_calls_total=sum(stats.gather_view_calls for stats in full_stats),
            gather_copied_tokens_total=sum(stats.gather_copied_tokens for stats in full_stats),
            non_contiguous_layers=sum(1 for stats in full_stats if stats.non_contiguous),
            estimated_total_bytes=sum(stats.estimated_total_bytes for stats in full_stats),
            max_valid_tokens=max((stats.valid_tokens for stats in full_stats), default=0),
            max_capacity_tokens=max((stats.capacity_tokens for stats in full_stats), default=0),
        )


def batch_kv_tensors(caches: Sequence[FullAttentionCache]) -> tuple[Any, Any, Any]:
    """Stack KV from multiple per-request caches, padding to max valid length.

    Each cache has batch=1. Returns:
      keys:       (B, heads, max_valid, head_dim)
      values:     (B, heads, max_valid, head_dim)
      valid_mask: (B, max_valid) bool — True for valid positions
    """
    import torch

    if not caches:
        raise ValueError("batch_kv_tensors requires at least one cache")
    valids = [cache.valid for cache in caches]
    max_valid = max(valids)
    if max_valid == 0:
        raise ValueError("batch_kv_tensors requires at least one non-empty cache")

    # Get shape info from first cache
    first_k, first_v = caches[0].as_tensors()
    # first_k shape: (1, heads, valid_0, head_dim)
    heads = first_k.shape[1]
    head_dim = first_k.shape[3]
    device = first_k.device
    dtype = first_k.dtype
    batch = len(caches)

    keys = torch.zeros(batch, heads, max_valid, head_dim, device=device, dtype=dtype)
    values = torch.zeros(batch, heads, max_valid, head_dim, device=device, dtype=dtype)
    valid_mask = torch.zeros(batch, max_valid, device=device, dtype=torch.bool)

    for i, cache in enumerate(caches):
        v = cache.valid
        if i == 0:
            k_i, v_i = first_k, first_v
        else:
            k_i, v_i = cache.as_tensors()
        # k_i shape: (1, heads, valid_i, head_dim)
        keys[i : i + 1, :, :v] = k_i
        values[i : i + 1, :, :v] = v_i
        valid_mask[i, :v] = True

    return keys, values, valid_mask
