from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParallel:
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} out of range for world_size {self.world_size}")

    @property
    def is_sharded(self) -> bool:
        return self.world_size > 1

    def shard_range(self, size: int) -> tuple[int, int]:
        if size % self.world_size != 0:
            raise ValueError(f"size {size} not divisible by world_size {self.world_size}")
        shard = size // self.world_size
        start = self.rank * shard
        return start, shard

    def dim_shard_shape(self, shape: tuple[int, ...], dim: int) -> tuple[int, ...]:
        start, shard = self.shard_range(shape[dim])
        del start
        out = list(shape)
        out[dim] = shard
        return tuple(out)

    def local_expert_count(self, num_experts: int) -> int:
        return self.shard_range(num_experts)[1]

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        start, per_rank = self.shard_range(num_experts)
        return start, start + per_rank

    def owns_expert(self, expert_index: int, num_experts: int) -> bool:
        start, end = self.expert_range(num_experts)
        return start <= expert_index < end

    @property
    def adds_shared_expert(self) -> bool:
        return True
