from __future__ import annotations

from pathlib import Path

import pytest
import torch

from checkpoint import TensorInfo
from tp_weights import MappedWeights
from weight_mapping import LanguageModelMapping, ShardedTensor, TensorShard


class _Loader:
    def __init__(self) -> None:
        self.infos = {
            "a": TensorInfo("a", "BF16", (2,), "model.safetensors", 0, 4, 0),
            "b": TensorInfo("b", "BF16", (4,), "model.safetensors", 0, 8, 0),
            "remote": TensorInfo("remote", "BF16", (1,), "model.safetensors", 0, 2, 0),
        }
        self.values = {name: torch.zeros(info.shape) for name, info in self.infos.items()}

    def tensor_info(self, name: str) -> TensorInfo:
        return self.infos[name]

    def tensor_shard(self, mapped: ShardedTensor, *, device: str | None = None) -> torch.Tensor:
        tensor = self.values[mapped.name]
        return tensor if device is None else tensor.to(device)


def test_mapped_weights_preloads_only_mapped_tensors() -> None:
    weights = MappedWeights(_Loader(), _mapping(("a", "b")), device="cpu")

    stats = weights.preload()

    assert stats.tensor_count == 2
    assert stats.bytes == 12
    assert sorted(weights._cache) == ["a", "b"]


def test_mapped_weights_rejects_unmapped_tensor() -> None:
    weights = MappedWeights(_Loader(), _mapping(("a",)), device="cpu")

    with pytest.raises(KeyError, match="not mapped"):
        weights.tensor("remote")


def _mapping(names: tuple[str, ...]) -> LanguageModelMapping:
    a = _mapped("a", (2,), 4)
    b = _mapped("b", (4,), 8)
    return LanguageModelMapping(
        model_dir=Path("."),
        embed_tokens=a,
        final_norm=a,
        lm_head=b,
        layers=(),
        mapped_tensor_names=frozenset(names),
        ignored_tensor_names=frozenset(),
        unmapped_language_tensor_names=(),
    )


def _mapped(name: str, shape: tuple[int, ...], nbytes: int) -> ShardedTensor:
    return ShardedTensor(
        TensorInfo(name, "BF16", shape, "model.safetensors", 0, nbytes, 0),
        TensorShard.replicated(shape),
    )
