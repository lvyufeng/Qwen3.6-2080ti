from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from checkpoint import TensorInfo
from tp_weights import MappedWeights
from weight_mapping import ExpertMapping, LanguageModelMapping, LinearTensor, ShardedTensor, TensorShard


class _Loader:
    def __init__(self) -> None:
        self.infos = {
            "a": TensorInfo("a", "BF16", (2,), "model.safetensors", 0, 4, 0),
            "b": TensorInfo("b", "BF16", (4,), "model.safetensors", 0, 8, 0),
            "remote": TensorInfo("remote", "BF16", (1,), "model.safetensors", 0, 2, 0),
        }
        self.values = {name: torch.zeros(info.shape) for name, info in self.infos.items()}
        self.calls: list[str] = []

    def tensor_info(self, name: str) -> TensorInfo:
        return self.infos[name]

    def tensor_shard(self, mapped: ShardedTensor, *, device: str | None = None) -> torch.Tensor:
        self.calls.append(mapped.name)
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


def test_mapped_weights_tensor_accepts_sharded_tensor_and_name_object() -> None:
    loader = _Loader()
    weights = MappedWeights(loader, _mapping(("a", "b")), device="cpu")
    mapped = weights.mapping.embed_tokens

    by_shard = weights.tensor(mapped)
    by_name_object = weights.tensor(SimpleNamespace(name="a"))
    by_string = weights.tensor("a")

    assert by_shard is by_name_object
    assert by_string is by_shard
    assert loader.calls == ["a"]


def test_mapped_weights_linear_weight_handles_optional_scale() -> None:
    weights = MappedWeights(_Loader(), _mapping(("a", "b")), device="cpu")
    linear = LinearTensor(weight=weights.mapping.lm_head, scale=weights.mapping.embed_tokens)

    weight, scale = weights.linear_weight(linear)
    head, no_scale = weights.linear_weight(LinearTensor(weight=weights.mapping.lm_head, scale=None))

    assert weight.shape == (4,)
    assert scale is weights.tensor("a")
    assert head is weight
    assert no_scale is None


def test_mapped_weights_linear_and_expert_reuse_preloaded_cache() -> None:
    loader = _Loader()
    loader.infos.update(
        {
            "w": TensorInfo("w", "BF16", (2, 2), "model.safetensors", 0, 8, 0),
            "gate": TensorInfo("gate", "BF16", (2, 2), "model.safetensors", 0, 8, 0),
            "up": TensorInfo("up", "BF16", (2, 2), "model.safetensors", 0, 8, 0),
            "down": TensorInfo("down", "BF16", (2, 2), "model.safetensors", 0, 8, 0),
        }
    )
    loader.values.update(
        {
            "w": torch.eye(2),
            "gate": torch.eye(2),
            "up": torch.eye(2),
            "down": torch.eye(2),
        }
    )
    mapping = _mapping(("a", "b"))
    weights = MappedWeights(loader, mapping, device="cpu")
    weights._tensors.update({name: _mapped(name, (2, 2), 8) for name in ("w", "gate", "up", "down")})

    weights.preload()
    calls_after_preload = list(loader.calls)
    hidden = torch.tensor([[1.0, 2.0]])
    out = weights.linear(hidden, LinearTensor(weight=weights._tensors["w"], scale=None))
    expert = ExpertMapping(0, LinearTensor(weights._tensors["gate"], None), LinearTensor(weights._tensors["up"], None), LinearTensor(weights._tensors["down"], None))
    expert_out = weights.expert(hidden, expert)

    torch.testing.assert_close(out, hidden)
    assert expert_out.shape == hidden.shape
    assert loader.calls == calls_after_preload
    assert weights.dispatch_stats.calls == 4
    assert weights.dispatch_stats.fallback_calls == 4
    assert weights.dispatch_stats.fallback_weight_dtype == 4
    assert weights.dispatch_stats.cuda_kernel_hits == 0


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
