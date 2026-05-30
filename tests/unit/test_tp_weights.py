from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from checkpoint import TensorInfo
from reference_ops import LinearDispatchStats, linear
from tensor_parallel import TensorParallel
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for resident FP8 dispatch")
def test_mapped_weights_resident_tp4_fp8_linear_hits_cuda_kernel() -> None:
    try:
        from fp8_cuda import fp8_e4m3_bf16_linear  # noqa: F401
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    torch.manual_seed(0)
    loader = _Loader()
    loader.infos.update(
        {
            "fp8_w": TensorInfo("fp8_w", "F8_E4M3", (512, 512), "model.safetensors", 0, 512 * 512, 0),
            "fp8_s": TensorInfo("fp8_s", "BF16", (4, 4), "model.safetensors", 0, 4 * 4 * 2, 0),
        }
    )
    loader.values.update(
        {
            "fp8_w": (torch.randn((128, 512), device=device, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn),
            "fp8_s": torch.full((1, 4), 0.25, device=device, dtype=torch.bfloat16),
        }
    )
    tp = TensorParallel(world_size=4, rank=0)
    weight = ShardedTensor(
        loader.infos["fp8_w"],
        TensorShard.dim_shard("column_parallel", loader.infos["fp8_w"].shape, dim=0, tp=tp),
    )
    scale = ShardedTensor(
        loader.infos["fp8_s"],
        TensorShard(rule="column_parallel", dim=0, start=0, size=1, local_shape=(1, 4)),
    )
    fp8_linear = LinearTensor(weight=weight, scale=scale)
    weights = MappedWeights(loader, _mapping(("a", "b")), device=device)
    weights._tensors.update({"fp8_w": weight, "fp8_s": scale})

    stats = weights.preload()
    resident_weight, resident_scale = weights.linear_weight(fp8_linear)

    assert stats.tensor_count == 4
    assert resident_weight.is_cuda
    assert resident_scale is not None and resident_scale.is_cuda
    assert resident_weight.dtype == torch.float8_e4m3fn
    assert resident_scale.dtype == torch.bfloat16
    assert resident_weight.shape == (128, 512)
    assert resident_scale.shape == (1, 4)

    for batch in (3, 17):
        weights.dispatch_stats = LinearDispatchStats()
        hidden = torch.randn((batch, 512), device=device, dtype=torch.float32)

        actual = weights.linear(hidden, fp8_linear)
        expected = linear(hidden, resident_weight, resident_scale, use_cuda_kernel=False)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        assert weights.dispatch_stats.calls == 1
        assert weights.dispatch_stats.fp8_weight_calls == 1
        assert weights.dispatch_stats.eligible_cuda_calls == 1
        assert weights.dispatch_stats.cuda_kernel_hits == 1
        assert weights.dispatch_stats.fallback_calls == 0


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
