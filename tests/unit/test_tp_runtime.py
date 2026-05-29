from __future__ import annotations

import pytest
import torch

from checkpoint import TensorInfo
from tp_runtime import TpLaunchConfig, TpRuntime, TpRuntimeError, mapped_tensor_bytes
from weight_mapping import LanguageModelMapping


def test_tp_launch_config_validates_rank() -> None:
    with pytest.raises(TpRuntimeError, match="out of range"):
        TpLaunchConfig(world_size=2, rank=2)


def test_single_rank_runtime_all_reduce_is_noop_cpu() -> None:
    tensor = torch.tensor([1.0, 2.0])

    with TpRuntime(TpLaunchConfig(backend="gloo", device="cpu")) as runtime:
        out = runtime.all_reduce_sum(tensor)

    assert out is tensor
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))


def test_mapped_tensor_bytes_counts_local_expert_shard() -> None:
    mapping = LanguageModelMapping(
        model_dir=__import__("pathlib").Path("."),
        embed_tokens=_info("embed", (4, 2), nbytes=16),
        final_norm=_info("norm", (2,), nbytes=4),
        lm_head=_info("head", (4, 2), nbytes=16),
        layers=(),
        mapped_tensor_names=frozenset(),
        ignored_tensor_names=frozenset(),
        unmapped_language_tensor_names=(),
    )

    assert mapped_tensor_bytes(mapping) == 36


def _info(name: str, shape: tuple[int, ...], *, nbytes: int) -> TensorInfo:
    return TensorInfo(name=name, dtype="BF16", shape=shape, shard="model.safetensors", begin=0, end=nbytes, data_start=0)
