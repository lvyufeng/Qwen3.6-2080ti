from __future__ import annotations

from pathlib import Path

import pytest
import torch

from checkpoint import TensorInfo
from reference_ops import ReferenceWeights, decoder_layer
from runtime_config import parse_runtime_config
from tensor_parallel import TensorParallel
from tp_runtime import TpLaunchConfig, TpRuntime, TpRuntimeError, mapped_tensor_bytes, tp_decoder_layer
from weight_mapping import ExpertMapping, FullAttentionMapping, LanguageModelMapping, LayerMapping, LinearTensor, MoEMapping


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


def test_two_rank_tp_decoder_layer_matches_dense_reference(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_decoder_layer_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def _tp_decoder_layer_worker(rank: int, tmp_path: Path) -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    q = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    o = torch.eye(2)
    loader = _FakeLoader(
        {
            "input_norm": torch.zeros(2),
            "q": q,
            "q_r0": q[:2].contiguous(),
            "q_r1": q[2:].contiguous(),
            "k": torch.tensor([[1.0, 0.0]]),
            "v": torch.tensor([[0.0, 1.0]]),
            "o": o,
            "o_r0": o[:, :1].contiguous(),
            "o_r1": o[:, 1:].contiguous(),
            "q_norm": torch.zeros(1),
            "k_norm": torch.zeros(1),
            "post_norm": torch.zeros(2),
            "gate": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
            "shared_gate": torch.tensor([[-100.0, -100.0]]),
            "e0_gate": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e0_up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e0_down": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e1_gate": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "e1_up": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "e1_down": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "shared_gate_proj": torch.zeros((2, 2)),
            "shared_up_proj": torch.zeros((2, 2)),
            "shared_down_proj": torch.zeros((2, 2)),
        }
    )
    config = parse_runtime_config(_config())
    init_method = f"file://{tmp_path / 'dist-init'}"
    dense_mapping = _layer_mapping((0, 1), TensorParallel(world_size=1, rank=0))
    tp_mapping = _layer_mapping((rank,), TensorParallel(world_size=2, rank=rank))

    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        tp_out = tp_decoder_layer(hidden, tp_mapping, config, ReferenceWeights(loader), runtime)
        expected = decoder_layer(hidden, dense_mapping, config, ReferenceWeights(loader))
        torch.testing.assert_close(tp_out, expected)


def _layer_mapping(local_experts: tuple[int, ...], tp: TensorParallel) -> LayerMapping:
    return LayerMapping(
        index=0,
        layer_type="full_attention",
        input_layernorm=_info("input_norm", (2,), nbytes=4),
        attention=FullAttentionMapping(
            q_proj=_linear_shape("q" if tp.world_size == 1 else f"q_r{tp.rank}", (4 // tp.world_size, 2) if tp.world_size > 1 else (4, 2)),
            k_proj=_linear_shape("k", (1, 2)),
            v_proj=_linear_shape("v", (1, 2)),
            o_proj=_linear_shape("o" if tp.world_size == 1 else f"o_r{tp.rank}", (2, 2 // tp.world_size) if tp.world_size > 1 else (2, 2)),
            q_norm=_info("q_norm", (1,), nbytes=2),
            k_norm=_info("k_norm", (1,), nbytes=2),
        ),
        post_attention_layernorm=_info("post_norm", (2,), nbytes=4),
        mlp=_moe_mapping(local_experts, tp),
    )


def _moe_mapping(local_experts: tuple[int, ...], tp: TensorParallel) -> MoEMapping:
    names = {0: "e0", 1: "e1"}
    return MoEMapping(
        gate=_info("gate", (2, 2), nbytes=8),
        experts=tuple(
            ExpertMapping(i, _linear(f"{names[i]}_gate"), _linear(f"{names[i]}_up"), _linear(f"{names[i]}_down"))
            for i in local_experts
        ),
        shared_expert=ExpertMapping(-1, _linear("shared_gate_proj"), _linear("shared_up_proj"), _linear("shared_down_proj")),
        shared_expert_gate=_info("shared_gate", (1, 2), nbytes=4),
        expert_start=local_experts[0],
        expert_end=local_experts[-1] + 1,
        num_experts=2,
        tp=tp,
    )


def _linear(name: str) -> LinearTensor:
    return _linear_shape(name, (2, 2))


def _linear_shape(name: str, shape: tuple[int, int]) -> LinearTensor:
    return LinearTensor(weight=_info(name, shape, nbytes=8), scale=None)


def _info(name: str, shape: tuple[int, ...], *, nbytes: int) -> TensorInfo:
    return TensorInfo(name=name, dtype="BF16", shape=shape, shard="model.safetensors", begin=0, end=nbytes, data_start=0)


class _FakeLoader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def tensor(self, name: str, *, device: str | None = None) -> torch.Tensor:
        tensor = self.tensors[name]
        return tensor if device is None else tensor.to(device)


def _config() -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2,
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "attn_output_gate": True,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000},
        }
    }

