from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from checkpoint import TensorInfo, build_manifest
from reference_ops import ReferenceWeights, decoder_layer, language_model
from runtime_config import parse_runtime_config
from tensor_parallel import TensorParallel
from tp_runtime import TpLaunchConfig, TpRuntime, TpRuntimeError, mapped_tensor_bytes, tp_decoder_layer, tp_language_model
from loader import TensorLoader
from weight_mapping import ExpertMapping, FullAttentionMapping, LanguageModelMapping, LayerMapping, LinearTensor, MoEMapping, build_language_model_mapping


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


def test_two_rank_safetensors_tp_language_model_matches_dense(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _tp_language_model_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def _tp_language_model_worker(rank: int, tmp_path: Path) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    model_dir = tmp_path / "tiny-full"
    if rank == 0:
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(_safetensors_config()), encoding="utf-8")
        save_file(_safetensors_tensors(), model_dir / "model.safetensors")
    init_method = f"file://{tmp_path / 'safetensors-dist-init'}"
    with TpRuntime(TpLaunchConfig(world_size=2, rank=rank, local_rank=rank, backend="gloo", init_method=init_method, device="cpu")) as runtime:
        runtime.barrier()
        manifest = build_manifest(model_dir)
        config = parse_runtime_config(manifest.config)
        dense_mapping = build_language_model_mapping(manifest, strict=True)
        tp_mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=TensorParallel(world_size=2, rank=rank))
        input_ids = torch.tensor([[0, 3]])
        with TensorLoader(manifest) as loader:
            expected = language_model(input_ids, dense_mapping, config, ReferenceWeights(loader))
            actual = tp_language_model(input_ids, tp_mapping, config, ReferenceWeights(loader), runtime)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)



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



def _safetensors_config() -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 4,
            "vocab_size": 4,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "attn_output_gate": True,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 1.0,
            "rope_parameters": {"rope_theta": 10000},
        }
    }


def _safetensors_tensors() -> dict[str, torch.Tensor]:
    p = "model.language_model.layers.0."
    q_proj = torch.zeros((8, 4), dtype=torch.bfloat16)
    q_proj[0, 0] = 1.0
    q_proj[1, 1] = 1.0
    q_proj[4, 2] = 1.0
    q_proj[5, 3] = 1.0
    k_proj = torch.zeros((2, 4), dtype=torch.bfloat16)
    k_proj[0, 0] = 1.0
    k_proj[1, 1] = 1.0
    v_proj = torch.zeros((2, 4), dtype=torch.bfloat16)
    v_proj[0, 2] = 1.0
    v_proj[1, 3] = 1.0
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [1.0, 1.0, 0.0, 0.0], [-1.0, 0.5, 1.0, 0.0]],
            dtype=torch.bfloat16,
        ),
        "model.language_model.norm.weight": torch.zeros(4, dtype=torch.bfloat16),
        "lm_head.weight": torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [1.0, 1.0, 0.0, 0.0], [-1.0, 0.5, 1.0, 0.0]],
            dtype=torch.bfloat16,
        ),
        p + "input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        p + "post_attention_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
        p + "self_attn.q_proj.weight": q_proj,
        p + "self_attn.k_proj.weight": k_proj,
        p + "self_attn.v_proj.weight": v_proj,
        p + "self_attn.o_proj.weight": torch.eye(4, dtype=torch.bfloat16),
        p + "self_attn.q_norm.weight": torch.zeros(2, dtype=torch.bfloat16),
        p + "self_attn.k_norm.weight": torch.zeros(2, dtype=torch.bfloat16),
        p + "mlp.gate.weight": torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], dtype=torch.bfloat16),
        p + "mlp.shared_expert_gate.weight": torch.full((1, 4), -100.0, dtype=torch.bfloat16),
        p + "mlp.shared_expert.gate_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        p + "mlp.shared_expert.up_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        p + "mlp.shared_expert.down_proj.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
    }
    for expert, scale in ((0, 1.0), (1, 2.0)):
        prefix = p + f"mlp.experts.{expert}."
        tensors[prefix + "gate_proj.weight"] = torch.eye(4, dtype=torch.bfloat16) * scale
        tensors[prefix + "up_proj.weight"] = torch.eye(4, dtype=torch.bfloat16)
        tensors[prefix + "down_proj.weight"] = torch.eye(4, dtype=torch.bfloat16)
    fp8_names = [
        p + "self_attn.q_proj.weight",
        p + "self_attn.k_proj.weight",
        p + "self_attn.v_proj.weight",
        p + "self_attn.o_proj.weight",
        p + "mlp.shared_expert.gate_proj.weight",
        p + "mlp.shared_expert.up_proj.weight",
        p + "mlp.shared_expert.down_proj.weight",
    ]
    fp8_names.extend(
        p + f"mlp.experts.{expert}.{suffix}.weight"
        for expert in (0, 1)
        for suffix in ("gate_proj", "up_proj", "down_proj")
    )
    for name in fp8_names:
        tensors[name] = tensors[name].to(torch.float8_e4m3fn)
        rows, cols = tensors[name].shape
        tensors[name + "_scale_inv"] = torch.ones(((rows + 127) // 128, (cols + 127) // 128), dtype=torch.bfloat16)
    return tensors
