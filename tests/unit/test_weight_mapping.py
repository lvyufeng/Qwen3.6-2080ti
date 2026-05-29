from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from checkpoint import build_manifest
from tensor_parallel import TensorParallel
from weight_mapping import build_language_model_mapping


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _numel(shape) * _dtype_size(dtype)
        data = bytes(nbytes)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        payloads.append(data)
        offset += nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payloads))


def test_build_language_model_mapping_for_mixed_layers(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_mapping_config()), encoding="utf-8")
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_linear_attention_layer(tensors, 0)
    add_full_attention_layer(tensors, 1)
    add_moe(tensors, 0)
    add_moe(tensors, 1)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    mapping = build_language_model_mapping(build_manifest(tmp_path))

    assert len(mapping.layers) == 2
    assert mapping.linear_attention_layers == 1
    assert mapping.full_attention_layers == 1
    assert mapping.experts_per_layer == 2
    assert mapping.routed_experts == 4
    assert mapping.unmapped_language_tensor_names == ()


def test_build_language_model_mapping_shards_routed_experts_tp4(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_mapping_config(num_hidden_layers=1, layer_types=("linear_attention",), num_experts=8)),
        encoding="utf-8",
    )
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_linear_attention_layer(tensors, 0)
    add_moe(tensors, 0, num_experts=8)
    write_safetensors(tmp_path / "model.safetensors", tensors)
    manifest = build_manifest(tmp_path)

    dense_mapping = build_language_model_mapping(manifest)
    mappings = [
        build_language_model_mapping(manifest, tensor_parallel=TensorParallel(world_size=4, rank=rank))
        for rank in range(4)
    ]

    assert [m.experts_per_layer for m in mappings] == [2, 2, 2, 2]
    assert [m.layers[0].mlp.expert_start for m in mappings] == [0, 2, 4, 6]
    assert [m.layers[0].mlp.expert_end for m in mappings] == [2, 4, 6, 8]
    assert [[expert.index for expert in m.layers[0].mlp.experts] for m in mappings] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert all(m.unmapped_language_tensor_names == () for m in mappings)
    assert len(mappings[0].mapped_tensor_names) < len(dense_mapping.mapped_tensor_names)


def test_build_language_model_mapping_uses_true_tp_rules(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_mapping_config(num_experts=8)), encoding="utf-8")
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.language_model.embed_tokens.weight": ("BF16", (320, 256)),
        "model.language_model.norm.weight": ("BF16", (256,)),
        "lm_head.weight": ("BF16", (320, 256)),
    }
    add_linear_attention_layer(tensors, 0)
    add_full_attention_layer(tensors, 1)
    add_moe(tensors, 0, num_experts=8)
    add_moe(tensors, 1, num_experts=8)
    write_safetensors(tmp_path / "model.safetensors", tensors)

    mapping = build_language_model_mapping(
        build_manifest(tmp_path),
        tensor_parallel=TensorParallel(world_size=4, rank=1),
    )
    linear = mapping.layers[0].attention
    full = mapping.layers[1].attention

    assert mapping.embed_tokens.shard.rule == "parallel_embedding"
    assert mapping.embed_tokens.shape == (80, 256)
    assert mapping.lm_head.shard.rule == "parallel_head"
    assert mapping.lm_head.shape == (80, 256)
    assert linear.in_proj_qkv.weight.shard.rule == "packed_qkv_column_parallel"
    assert linear.in_proj_qkv.weight.shape == (64, 256)
    assert linear.in_proj_qkv.scale is not None
    assert linear.in_proj_qkv.scale.shape == (3, 2)
    assert [segment.size for segment in linear.in_proj_qkv.scale.shard.segments] == [1, 1, 1]
    assert linear.conv1d_weight.shard.rule == "packed_conv1d_channel_parallel"
    assert linear.conv1d_weight.shape == (64, 1, 4)
    assert linear.out_proj.weight.shard.rule == "row_parallel"
    assert linear.out_proj.weight.shape == (256, 32)
    assert linear.out_proj.scale is not None
    assert linear.out_proj.scale.shape == (2, 1)
    assert full.q_proj.weight.shard.rule == "column_parallel"
    assert full.q_proj.weight.shape == (64, 256)
    assert full.k_proj.weight.shard.rule == "replicated"
    assert full.k_proj.weight.shape == (64, 256)
    assert full.o_proj.weight.shard.rule == "row_parallel"
    assert full.o_proj.weight.shape == (256, 32)
    assert [expert.index for expert in mapping.layers[0].mlp.experts] == [2, 3]
    assert mapping.layers[0].mlp.shared_expert.gate_proj.weight.shard.rule == "replicated"


def _mapping_config(
    *,
    num_hidden_layers: int = 2,
    layer_types: tuple[str, ...] = ("linear_attention", "full_attention"),
    num_experts: int = 2,
) -> dict[str, object]:
    return {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "vocab_size": 320,
            "num_hidden_layers": num_hidden_layers,
            "layer_types": list(layer_types),
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "attn_output_gate": True,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "num_experts": num_experts,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000},
        }
    }


def add_linear_attention_layer(tensors: dict[str, tuple[str, tuple[int, ...]]], layer: int) -> None:
    p = f"model.language_model.layers.{layer}."
    tensors[p + "input_layernorm.weight"] = ("BF16", (256,))
    tensors[p + "post_attention_layernorm.weight"] = ("BF16", (256,))
    tensors[p + "linear_attn.A_log"] = ("BF16", (4,))
    tensors[p + "linear_attn.dt_bias"] = ("BF16", (4,))
    tensors[p + "linear_attn.norm.weight"] = ("BF16", (32,))
    tensors[p + "linear_attn.conv1d.weight"] = ("BF16", (256, 1, 4))
    add_fp8(tensors, p + "linear_attn.in_proj_qkv.weight", (256, 256))
    add_fp8(tensors, p + "linear_attn.in_proj_z.weight", (128, 256))
    add_fp8(tensors, p + "linear_attn.out_proj.weight", (256, 128))
    tensors[p + "linear_attn.in_proj_a.weight"] = ("BF16", (4, 256))
    tensors[p + "linear_attn.in_proj_b.weight"] = ("BF16", (4, 256))


def add_full_attention_layer(tensors: dict[str, tuple[str, tuple[int, ...]]], layer: int) -> None:
    p = f"model.language_model.layers.{layer}."
    tensors[p + "input_layernorm.weight"] = ("BF16", (256,))
    tensors[p + "post_attention_layernorm.weight"] = ("BF16", (256,))
    add_fp8(tensors, p + "self_attn.q_proj.weight", (256, 256))
    add_fp8(tensors, p + "self_attn.k_proj.weight", (64, 256))
    add_fp8(tensors, p + "self_attn.v_proj.weight", (64, 256))
    add_fp8(tensors, p + "self_attn.o_proj.weight", (256, 128))
    tensors[p + "self_attn.q_norm.weight"] = ("BF16", (64,))
    tensors[p + "self_attn.k_norm.weight"] = ("BF16", (64,))


def add_moe(tensors: dict[str, tuple[str, tuple[int, ...]]], layer: int, *, num_experts: int = 2) -> None:
    p = f"model.language_model.layers.{layer}.mlp."
    tensors[p + "gate.weight"] = ("BF16", (num_experts, 256))
    tensors[p + "shared_expert_gate.weight"] = ("BF16", (1, 256))
    for expert in range(num_experts):
        add_expert(tensors, f"{p}experts.{expert}.")
    add_expert(tensors, p + "shared_expert.")


def add_expert(tensors: dict[str, tuple[str, tuple[int, ...]]], prefix: str) -> None:
    add_fp8(tensors, prefix + "gate_proj.weight", (128, 256))
    add_fp8(tensors, prefix + "up_proj.weight", (128, 256))
    add_fp8(tensors, prefix + "down_proj.weight", (256, 128))


def add_fp8(tensors: dict[str, tuple[str, tuple[int, ...]]], name: str, shape: tuple[int, int]) -> None:
    tensors[name] = ("F8_E4M3", shape)
    tensors[name + "_scale_inv"] = ("BF16", (_ceil_div(shape[0], 128), _ceil_div(shape[1], 128)))


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _dtype_size(dtype: str) -> int:
    return 1 if dtype == "F8_E4M3" else 2


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
