from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from checkpoint import build_manifest
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
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "vocab_size": 320,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "attn_output_gate": True,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "num_experts": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
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


def add_moe(tensors: dict[str, tuple[str, tuple[int, ...]]], layer: int) -> None:
    p = f"model.language_model.layers.{layer}.mlp."
    tensors[p + "gate.weight"] = ("BF16", (2, 256))
    tensors[p + "shared_expert_gate.weight"] = ("BF16", (1, 256))
    for expert in range(2):
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
