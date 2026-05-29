from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from qwen36_2080ti.checkpoint import CheckpointError, build_manifest, read_safetensors_header


def write_safetensors(path: Path, tensors: dict[str, dict[str, Any]]) -> None:
    header: dict[str, Any] = {}
    offset = 0
    payloads: list[bytes] = []
    for name, meta in tensors.items():
        data = meta["data"]
        header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + len(data)],
        }
        payloads.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payloads))


def test_build_manifest_from_indexed_shard(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "hidden_size": 4}),
        encoding="utf-8",
    )
    write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.layers.0.mlp.experts.0.up_proj.weight": {
                "dtype": "F8_E4M3",
                "shape": [2, 4],
                "data": bytes(range(8)),
            },
            "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv": {
                "dtype": "F32",
                "shape": [1],
                "data": b"\x00\x00\x80?",
            },
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "model.layers.0.mlp.experts.0.up_proj.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(tmp_path)

    weight = manifest.tensors["model.layers.0.mlp.experts.0.up_proj.weight"]
    assert manifest.config["model_type"] == "qwen3_moe"
    assert len(manifest.tensors) == 2
    assert manifest.fp8_tensor_count == 1
    assert manifest.scale_of[weight.name] == "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv"
    assert weight.nbytes == 8
    assert weight.data_start > 8


def test_read_safetensors_header_rejects_bad_shape_size(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, {"bad": {"dtype": "F32", "shape": [2], "data": b"\x00\x00\x00\x00"}})
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointError, match="expected 8 bytes"):
        build_manifest(tmp_path)


def test_read_safetensors_header_returns_data_offset(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, {"x": {"dtype": "F16", "shape": [2], "data": b"abcd"}})

    header, data_offset = read_safetensors_header(path)

    assert set(header) == {"x"}
    assert data_offset == 8 + len(json.dumps(header).encode("utf-8"))
