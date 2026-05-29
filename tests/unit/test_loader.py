from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint import build_manifest
from loader import LoaderError, TensorLoader
from weight_mapping import PackedSegment, ShardedTensor, TensorShard


def test_tensor_loader_reads_torch_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x": torch.tensor([1.5, -2.0], dtype=torch.float32)}, tmp_path / "model.safetensors")

    with TensorLoader(build_manifest(tmp_path)) as loader:
        payload = loader.payload("x")
        tensor = loader.tensor("x")

    assert payload.nbytes == 8
    assert tensor.dtype is torch.float32
    assert tensor.tolist() == [1.5, -2.0]


def test_tensor_loader_reads_bf16_as_torch_bfloat16(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}, tmp_path / "model.safetensors")

    with TensorLoader(build_manifest(tmp_path)) as loader:
        tensor = loader.tensor("x")

    assert tensor.dtype is torch.bfloat16
    assert tensor.tolist() == [1.0, 2.0]


def test_tensor_loader_reports_unknown_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({}, tmp_path / "model.safetensors")

    with TensorLoader(build_manifest(tmp_path)) as loader:
        with pytest.raises(LoaderError, match="unknown tensor"):
            loader.payload("missing")

def test_tensor_loader_reads_dim_shards(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x": torch.arange(24, dtype=torch.float32).reshape(4, 6)}, tmp_path / "model.safetensors")
    manifest = build_manifest(tmp_path)

    row_shard = ShardedTensor(manifest.tensors["x"], TensorShard("column_parallel", dim=0, start=1, size=2, local_shape=(2, 6)))
    col_shard = ShardedTensor(manifest.tensors["x"], TensorShard("row_parallel", dim=1, start=2, size=3, local_shape=(4, 3)))
    with TensorLoader(manifest) as loader:
        rows = loader.tensor_shard(row_shard)
        cols = loader.tensor_shard(col_shard)

    assert rows.tolist() == torch.arange(24, dtype=torch.float32).reshape(4, 6)[1:3].tolist()
    assert cols.tolist() == torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, 2:5].tolist()


def test_tensor_loader_reads_packed_segments(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x": torch.arange(10, dtype=torch.float32).reshape(10, 1)}, tmp_path / "model.safetensors")
    manifest = build_manifest(tmp_path)
    shard = ShardedTensor(
        manifest.tensors["x"],
        TensorShard(
            "packed_qkv_column_parallel",
            dim=0,
            local_shape=(4, 1),
            segments=(PackedSegment(1, 2), PackedSegment(6, 2)),
        ),
    )

    with TensorLoader(manifest) as loader:
        out = loader.tensor_shard(shard)

    assert out.flatten().tolist() == [1.0, 2.0, 6.0, 7.0]
