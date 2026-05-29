from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint import build_manifest
from loader import LoaderError, TensorLoader


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
