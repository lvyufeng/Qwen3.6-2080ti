from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from qwen36_2080ti.checkpoint import build_manifest
from qwen36_2080ti.fp8_smoke import inspect_fp8_checkpoint


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


def write_checkpoint(tmp_path: Path, tensors: dict[str, dict[str, Any]]) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8")
    write_safetensors(tmp_path / "model.safetensors", tensors)


def test_fp8_smoke_accepts_scaled_fp8_tensor(tmp_path: Path) -> None:
    write_checkpoint(
        tmp_path,
        {
            "w": {"dtype": "F8_E4M3", "shape": [2, 4], "data": bytes(range(8))},
            "w_scale_inv": {"dtype": "F32", "shape": [1], "data": b"\x00\x00\x80?"},
        },
    )

    report = inspect_fp8_checkpoint(build_manifest(tmp_path))

    assert report.ok
    assert report.fp8_tensors == 1
    assert report.scale_links == 1
    assert report.missing_scales == ()
    assert report.fp8_bytes == 8
    assert report.scale_bytes == 4


def test_fp8_smoke_rejects_unscaled_fp8_tensor(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, {"w": {"dtype": "F8_E4M3", "shape": [2, 4], "data": bytes(range(8))}})

    report = inspect_fp8_checkpoint(build_manifest(tmp_path))

    assert not report.ok
    assert report.missing_scales == ("w",)


def test_fp8_smoke_rejects_checkpoint_without_fp8(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, {"w": {"dtype": "F16", "shape": [2, 4], "data": bytes(range(16))}})

    report = inspect_fp8_checkpoint(build_manifest(tmp_path))

    assert not report.ok
    assert report.fp8_tensors == 0
