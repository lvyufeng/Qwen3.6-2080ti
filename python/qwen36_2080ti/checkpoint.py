from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DTYPE_SIZES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"
_CONFIG_FILE = "config.json"

_SCALE_SUFFIXES = ("_scale_inv", "_scale")


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    begin: int
    end: int
    data_start: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def is_fp8(self) -> bool:
        return self.dtype.startswith("F8")

    @property
    def is_scale(self) -> bool:
        return self.name.endswith(_SCALE_SUFFIXES)


@dataclass
class Manifest:
    model_dir: Path
    config: dict[str, Any]
    tensors: dict[str, TensorInfo]
    scale_of: dict[str, str]

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    @property
    def param_count(self) -> int:
        return sum(t.numel for t in self.tensors.values() if not t.is_scale)

    @property
    def fp8_tensor_count(self) -> int:
        return sum(1 for t in self.tensors.values() if t.is_fp8)


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / _CONFIG_FILE
    if not config_path.is_file():
        raise CheckpointError(f"missing {_CONFIG_FILE} under {model_dir}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CheckpointError(f"expected a JSON object in {config_path}")
    return config


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) != 8:
            raise CheckpointError(f"{path} is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", size_bytes)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise CheckpointError(f"{path} header is truncated")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"invalid safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointError(f"safetensors header in {path} is not an object")
    return header, 8 + header_len


def find_shard_files(model_dir: Path) -> list[str]:
    index_path = model_dir / _INDEX_FILE
    if index_path.is_file():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"invalid JSON in {index_path}: {exc}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(f"{index_path} has no usable weight_map")
        return sorted({str(v) for v in weight_map.values()})
    if (model_dir / _SINGLE_FILE).is_file():
        return [_SINGLE_FILE]
    return []


def _tensors_from_shard(model_dir: Path, shard: str) -> list[TensorInfo]:
    path = model_dir / shard
    if not path.is_file():
        raise CheckpointError(f"shard referenced but not found: {path}")
    header, data_offset = read_safetensors_header(path)
    infos: list[TensorInfo] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise CheckpointError(f"bad header entry for {name} in {shard}")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_SIZES:
            raise CheckpointError(f"unknown dtype {dtype!r} for {name} in {shard}")
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            raise CheckpointError(f"bad shape for {name} in {shard}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise CheckpointError(f"bad data_offsets for {name} in {shard}")
        begin, end = offsets
        info = TensorInfo(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            shard=shard,
            begin=begin,
            end=end,
            data_start=data_offset + begin,
        )
        expected = info.numel * _DTYPE_SIZES[dtype]
        if expected != info.nbytes:
            raise CheckpointError(
                f"{name} in {shard}: expected {expected} bytes from shape/dtype "
                f"but data_offsets span {info.nbytes}"
            )
        infos.append(info)
    return infos


def _link_scales(tensors: dict[str, TensorInfo]) -> dict[str, str]:
    scale_of: dict[str, str] = {}
    for name, info in tensors.items():
        if not info.is_fp8:
            continue
        for suffix in _SCALE_SUFFIXES:
            candidate = name + suffix
            if candidate in tensors:
                scale_of[name] = candidate
                break
    return scale_of


def build_manifest(model_dir: Path) -> Manifest:
    config = load_config(model_dir)
    tensors: dict[str, TensorInfo] = {}
    for shard in find_shard_files(model_dir):
        for info in _tensors_from_shard(model_dir, shard):
            if info.name in tensors:
                raise CheckpointError(f"duplicate tensor {info.name} across shards")
            tensors[info.name] = info
    return Manifest(
        model_dir=model_dir,
        config=config,
        tensors=tensors,
        scale_of=_link_scales(tensors),
    )
