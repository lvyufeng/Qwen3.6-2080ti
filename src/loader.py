from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from checkpoint import CheckpointError, Manifest, TensorInfo
from weight_mapping import ShardedTensor, TensorShard


class LoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorPayload:
    info: TensorInfo
    tensor: Any

    @property
    def nbytes(self) -> int:
        return self.info.nbytes


class TensorLoader:
    def __init__(self, manifest: Manifest, *, device: str = "cpu") -> None:
        self.manifest = manifest
        self.device = device
        self._files: dict[str, Any] = {}

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self.manifest.tensors[name]
        except KeyError as exc:
            raise LoaderError(f"unknown tensor: {name}") from exc

    def tensor(self, name: str, *, device: str | None = None) -> Any:
        info = self.tensor_info(name)
        tensor = self._file(info.shard).get_tensor(name)
        target_device = self.device if device is None else device
        return tensor if target_device == "cpu" else tensor.to(target_device, non_blocking=True)

    def tensor_shard(self, tensor: ShardedTensor, *, device: str | None = None) -> Any:
        data = self._read_shard(tensor)
        target_device = self.device if device is None else device
        return data if target_device == "cpu" else data.to(target_device, non_blocking=True)

    def payload(self, name: str, *, device: str | None = None) -> TensorPayload:
        info = self.tensor_info(name)
        return TensorPayload(info=info, tensor=self.tensor(name, device=device))

    def slice(self, name: str) -> Any:
        info = self.tensor_info(name)
        return self._file(info.shard).get_slice(name)

    def _read_shard(self, tensor: ShardedTensor) -> Any:
        shard = tensor.shard
        if shard.rule in {"replicated", "expert_owned"}:
            return self.tensor(tensor.name, device="cpu")
        sliced = self.slice(tensor.name)
        if shard.rule in {"column_parallel", "parallel_embedding", "parallel_head", "vector_column_parallel"}:
            return _slice_dim(sliced, shard, 0)
        if shard.rule == "row_parallel":
            return _slice_dim(sliced, shard, 1)
        if shard.rule in {"packed_qkv_column_parallel", "packed_conv1d_channel_parallel"}:
            return torch.cat([_slice_dim(sliced, TensorShard(rule=shard.rule, dim=0, start=s.start, size=s.size), 0) for s in shard.segments], dim=0).contiguous()
        raise LoaderError(f"unsupported tensor shard rule for {tensor.name}: {shard.rule}")

    def close(self) -> None:
        for handle in self._files.values():
            handle.__exit__(None, None, None)
        self._files.clear()

    def __enter__(self) -> TensorLoader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _file(self, shard: str) -> Any:
        handle = self._files.get(shard)
        if handle is None:
            from safetensors import safe_open

            path = self.manifest.model_dir / shard
            if not path.is_file():
                raise CheckpointError(f"missing shard file: {path}")
            handle = safe_open(path, framework="pt", device="cpu")
            handle.__enter__()
            self._files[shard] = handle
        return handle


def _slice_dim(sliced: Any, shard: TensorShard, dim: int) -> Any:
    if shard.start is None or shard.size is None:
        raise LoaderError("shard start/size are required for sliced tensor loads")
    start = shard.start
    end = start + shard.size
    shape = tuple(sliced.get_shape())
    if len(shape) == 1:
        if dim != 0:
            raise LoaderError(f"cannot slice rank-1 tensor on dim {dim}")
        return sliced[start:end].contiguous()
    if len(shape) == 2:
        if dim == 0:
            return sliced[start:end, :].contiguous()
        if dim == 1:
            return sliced[:, start:end].contiguous()
    if len(shape) == 3 and dim == 0:
        return sliced[start:end, :, :].contiguous()
    raise LoaderError(f"unsupported sliced tensor rank/dim: shape={shape} dim={dim}")
