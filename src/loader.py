from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from checkpoint import CheckpointError, Manifest, TensorInfo


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

    def payload(self, name: str, *, device: str | None = None) -> TensorPayload:
        info = self.tensor_info(name)
        return TensorPayload(info=info, tensor=self.tensor(name, device=device))

    def slice(self, name: str) -> Any:
        info = self.tensor_info(name)
        return self._file(info.shard).get_slice(name)

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
