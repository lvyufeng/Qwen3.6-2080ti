from __future__ import annotations

import ctypes
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXT_DIR = _REPO_ROOT / "build" / "extensions"
_NATIVE_MOD: Any | None = None
_PRELOADED_TORCH_LIBS = False


def _find_built_extension(module_name: str) -> Path | None:
    suffixes = sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True)
    candidates = [f"{module_name}.so", *(f"{module_name}{suffix}" for suffix in suffixes)]
    for name in candidates:
        path = _EXT_DIR / name
        if path.exists():
            return path
    matches = sorted(_EXT_DIR.glob(f"{module_name}*.so"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _preload_torch_libs() -> None:
    global _PRELOADED_TORCH_LIBS
    if _PRELOADED_TORCH_LIBS:
        return
    import torch

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for name in ("libc10.so", "libc10_cuda.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so", "libtorch.so"):
        path = torch_lib_dir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_TORCH_LIBS = True


def load_fp8_extension() -> Any:
    global _NATIVE_MOD
    if _NATIVE_MOD is not None:
        return _NATIVE_MOD
    ext_path = _find_built_extension("qwen36_fp8")
    if ext_path is None:
        raise RuntimeError("qwen36_fp8 extension is not built; run `python setup.py build_ext --inplace`")
    _preload_torch_libs()
    spec = importlib.util.spec_from_file_location("qwen36_fp8", ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load qwen36_fp8 extension from {ext_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _NATIVE_MOD = module
    return module
