from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from checkpoint import Manifest
from decode_state import DecodeState
from loader import TensorLoader
from reference_ops import LinearDispatchStats
from runtime_config import RuntimeConfig
from tensor_parallel import TensorParallel
from tp_runtime import TpLaunchConfig, TpRuntime, mapped_tensor_bytes, tp_decode_step_local_logits, tp_greedy_next_token
from tp_weights import MappedWeightStats, MappedWeights
from weight_mapping import build_language_model_mapping


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaMemoryStats:
    available: bool
    free_bytes: int | None = None
    total_bytes: int | None = None
    max_allocated: int | None = None
    max_reserved: int | None = None


@dataclass(frozen=True)
class GenerateResult:
    backend: str
    world_size: int
    rank: int
    local_rank: int
    device: Any
    prompt_tokens: int
    max_new_tokens: int
    layers: int
    mapped_tensors: int
    mapped_bytes: int
    load_stats: MappedWeightStats
    load_seconds: float
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    decode_tokens_per_second: float
    total_tokens_per_second: float
    dispatch_stats: LinearDispatchStats
    all_finite: bool
    cuda_memory: CudaMemoryStats
    generated_token_ids: list[int]
    text: str | None


class TpModelRunner:
    def __init__(self, manifest: Manifest, runtime_config: RuntimeConfig, launch: TpLaunchConfig) -> None:
        self.manifest = manifest
        self.runtime_config = runtime_config
        self.launch = launch
        self.tp = TensorParallel(world_size=launch.world_size, rank=launch.rank)
        self.mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=self.tp)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerateResult:
        import time

        import torch
        from transformers import AutoTokenizer

        with TpRuntime(self.launch) as runtime:
            tokenizer = AutoTokenizer.from_pretrained(self.manifest.model_dir, local_files_only=True, trust_remote_code=True)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(runtime.device)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise EngineError("TP generation currently supports exactly one prompt")
            if input_ids.shape[1] == 0:
                raise EngineError("TP generation requires at least one prompt token")
            _sync_device(runtime.device)
            total_start = time.perf_counter()
            with TensorLoader(self.manifest) as loader:
                weights = MappedWeights(loader, self.mapping, device=str(runtime.device))
                load_start = time.perf_counter()
                load_stats = weights.preload()
                _sync_device(runtime.device)
                load_end = time.perf_counter()
                state = DecodeState.empty(self.mapping, self.runtime_config)
                prefill_start = time.perf_counter()
                logits = tp_decode_step_local_logits(input_ids, self.mapping, self.runtime_config, weights, runtime, state)
                _sync_device(runtime.device)
                prefill_end = time.perf_counter()
                generated: list[int] = []
                step_finite: list[bool] = []
                decode_start = time.perf_counter()
                for step in range(max_new_tokens):
                    last_logits = logits[:, -1].float()
                    step_finite.append(bool(torch.isfinite(last_logits).all().item()))
                    next_token = tp_greedy_next_token(logits, self.mapping.lm_head, runtime)
                    next_token = _sync_next_token(next_token, runtime)
                    generated.append(int(next_token.item()))
                    if step + 1 < max_new_tokens:
                        logits = tp_decode_step_local_logits(next_token[:, None], self.mapping, self.runtime_config, weights, runtime, state)
                _sync_device(runtime.device)
                decode_end = time.perf_counter()
                dispatch = weights.dispatch_stats
            total_end = time.perf_counter()
            load_seconds = _elapsed_seconds(load_start, load_end)
            prefill_seconds = _elapsed_seconds(prefill_start, prefill_end)
            decode_seconds = _elapsed_seconds(decode_start, decode_end)
            total_seconds = _elapsed_seconds(total_start, total_end)
            result = GenerateResult(
                backend=self.launch.backend,
                world_size=self.launch.world_size,
                rank=self.launch.rank,
                local_rank=self.launch.local_rank,
                device=runtime.device,
                prompt_tokens=input_ids.numel(),
                max_new_tokens=max_new_tokens,
                layers=len(self.mapping.layers),
                mapped_tensors=len(self.mapping.mapped_tensor_names),
                mapped_bytes=mapped_tensor_bytes(self.mapping),
                load_stats=load_stats,
                load_seconds=load_seconds,
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                total_seconds=total_seconds,
                decode_tokens_per_second=max_new_tokens / decode_seconds if decode_seconds > 0 else float("inf"),
                total_tokens_per_second=max_new_tokens / total_seconds if total_seconds > 0 else float("inf"),
                dispatch_stats=dispatch,
                all_finite=all(step_finite),
                cuda_memory=_cuda_memory_stats(runtime.device),
                generated_token_ids=generated if self.launch.rank == 0 else [],
                text=tokenizer.decode(generated, skip_special_tokens=False) if self.launch.rank == 0 else None,
            )
            runtime.barrier()
            return result


def _sync_next_token(next_token: Any, runtime: TpRuntime) -> Any:
    if runtime.config.is_distributed:
        import torch.distributed as dist

        dist.broadcast(next_token, src=0)
    return next_token


def _sync_device(device: Any) -> None:
    if getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def _elapsed_seconds(start: float, end: float) -> float:
    return max(0.0, end - start)


def _cuda_memory_stats(device: Any) -> CudaMemoryStats:
    if getattr(device, "type", None) != "cuda":
        return CudaMemoryStats(available=False)
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return CudaMemoryStats(
        available=True,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        max_allocated=torch.cuda.max_memory_allocated(device),
        max_reserved=torch.cuda.max_memory_reserved(device),
    )
