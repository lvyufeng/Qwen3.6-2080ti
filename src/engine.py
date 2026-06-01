from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from checkpoint import Manifest
from decode_state import DecodeState, KVCacheStats
from loader import TensorLoader
from reference_ops import LinearDispatchStats
from runtime_config import RuntimeConfig
from tensor_parallel import TensorParallel
from tp_runtime import PagedAttentionDispatchStats, RuntimeProfileConfig, RuntimeProfileStats, TpLaunchConfig, TpRuntime, mapped_tensor_bytes, tp_decode_step_batch, tp_decode_step_local_logits, tp_greedy_next_token
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
    paged_attention_stats: PagedAttentionDispatchStats
    all_finite: bool
    cuda_memory: CudaMemoryStats
    profile: RuntimeProfileStats
    kv_cache: KVCacheStats
    generated_token_ids: list[int]
    text: str | None


@dataclass(frozen=True)
class GenerationStep:
    index: int
    token_id: int
    all_finite: bool
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    is_complete: bool


@dataclass
class GenerationRequestState:
    prompt: str
    max_new_tokens: int
    input_ids: Any
    decode_state: DecodeState
    logits: Any
    prompt_tokens: int
    total_start: float
    prefill_seconds: float
    generated_token_ids: list[int] = field(default_factory=list)
    step_all_finite: list[bool] = field(default_factory=list)
    decode_seconds: float = 0.0
    decode_start: float | None = None
    completed: bool = False
    result_built: bool = False


class TpModelSession:
    def __init__(
        self,
        manifest: Manifest,
        runtime_config: RuntimeConfig,
        launch: TpLaunchConfig,
        *,
        profile_config: RuntimeProfileConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.runtime_config = runtime_config
        self.launch = launch
        self.profile_config = profile_config or RuntimeProfileConfig()
        self.tp = TensorParallel(world_size=launch.world_size, rank=launch.rank)
        self.mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=self.tp)
        self.runtime: TpRuntime | None = None
        self.tokenizer: Any = None
        self.loader: TensorLoader | None = None
        self.weights: MappedWeights | None = None
        self.load_stats: MappedWeightStats | None = None
        self.load_seconds: float = 0.0
        self._loaded = False
        self._closed = False

    def load(self) -> TpModelSession:
        import time

        from transformers import AutoTokenizer

        if self._closed:
            raise EngineError("TP model session is closed")
        if self._loaded:
            return self
        try:
            runtime = TpRuntime(self.launch)
            runtime.__enter__()
            runtime.configure_profiling(self.profile_config)
            self.runtime = runtime
            self.tokenizer = AutoTokenizer.from_pretrained(self.manifest.model_dir, local_files_only=True, trust_remote_code=True)
            loader = TensorLoader(self.manifest)
            loader.__enter__()
            self.loader = loader
            weights = MappedWeights(loader, self.mapping, device=str(runtime.device))
            _sync_device(runtime.device)
            load_start = time.perf_counter()
            self.load_stats = weights.preload()
            _sync_device(runtime.device)
            load_end = time.perf_counter()
            self.weights = weights
            self.load_seconds = _elapsed_seconds(load_start, load_end)
            runtime.barrier()
            self._loaded = True
        except Exception:
            self.close()
            raise
        return self

    def generate(self, prompt: str, max_new_tokens: int) -> GenerateResult:
        state = self.start_generation(prompt, max_new_tokens)
        try:
            while not state.completed:
                self.step_generation(state)
            return self.finish_generation(state)
        except Exception:
            state.decode_state.release()
            raise

    def start_generation(self, prompt: str, max_new_tokens: int) -> GenerationRequestState:
        import time

        runtime, weights, _load_stats = self._require_loaded_components()
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(runtime.device)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise EngineError("TP generation currently supports exactly one prompt")
        if input_ids.shape[1] == 0:
            raise EngineError("TP generation requires at least one prompt token")
        weights.dispatch_stats = LinearDispatchStats()
        runtime.reset_profile()
        # Final cache length is prompt_tokens + max_new_tokens - 1; this upper bound lets the
        # full-attention buffers allocate once so the decode loop never reallocates.
        max_seq_len = input_ids.shape[1] + max_new_tokens
        decode_state = DecodeState.empty(self.mapping, self.runtime_config, max_seq_len=max_seq_len)
        _sync_device(runtime.device)
        total_start = time.perf_counter()
        prefill_start = time.perf_counter()
        try:
            logits = tp_decode_step_local_logits(input_ids, self.mapping, self.runtime_config, weights, runtime, decode_state)
        except Exception:
            decode_state.release()
            raise
        _sync_device(runtime.device)
        prefill_end = time.perf_counter()
        return GenerationRequestState(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_ids=input_ids,
            decode_state=decode_state,
            logits=logits,
            prompt_tokens=input_ids.numel(),
            total_start=total_start,
            prefill_seconds=_elapsed_seconds(prefill_start, prefill_end),
            completed=max_new_tokens == 0,
        )

    def step_generation(self, state: GenerationRequestState) -> GenerationStep:
        import time

        import torch

        runtime, weights, _load_stats = self._require_loaded_components()
        if state.completed:
            raise EngineError("TP generation request is already complete")
        if state.decode_start is None:
            state.decode_start = time.perf_counter()
        last_logits = state.logits[:, -1].float()
        step_finite = bool(torch.isfinite(last_logits).all().item())
        next_token = tp_greedy_next_token(state.logits, self.mapping.lm_head, runtime)
        next_token = _sync_next_token(next_token, runtime)
        token_id = int(next_token.item())
        state.generated_token_ids.append(token_id)
        state.step_all_finite.append(step_finite)
        step_index = len(state.generated_token_ids) - 1
        if len(state.generated_token_ids) < state.max_new_tokens:
            state.logits = tp_decode_step_local_logits(
                next_token[:, None], self.mapping, self.runtime_config, weights, runtime, state.decode_state
            )
        else:
            state.completed = True
        _sync_device(runtime.device)
        decode_end = time.perf_counter()
        state.decode_seconds = _elapsed_seconds(state.decode_start, decode_end)
        return GenerationStep(
            index=step_index,
            token_id=token_id,
            all_finite=step_finite,
            prefill_seconds=state.prefill_seconds,
            decode_seconds=state.decode_seconds,
            total_seconds=_elapsed_seconds(state.total_start, decode_end),
            is_complete=state.completed,
        )

    def step_generations_batch(self, states: list[GenerationRequestState]) -> list[GenerationStep]:
        """Advance multiple active requests in a single batched forward pass.

        Each state must have completed prefill (has logits from the last step).
        Returns one GenerationStep per state, same as step_generation would.
        Falls back to sequential step_generation for single-request case.
        """
        import time

        import torch

        if not states:
            return []
        if len(states) == 1:
            return [self.step_generation(states[0])]

        runtime, weights, _load_stats = self._require_loaded_components()

        # Validate all states are steppable
        for state in states:
            if state.completed:
                raise EngineError("TP generation request is already complete")

        # Initialize decode timing for states that haven't started decoding
        now = time.perf_counter()
        for state in states:
            if state.decode_start is None:
                state.decode_start = now

        # Step 1: Extract next token for each state from their current logits
        # Each state.logits is (1, seq, vocab_local) — take last position
        next_tokens = []
        step_finites = []
        for state in states:
            last_logits = state.logits[:, -1].float()
            step_finite = bool(torch.isfinite(last_logits).all().item())
            step_finites.append(step_finite)
            next_token = tp_greedy_next_token(state.logits, self.mapping.lm_head, runtime)
            next_token = _sync_next_token(next_token, runtime)
            next_tokens.append(next_token)

        # Step 2: Stack tokens into (B, 1) and run batched decode
        input_ids = torch.stack(next_tokens, dim=0)  # (B,) -> need (B, 1)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)

        # Determine which states need a forward pass (not completing this step)
        needs_forward = []
        for i, state in enumerate(states):
            token_id = int(next_tokens[i].item())
            state.generated_token_ids.append(token_id)
            state.step_all_finite.append(step_finites[i])
            if len(state.generated_token_ids) >= state.max_new_tokens:
                state.completed = True
                needs_forward.append(False)
            else:
                needs_forward.append(True)

        # Run batched forward pass only for states that need new logits
        forward_indices = [i for i, need in enumerate(needs_forward) if need]
        if forward_indices:
            forward_ids = input_ids[forward_indices]  # (F, 1)
            forward_states = [states[i].decode_state for i in forward_indices]
            batch_logits = tp_decode_step_batch(
                forward_ids, self.mapping, self.runtime_config, weights, runtime, forward_states
            )
            # Split logits back to per-state
            for idx_in_batch, state_idx in enumerate(forward_indices):
                states[state_idx].logits = batch_logits[idx_in_batch: idx_in_batch + 1]
        else:
            # All states completed — still need to advance decode states for completed ones
            # (they don't need new logits but their caches are already consistent)
            pass

        _sync_device(runtime.device)
        decode_end = time.perf_counter()

        # Build GenerationStep results
        results = []
        for i, state in enumerate(states):
            state.decode_seconds = _elapsed_seconds(state.decode_start, decode_end)
            step_index = len(state.generated_token_ids) - 1
            results.append(GenerationStep(
                index=step_index,
                token_id=state.generated_token_ids[-1],
                all_finite=step_finites[i],
                prefill_seconds=state.prefill_seconds,
                decode_seconds=state.decode_seconds,
                total_seconds=_elapsed_seconds(state.total_start, decode_end),
                is_complete=state.completed,
            ))
        return results

    def finish_generation(self, state: GenerationRequestState) -> GenerateResult:
        import time

        runtime, weights, load_stats = self._require_loaded_components()
        if not state.completed:
            raise EngineError("TP generation request is not complete")
        if state.result_built:
            raise EngineError("TP generation result is already finalized")
        _sync_device(runtime.device)
        total_end = time.perf_counter()
        total_seconds = _elapsed_seconds(state.total_start, total_end)
        state.result_built = True
        profile = runtime.profile_stats.snapshot()
        kv_cache = state.decode_state.kv_stats()
        result = GenerateResult(
            backend=self.launch.backend,
            world_size=self.launch.world_size,
            rank=self.launch.rank,
            local_rank=self.launch.local_rank,
            device=runtime.device,
            prompt_tokens=state.prompt_tokens,
            max_new_tokens=state.max_new_tokens,
            layers=len(self.mapping.layers),
            mapped_tensors=len(self.mapping.mapped_tensor_names),
            mapped_bytes=mapped_tensor_bytes(self.mapping),
            load_stats=load_stats,
            load_seconds=self.load_seconds,
            prefill_seconds=state.prefill_seconds,
            decode_seconds=state.decode_seconds,
            total_seconds=total_seconds,
            decode_tokens_per_second=state.max_new_tokens / state.decode_seconds if state.decode_seconds > 0 else float("inf"),
            total_tokens_per_second=state.max_new_tokens / total_seconds if total_seconds > 0 else float("inf"),
            dispatch_stats=weights.dispatch_stats,
            paged_attention_stats=runtime.paged_attention_stats.snapshot(),
            all_finite=all(state.step_all_finite),
            cuda_memory=_cuda_memory_stats(runtime.device),
            profile=profile,
            kv_cache=kv_cache,
            generated_token_ids=state.generated_token_ids if self.launch.rank == 0 else [],
            text=self.tokenizer.decode(state.generated_token_ids, skip_special_tokens=False) if self.launch.rank == 0 else None,
        )
        runtime.barrier()
        state.decode_state.release()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.loader is not None:
            self.loader.close()
            self.loader = None
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.weights = None
        self.tokenizer = None
        self.load_stats = None
        self.load_seconds = 0.0
        self._loaded = False

    def __enter__(self) -> TpModelSession:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_loaded_components(self) -> tuple[TpRuntime, MappedWeights, MappedWeightStats]:
        if self._closed:
            raise EngineError("TP model session is closed")
        if not self._loaded:
            raise EngineError("TP model session is not loaded")
        runtime = self.runtime
        weights = self.weights
        load_stats = self.load_stats
        assert runtime is not None
        assert weights is not None
        assert load_stats is not None
        return runtime, weights, load_stats


class TpModelRunner:
    def __init__(
        self,
        manifest: Manifest,
        runtime_config: RuntimeConfig,
        launch: TpLaunchConfig,
        *,
        profile_config: RuntimeProfileConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.runtime_config = runtime_config
        self.launch = launch
        self.profile_config = profile_config or RuntimeProfileConfig()
        self.tp = TensorParallel(world_size=launch.world_size, rank=launch.rank)
        self.mapping = build_language_model_mapping(manifest, strict=True, tensor_parallel=self.tp)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerateResult:
        with TpModelSession(self.manifest, self.runtime_config, self.launch, profile_config=self.profile_config) as session:
            return session.generate(prompt, max_new_tokens)


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
