from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO, cast

from checkpoint import Manifest, build_manifest
from engine import GenerationRequestState, GenerationStep, GenerateResult, TpModelSession
from runtime_config import RuntimeConfig, parse_runtime_config
from scheduler import RequestScheduler, RequestStatus, ScheduledResult, SchedulerError, SchedulerSnapshot
from tp_runtime import TpLaunchConfig, TpRuntime

LOAD = "LOAD"
GENERATE = "GENERATE"
SUBMIT = "SUBMIT"
POLL = "POLL"
STEP = "STEP"
STATUS = "STATUS"
SHUTDOWN = "SHUTDOWN"


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerCommand:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerResult:
    kind: str
    rank: int
    ok: bool
    data: dict[str, Any]
    error: str | None = None


@dataclass
class ActiveGeneration:
    request_id: str
    state: GenerationRequestState
    last_step: GenerationStep | None = None


@dataclass
class WorkerState:
    launch: TpLaunchConfig
    manifest: Manifest | None = None
    runtime_config: RuntimeConfig | None = None
    session: TpModelSession | None = None
    loaded_model_dir: str | None = None
    should_shutdown: bool = False
    scheduler: RequestScheduler = field(default_factory=RequestScheduler)
    active_generation: ActiveGeneration | None = None


def broadcast_command(command: WorkerCommand | None, runtime: TpRuntime, *, src: int = 0) -> WorkerCommand:
    if not runtime.config.is_distributed:
        if command is None:
            raise WorkerError("single-rank command broadcast requires a command")
        return _validate_command(command)
    import torch.distributed as dist

    if runtime.config.rank == src and command is None:
        raise WorkerError("source rank must provide a worker command")
    objects: list[Any] = [command]
    dist.broadcast_object_list(objects, src=src)
    return _validate_command(objects[0])


def execute_command(state: WorkerState, command: WorkerCommand, runtime: TpRuntime | None = None) -> WorkerResult:
    try:
        command = _validate_command(command)
        if command.kind == LOAD:
            return _execute_load(state, command)
        if command.kind == GENERATE:
            return _execute_generate(state, command, runtime)
        if command.kind == SUBMIT:
            return _execute_submit(state, command, runtime)
        if command.kind == POLL:
            return _execute_poll(state, command, runtime)
        if command.kind == STEP:
            return _execute_step(state, command, runtime)
        if command.kind == STATUS:
            return _execute_status(state)
        if command.kind == SHUTDOWN:
            _close_worker_state(state)
            state.should_shutdown = True
            return WorkerResult(SHUTDOWN, state.launch.rank, True, {"shutdown": True})
        raise WorkerError(f"unknown worker command: {command.kind}")
    except Exception as exc:
        return WorkerResult(command.kind if isinstance(command, WorkerCommand) else "UNKNOWN", state.launch.rank, False, {}, str(exc))


def run_worker_loop(
    state: WorkerState,
    runtime: TpRuntime,
    commands: Iterable[WorkerCommand] | None = None,
) -> list[WorkerResult]:
    iterator = iter(commands) if runtime.config.rank == 0 else None
    results: list[WorkerResult] = []
    while True:
        if iterator is not None:
            try:
                command = next(iterator)
            except StopIteration as exc:
                raise WorkerError("rank0 command loop ended without SHUTDOWN") from exc
        else:
            command = None
        command = broadcast_command(command, runtime)
        result = execute_command(state, command, runtime)
        results.append(result)
        if state.should_shutdown:
            break
    return results


def command_from_dict(obj: Any) -> tuple[str | None, WorkerCommand]:
    if not isinstance(obj, dict):
        raise WorkerError(f"worker protocol command must be an object, got {type(obj).__name__}")
    request_id = obj.get("id")
    if request_id is not None and not isinstance(request_id, str):
        raise WorkerError("worker protocol command id must be a string")
    kind = obj.get("kind")
    if not isinstance(kind, str):
        raise WorkerError("worker protocol command kind must be a string")
    payload = obj.get("payload", {})
    if not isinstance(payload, dict):
        raise WorkerError("worker protocol command payload must be a dict")
    return request_id, WorkerCommand(kind, payload)


def worker_result_to_dict(result: WorkerResult) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "rank": result.rank,
        "ok": result.ok,
        "data": result.data,
        "error": result.error,
    }


def gather_worker_results(result: WorkerResult, runtime: TpRuntime, *, dst: int = 0) -> list[WorkerResult] | None:
    if not runtime.config.is_distributed:
        return [result]
    import torch.distributed as dist

    gathered: list[Any] | None = [None] * runtime.config.world_size if runtime.config.rank == dst else None
    dist.gather_object(result, object_gather_list=gathered, dst=dst)
    if runtime.config.rank != dst:
        return None
    assert gathered is not None
    results: list[WorkerResult] = []
    for item in gathered:
        if not isinstance(item, WorkerResult):
            raise WorkerError(f"expected gathered WorkerResult, got {type(item).__name__}")
        results.append(item)
    return sorted(results, key=lambda item: item.rank)


def protocol_response(request_id: str | None, command: WorkerCommand, rank_results: list[WorkerResult]) -> dict[str, Any]:
    if not rank_results:
        raise WorkerError("worker protocol response requires at least one rank result")
    rank_results = sorted(rank_results, key=lambda item: item.rank)
    ok = all(result.ok for result in rank_results)
    primary = next((result for result in rank_results if result.rank == 0), rank_results[0])
    errors = [f"rank {result.rank}: {result.error}" for result in rank_results if not result.ok]
    return {
        "id": request_id,
        "kind": command.kind,
        "ok": ok,
        "rank": primary.rank,
        "data": primary.data,
        "rank_results": [worker_result_to_dict(result) for result in rank_results],
        "error": None if ok else "; ".join(errors),
        "control": _control_data(len(rank_results)),
    }


def run_worker_protocol_loop(
    state: WorkerState,
    runtime: TpRuntime,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    iterator = iter(input_stream) if runtime.config.rank == 0 else None
    while True:
        request_id: str | None = None
        command: WorkerCommand | None = None
        if iterator is not None:
            request_id, command = _read_protocol_command(iterator, output_stream)
        command = broadcast_command(command, runtime)
        result = execute_command(state, command, runtime)
        rank_results = gather_worker_results(result, runtime)
        if runtime.config.rank == 0:
            assert rank_results is not None
            _write_protocol_response(output_stream, protocol_response(request_id, command, rank_results))
        if state.should_shutdown:
            break


def _read_protocol_command(iterator: Iterable[str], output_stream: TextIO) -> tuple[str | None, WorkerCommand]:
    for line in iterator:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            return command_from_dict(request)
        except json.JSONDecodeError as exc:
            _write_protocol_response(output_stream, _protocol_error_response(None, f"invalid JSON command: {exc.msg}"))
        except WorkerError as exc:
            request_id = request.get("id") if isinstance(request, dict) and isinstance(request.get("id"), str) else None
            _write_protocol_response(output_stream, _protocol_error_response(request_id, str(exc)))
    return None, WorkerCommand(SHUTDOWN, {})


def _protocol_error_response(request_id: str | None, error: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "kind": "UNKNOWN",
        "ok": False,
        "rank": 0,
        "data": {},
        "rank_results": [],
        "error": error,
        "control": _control_data(0),
    }


def _control_data(rank_count: int) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "response_shape": "rank_aggregate",
        "streaming_ready": True,
        "rank_count": rank_count,
    }


def _write_protocol_response(output_stream: TextIO, response: dict[str, Any]) -> None:
    output_stream.write(json.dumps(response, sort_keys=True) + "\n")
    output_stream.flush()


def _execute_load(state: WorkerState, command: WorkerCommand) -> WorkerResult:
    model_dir = _payload_str(command, "model_dir")
    _close_worker_state(state)
    manifest = build_manifest(Path(model_dir))
    runtime_config = parse_runtime_config(manifest.config)
    session = TpModelSession(manifest, runtime_config, state.launch)
    session.load()
    state.manifest = manifest
    state.runtime_config = runtime_config
    state.session = session
    state.loaded_model_dir = model_dir
    return WorkerResult(
        LOAD,
        state.launch.rank,
        True,
        {
            "model_dir": model_dir,
            "world_size": state.launch.world_size,
            "rank": state.launch.rank,
            "layers": len(session.mapping.layers),
            "mapped_tensors": len(session.mapping.mapped_tensor_names),
            "loaded_tensors": session.load_stats.tensor_count if session.load_stats else 0,
            "loaded_bytes": session.load_stats.bytes if session.load_stats else 0,
        },
    )


def _execute_generate(state: WorkerState, command: WorkerCommand, runtime: TpRuntime | None = None) -> WorkerResult:
    session = _require_loaded_session(state, runtime)
    prompt = _payload_str(command, "prompt")
    max_new_tokens = _payload_positive_int(command, "max_new_tokens")
    scheduled = state.scheduler.run_blocking_generate(
        prompt,
        max_new_tokens,
        lambda request: session.generate(request.prompt, request.max_new_tokens),
    )
    assert scheduled.result is not None
    data = _generate_result_data(scheduled.result)
    data["request_id"] = scheduled.request_id
    data["status"] = scheduled.status.value
    data["error"] = scheduled.error
    data["event"] = _event_data(
        request_id=scheduled.request_id,
        type="completed",
        status=scheduled.status.value,
        sequence=scheduled.result.max_new_tokens,
        token_ids=scheduled.result.generated_token_ids,
        is_terminal=True,
    )
    data["scheduler"] = _scheduled_result_data(scheduled)
    return WorkerResult(GENERATE, state.launch.rank, True, data)


def _execute_submit(state: WorkerState, command: WorkerCommand, runtime: TpRuntime | None = None) -> WorkerResult:
    _require_loaded_session(state, runtime)
    prompt = _payload_str(command, "prompt")
    max_new_tokens = _payload_positive_int(command, "max_new_tokens")
    request_id = _payload_optional_str(command, "request_id")
    request = state.scheduler.submit_generate(prompt, max_new_tokens, request_id=request_id)
    snapshot = state.scheduler.snapshot()
    return WorkerResult(
        SUBMIT,
        state.launch.rank,
        True,
        {
            "request_id": request.request_id,
            "status": RequestStatus.PENDING.value,
            "pending": snapshot.pending,
            "event": _event_data(
                request_id=request.request_id,
                type="queued",
                status=RequestStatus.PENDING.value,
                sequence=0,
                token_ids=[],
                is_terminal=False,
            ),
            "scheduler": _scheduler_snapshot_data(snapshot),
        },
    )


def _execute_poll(state: WorkerState, command: WorkerCommand, runtime: TpRuntime | None = None) -> WorkerResult:
    session = _require_loaded_session(state, runtime)
    request_id = _payload_str(command, "request_id")
    active = state.active_generation
    if active is not None and active.request_id == request_id:
        data = _step_running_data(state, active, event_type="running")
        return WorkerResult(POLL, state.launch.rank, True, data)
    scheduled = state.scheduler.result_for(request_id)
    if scheduled is None and active is None and state.scheduler.is_pending(request_id):
        state.scheduler.run_next(lambda request: session.generate(request.prompt, request.max_new_tokens), reraise=False)
        scheduled = state.scheduler.result_for(request_id)
    if scheduled is not None:
        data = _poll_terminal_data(request_id, scheduled)
    elif state.scheduler.is_pending(request_id):
        data = _poll_pending_data(request_id)
    else:
        data = _poll_unknown_data(request_id)
    data["scheduler"] = _scheduler_snapshot_data(state.scheduler.snapshot())
    return WorkerResult(POLL, state.launch.rank, True, data)


def _execute_step(state: WorkerState, command: WorkerCommand, runtime: TpRuntime | None = None) -> WorkerResult:
    session = _require_loaded_session(state, runtime)
    snapshot = state.scheduler.snapshot()
    active = state.active_generation
    if active is None:
        try:
            request = state.scheduler.begin_next()
        except SchedulerError as exc:
            if "no pending request" not in str(exc):
                raise
            return WorkerResult(
                STEP,
                state.launch.rank,
                True,
                {
                    "request_id": None,
                    "found": False,
                    "status": None,
                    "result": None,
                    "error": None,
                    "progress": None,
                    "timings": None,
                    "event": _event_data(
                        request_id=None,
                        type="idle",
                        status=None,
                        sequence=None,
                        token_ids=[],
                        is_terminal=False,
                    ),
                    "scheduler": _scheduler_snapshot_data(snapshot),
                },
            )
        try:
            generation_state = session.start_generation(request.prompt, request.max_new_tokens)
        except Exception as exc:
            state.scheduler.fail_running(exc)
            state.active_generation = None
            return _step_failed_response(
                state,
                request.request_id,
                exc,
                max_new_tokens=request.max_new_tokens,
                generated_tokens=0,
            )
        active = ActiveGeneration(request.request_id, generation_state)
        state.active_generation = active
    try:
        step = session.step_generation(active.state)
        active.last_step = step
        if step.is_complete:
            result = session.finish_generation(active.state)
            scheduled = state.scheduler.complete_running(result)
            data = _step_completed_data(state, active, result, scheduled)
            state.active_generation = None
            return WorkerResult(STEP, state.launch.rank, True, data)
        data = _step_running_data(state, active, event_type="token")
        return WorkerResult(STEP, state.launch.rank, True, data)
    except Exception as exc:
        max_new_tokens = active.state.max_new_tokens if active is not None else None
        generated_tokens = len(active.state.generated_token_ids) if active is not None else 0
        if state.scheduler.running_request() is not None:
            state.scheduler.fail_running(exc)
        state.active_generation = None
        return _step_failed_response(
            state,
            active.request_id if active is not None else "unknown",
            exc,
            max_new_tokens=max_new_tokens,
            generated_tokens=generated_tokens,
        )


def _execute_status(state: WorkerState) -> WorkerResult:
    active = state.active_generation
    active_data = None
    if active is not None:
        latest = active.last_step
        active_data = {
            "request_id": active.request_id,
            "generated_tokens": len(active.state.generated_token_ids),
            "max_new_tokens": active.state.max_new_tokens,
            "latest_token_id": latest.token_id if latest is not None else None,
            "latest_token_index": latest.index if latest is not None else None,
        }
    return WorkerResult(
        STATUS,
        state.launch.rank,
        True,
        {
            "loaded": state.session is not None,
            "loaded_model_dir": state.loaded_model_dir,
            "should_shutdown": state.should_shutdown,
            "active_generation": active_data,
            "scheduler": _scheduler_snapshot_data(state.scheduler.snapshot()),
        },
    )


def _generate_result_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "world_size": result.world_size,
        "rank": result.rank,
        "device": str(result.device),
        "prompt_tokens": result.prompt_tokens,
        "max_new_tokens": result.max_new_tokens,
        "load_seconds": result.load_seconds,
        "prefill_seconds": result.prefill_seconds,
        "decode_seconds": result.decode_seconds,
        "total_seconds": result.total_seconds,
        "decode_tokens_per_second": result.decode_tokens_per_second,
        "total_tokens_per_second": result.total_tokens_per_second,
        "all_finite": result.all_finite,
        "generated_token_ids": list(result.generated_token_ids),
        "text": result.text,
        "runtime": _runtime_data(result),
        "model": _model_data(result),
        "load": _load_data(result),
        "timings": _timings_data(result),
        "throughput": _throughput_data(result),
        "dispatch": _dispatch_stats_data(result),
        "memory": _cuda_memory_data(result),
    }


def _runtime_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "backend": result.backend,
        "world_size": result.world_size,
        "rank": result.rank,
        "local_rank": result.local_rank,
        "device": str(result.device),
    }


def _model_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "layers": result.layers,
        "mapped_tensors": result.mapped_tensors,
        "mapped_bytes": result.mapped_bytes,
    }


def _load_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "loaded_tensors": result.load_stats.tensor_count,
        "loaded_bytes": result.load_stats.bytes,
        "load_seconds": result.load_seconds,
    }


def _timings_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "load_seconds": result.load_seconds,
        "prefill_seconds": result.prefill_seconds,
        "decode_seconds": result.decode_seconds,
        "total_seconds": result.total_seconds,
    }


def _throughput_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "decode_tokens_per_second": result.decode_tokens_per_second,
        "total_tokens_per_second": result.total_tokens_per_second,
    }


def _dispatch_stats_data(result: GenerateResult) -> dict[str, Any]:
    dispatch = result.dispatch_stats
    return {
        "calls": dispatch.calls,
        "fp8_weight_calls": dispatch.fp8_weight_calls,
        "eligible_cuda_calls": dispatch.eligible_cuda_calls,
        "cuda_kernel_hits": dispatch.cuda_kernel_hits,
        "fallback_calls": dispatch.fallback_calls,
        "fallback_disabled_cuda_kernel": dispatch.fallback_disabled_cuda_kernel,
        "fallback_missing_scale": dispatch.fallback_missing_scale,
        "fallback_hidden_not_cuda": dispatch.fallback_hidden_not_cuda,
        "fallback_weight_not_cuda": dispatch.fallback_weight_not_cuda,
        "fallback_scale_not_cuda": dispatch.fallback_scale_not_cuda,
        "fallback_weight_dtype": dispatch.fallback_weight_dtype,
        "fallback_scale_dtype": dispatch.fallback_scale_dtype,
        "fallback_hidden_alignment": dispatch.fallback_hidden_alignment,
        "fallback_weight_alignment": dispatch.fallback_weight_alignment,
    }


def _cuda_memory_data(result: GenerateResult) -> dict[str, Any]:
    memory = result.cuda_memory
    return {
        "available": memory.available,
        "free_bytes": memory.free_bytes,
        "total_bytes": memory.total_bytes,
        "max_allocated": memory.max_allocated,
        "max_reserved": memory.max_reserved,
    }


def _scheduled_result_data(result: ScheduledResult[GenerateResult]) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "status": result.status.value,
        "queued_seconds": result.queued_seconds,
        "run_seconds": result.run_seconds,
    }


def _step_running_data(state: WorkerState, active: ActiveGeneration, *, event_type: str) -> dict[str, Any]:
    step = active.last_step
    assert step is not None
    return {
        "request_id": active.request_id,
        "found": True,
        "status": RequestStatus.RUNNING.value,
        "result": None,
        "error": None,
        "progress": _step_progress_data(active.state, step, is_complete=False),
        "timings": {
            "prefill_seconds": active.state.prefill_seconds,
            "decode_seconds": active.state.decode_seconds,
            "total_seconds": step.total_seconds,
        },
        "event": _event_data(
            request_id=active.request_id,
            type=event_type,
            status=RequestStatus.RUNNING.value,
            sequence=len(active.state.generated_token_ids),
            token_ids=[step.token_id] if event_type == "token" else [],
            is_terminal=False,
        ),
        "scheduler": _scheduler_snapshot_data(state.scheduler.snapshot()),
    }


def _step_completed_data(
    state: WorkerState,
    active: ActiveGeneration,
    result: GenerateResult,
    scheduled: ScheduledResult[GenerateResult],
) -> dict[str, Any]:
    step = active.last_step
    assert step is not None
    return {
        "request_id": scheduled.request_id,
        "found": True,
        "status": RequestStatus.COMPLETED.value,
        "result": _generate_result_data(result),
        "error": None,
        "queued_seconds": scheduled.queued_seconds,
        "run_seconds": scheduled.run_seconds,
        "progress": _step_progress_data(active.state, step, is_complete=True),
        "timings": {
            "prefill_seconds": result.prefill_seconds,
            "decode_seconds": result.decode_seconds,
            "total_seconds": result.total_seconds,
        },
        "event": _event_data(
            request_id=scheduled.request_id,
            type="completed",
            status=RequestStatus.COMPLETED.value,
            sequence=scheduled.result.max_new_tokens,
            token_ids=[step.token_id],
            is_terminal=True,
        ),
        "scheduler": _scheduler_snapshot_data(state.scheduler.snapshot()),
    }


def _step_progress_data(state: GenerationRequestState, step: GenerationStep, *, is_complete: bool) -> dict[str, Any]:
    return {
        "generated_tokens": len(state.generated_token_ids),
        "max_new_tokens": state.max_new_tokens,
        "latest_token_id": step.token_id,
        "latest_token_index": step.index,
        "is_complete": is_complete,
    }


def _step_failed_response(
    state: WorkerState,
    request_id: str,
    exc: Exception | BaseException | str,
    *,
    max_new_tokens: int | None,
    generated_tokens: int = 0,
) -> WorkerResult:
    snapshot = state.scheduler.snapshot()
    return WorkerResult(
        STEP,
        state.launch.rank,
        True,
        {
            "request_id": request_id,
            "found": True,
            "status": RequestStatus.FAILED.value,
            "result": None,
            "error": str(exc),
            "progress": {
                "generated_tokens": generated_tokens,
                "max_new_tokens": max_new_tokens,
                "latest_token_id": None,
                "latest_token_index": None,
                "is_complete": False,
            },
            "timings": None,
            "event": _event_data(
                request_id=request_id,
                type="failed",
                status=RequestStatus.FAILED.value,
                sequence=generated_tokens,
                token_ids=[],
                is_terminal=True,
            ),
            "scheduler": _scheduler_snapshot_data(snapshot),
        },
    )


def _poll_terminal_data(request_id: str, scheduled: ScheduledResult[object]) -> dict[str, Any]:
    result_data = None
    event_type = "failed"
    sequence = 0
    if scheduled.status is RequestStatus.COMPLETED:
        assert scheduled.result is not None
        result = cast(GenerateResult, scheduled.result)
        result_data = _generate_result_data(result)
        sequence = len(result.generated_token_ids)
        event_type = "completed"
    return {
        "request_id": request_id,
        "found": True,
        "status": scheduled.status.value,
        "result": result_data,
        "error": scheduled.error,
        "queued_seconds": scheduled.queued_seconds,
        "run_seconds": scheduled.run_seconds,
        "event": _event_data(
            request_id=request_id,
            type=event_type,
            status=scheduled.status.value,
            sequence=sequence,
            token_ids=[],
            is_terminal=True,
        ),
    }


def _poll_pending_data(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "found": True,
        "status": RequestStatus.PENDING.value,
        "result": None,
        "error": None,
        "queued_seconds": None,
        "run_seconds": None,
        "event": _event_data(
            request_id=request_id,
            type="pending",
            status=RequestStatus.PENDING.value,
            sequence=0,
            token_ids=[],
            is_terminal=False,
        ),
    }


def _poll_unknown_data(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "found": False,
        "status": None,
        "result": None,
        "error": None,
        "queued_seconds": None,
        "run_seconds": None,
        "event": _event_data(
            request_id=request_id,
            type="not_found",
            status=None,
            sequence=None,
            token_ids=[],
            is_terminal=True,
        ),
    }


def _scheduler_snapshot_data(snapshot: SchedulerSnapshot) -> dict[str, Any]:
    return {
        "pending": snapshot.pending,
        "running": snapshot.running,
        "completed": snapshot.completed,
        "failed": snapshot.failed,
        "total_submitted": snapshot.total_submitted,
        "total_completed": snapshot.total_completed,
        "total_failed": snapshot.total_failed,
    }


def _event_data(
    *,
    request_id: str | None,
    type: str,
    status: str | None,
    sequence: int | None,
    token_ids: list[int],
    is_terminal: bool,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "type": type,
        "status": status,
        "sequence": sequence,
        "token_ids": list(token_ids),
        "is_terminal": is_terminal,
    }


def _close_worker_state(state: WorkerState) -> None:
    if state.session is not None:
        state.session.close()
    state.session = None
    state.manifest = None
    state.runtime_config = None
    state.loaded_model_dir = None
    state.active_generation = None
    state.scheduler.clear()


def _require_loaded_session(state: WorkerState, runtime: TpRuntime | None = None) -> TpModelSession:
    session = state.session
    if runtime is not None and runtime.config.is_distributed:
        import torch.distributed as dist

        loaded_by_rank: list[Any] = [None] * runtime.config.world_size
        dist.all_gather_object(loaded_by_rank, session is not None)
        if not all(loaded_by_rank):
            raise WorkerError("worker has not loaded a model")
    if session is None:
        raise WorkerError("worker has not loaded a model")
    return session


def _payload_str(command: WorkerCommand, key: str) -> str:
    value = command.payload.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError(f"{command.kind} requires string payload field: {key}")
    return value


def _payload_optional_str(command: WorkerCommand, key: str) -> str | None:
    value = command.payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise WorkerError(f"{command.kind} requires optional string payload field: {key}")
    return value


def _payload_positive_int(command: WorkerCommand, key: str) -> int:
    value = command.payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise WorkerError(f"{command.kind} requires positive integer payload field: {key}")
    return value


def _validate_command(command: Any) -> WorkerCommand:
    if not isinstance(command, WorkerCommand):
        raise WorkerError(f"expected WorkerCommand, got {type(command).__name__}")
    if not isinstance(command.kind, str):
        raise WorkerError("worker command kind must be a string")
    if not isinstance(command.payload, dict):
        raise WorkerError("worker command payload must be a dict")
    return command
