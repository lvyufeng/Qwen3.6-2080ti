from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

from checkpoint import Manifest, build_manifest
from engine import GenerateResult, TpModelSession
from runtime_config import RuntimeConfig, parse_runtime_config
from tp_runtime import TpLaunchConfig, TpRuntime

LOAD = "LOAD"
GENERATE = "GENERATE"
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
class WorkerState:
    launch: TpLaunchConfig
    manifest: Manifest | None = None
    runtime_config: RuntimeConfig | None = None
    session: TpModelSession | None = None
    loaded_model_dir: str | None = None
    should_shutdown: bool = False


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
            return _execute_generate(state, command)
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


def _execute_generate(state: WorkerState, command: WorkerCommand) -> WorkerResult:
    if state.session is None:
        raise WorkerError("worker has not loaded a model")
    prompt = _payload_str(command, "prompt")
    max_new_tokens = _payload_positive_int(command, "max_new_tokens")
    return WorkerResult(GENERATE, state.launch.rank, True, _generate_result_data(state.session.generate(prompt, max_new_tokens)))


def _generate_result_data(result: GenerateResult) -> dict[str, Any]:
    return {
        "world_size": result.world_size,
        "rank": result.rank,
        "device": str(result.device),
        "prompt_tokens": result.prompt_tokens,
        "max_new_tokens": result.max_new_tokens,
        "all_finite": result.all_finite,
        "generated_token_ids": list(result.generated_token_ids),
        "text": result.text,
    }


def _close_worker_state(state: WorkerState) -> None:
    if state.session is not None:
        state.session.close()
    state.session = None
    state.manifest = None
    state.runtime_config = None
    state.loaded_model_dir = None


def _payload_str(command: WorkerCommand, key: str) -> str:
    value = command.payload.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError(f"{command.kind} requires string payload field: {key}")
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
