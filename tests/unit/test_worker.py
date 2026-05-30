from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import pytest
import torch

import engine
from test_engine import _patch_tokenizer, _write_tiny_model
from tp_runtime import TpLaunchConfig, TpRuntime
from worker import (
    GENERATE,
    LOAD,
    SHUTDOWN,
    WorkerCommand,
    WorkerError,
    WorkerResult,
    WorkerState,
    broadcast_command,
    command_from_dict,
    execute_command,
    gather_worker_results,
    protocol_response,
    run_worker_loop,
    run_worker_protocol_loop,
    worker_result_to_dict,
)


def test_worker_command_and_result_are_pickle_safe() -> None:
    command = WorkerCommand(GENERATE, {"prompt": "hello", "max_new_tokens": 2})
    result = WorkerResult(GENERATE, 0, True, {"generated_token_ids": [1, 2], "text": "hi"})

    assert pickle.loads(pickle.dumps(command)) == command
    assert pickle.loads(pickle.dumps(result)) == result


def test_worker_single_rank_broadcast_and_shutdown() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    with TpRuntime(launch) as runtime:
        command = broadcast_command(WorkerCommand(SHUTDOWN, {}), runtime)
        result = execute_command(state, command)

    assert result.ok is True
    assert result.kind == SHUTDOWN
    assert result.data["shutdown"] is True
    assert state.should_shutdown is True


def test_worker_command_from_dict_accepts_protocol_envelope() -> None:
    request_id, command = command_from_dict(
        {"id": "gen-1", "kind": GENERATE, "payload": {"prompt": "hello", "max_new_tokens": 2}}
    )

    assert request_id == "gen-1"
    assert command == WorkerCommand(GENERATE, {"prompt": "hello", "max_new_tokens": 2})


def test_worker_command_from_dict_defaults_payload() -> None:
    request_id, command = command_from_dict({"kind": SHUTDOWN})

    assert request_id is None
    assert command == WorkerCommand(SHUTDOWN, {})


@pytest.mark.parametrize(
    "obj, message",
    [
        ([], "must be an object"),
        ({"id": 1, "kind": SHUTDOWN}, "id must be a string"),
        ({"payload": {}}, "kind must be a string"),
        ({"kind": 1}, "kind must be a string"),
        ({"kind": SHUTDOWN, "payload": []}, "payload must be a dict"),
    ],
)
def test_worker_command_from_dict_rejects_invalid_envelope(obj, message: str) -> None:
    with pytest.raises(WorkerError, match=message):
        command_from_dict(obj)


def test_worker_result_to_dict_and_protocol_response() -> None:
    ok = WorkerResult(GENERATE, 0, True, {"text": "hi"})
    failed = WorkerResult(GENERATE, 1, False, {}, "boom")

    assert worker_result_to_dict(ok) == {"kind": GENERATE, "rank": 0, "ok": True, "data": {"text": "hi"}, "error": None}
    response = protocol_response("gen-1", WorkerCommand(GENERATE, {}), [ok, failed])

    assert response["id"] == "gen-1"
    assert response["kind"] == GENERATE
    assert response["ok"] is False
    assert response["rank"] == 0
    assert response["data"] == {"text": "hi"}
    assert response["rank_results"] == [worker_result_to_dict(ok), worker_result_to_dict(failed)]
    assert "rank 1: boom" in response["error"]


def test_gather_worker_results_single_rank() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    result = WorkerResult(GENERATE, 0, True, {"rank": 0})
    with TpRuntime(launch) as runtime:
        gathered = gather_worker_results(result, runtime)

    assert gathered == [result]


def test_worker_load_initializes_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)

    result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))

    assert result.ok is True
    assert result.kind == LOAD
    assert result.data["model_dir"] == str(tmp_path)
    assert result.data["world_size"] == 1
    assert result.data["rank"] == 0
    assert result.data["layers"] == 1
    assert result.data["mapped_tensors"] > 0
    assert result.data["loaded_tensors"] > 0
    assert result.data["loaded_bytes"] > 0
    assert state.manifest is not None
    assert state.runtime_config is not None
    assert state.session is not None
    assert state.loaded_model_dir == str(tmp_path)


def test_worker_generate_requires_loaded_model() -> None:
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    result = execute_command(state, WorkerCommand(GENERATE, {"prompt": "hello", "max_new_tokens": 1}))

    assert result.ok is False
    assert "not loaded" in (result.error or "")


def test_worker_load_then_generate_single_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)

    load_result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))
    generate_result = execute_command(state, WorkerCommand(GENERATE, {"prompt": "hello", "max_new_tokens": 2}))

    assert load_result.ok is True
    assert generate_result.ok is True
    assert generate_result.data["world_size"] == 1
    assert generate_result.data["rank"] == 0
    assert generate_result.data["device"] == "cpu"
    assert generate_result.data["prompt_tokens"] == 2
    assert generate_result.data["max_new_tokens"] == 2
    assert generate_result.data["all_finite"] is True
    assert len(generate_result.data["generated_token_ids"]) == 2
    assert generate_result.data["text"].startswith("decoded:")


def test_worker_protocol_loop_single_rank_load_generate_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    preload_calls = 0
    original_preload = engine.MappedWeights.preload

    def counted_preload(self):
        nonlocal preload_calls
        preload_calls += 1
        return original_preload(self)

    monkeypatch.setattr(engine.MappedWeights, "preload", counted_preload)
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    input_stream = io.StringIO(
        "\n".join(
            [
                json.dumps({"id": "load", "kind": LOAD, "payload": {"model_dir": str(tmp_path)}}),
                json.dumps({"id": "gen-1", "kind": GENERATE, "payload": {"prompt": "hello", "max_new_tokens": 2}}),
                json.dumps({"id": "gen-2", "kind": GENERATE, "payload": {"prompt": "again", "max_new_tokens": 2}}),
                json.dumps({"id": "stop", "kind": SHUTDOWN, "payload": {}}),
            ]
        )
        + "\n"
    )
    output_stream = io.StringIO()

    with TpRuntime(launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert [response["id"] for response in responses] == ["load", "gen-1", "gen-2", "stop"]
    assert [response["kind"] for response in responses] == [LOAD, GENERATE, GENERATE, SHUTDOWN]
    assert all(response["ok"] for response in responses)
    assert responses[1]["data"]["text"].startswith("decoded:")
    assert responses[2]["data"]["text"].startswith("decoded:")
    assert len(responses[1]["data"]["generated_token_ids"]) == 2
    assert responses[1]["rank_results"][0]["rank"] == 0
    assert responses[3]["data"]["shutdown"] is True
    assert preload_calls == 1
    assert state.should_shutdown is True


def test_worker_protocol_loop_reports_invalid_json_without_stopping() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    input_stream = io.StringIO('not-json\n{"id":"stop","kind":"SHUTDOWN","payload":{}}\n')
    output_stream = io.StringIO()

    with TpRuntime(launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert len(responses) == 2
    assert responses[0]["ok"] is False
    assert responses[0]["kind"] == "UNKNOWN"
    assert "invalid JSON" in responses[0]["error"]
    assert responses[1]["ok"] is True
    assert responses[1]["kind"] == SHUTDOWN
    assert state.should_shutdown is True


def test_worker_shutdown_closes_loaded_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    load_result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))
    session = state.session
    shutdown_result = execute_command(state, WorkerCommand(SHUTDOWN, {}))

    assert load_result.ok is True
    assert shutdown_result.ok is True
    assert session is not None
    assert session._closed is True
    assert state.session is None
    assert state.manifest is None
    assert state.runtime_config is None
    assert state.loaded_model_dir is None
    assert state.should_shutdown is True


def test_worker_reload_closes_previous_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    first_result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))
    first_session = state.session
    second_result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))

    assert first_result.ok is True
    assert second_result.ok is True
    assert first_session is not None
    assert first_session._closed is True
    assert state.session is not None
    assert state.session is not first_session


def test_two_rank_gather_worker_results(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _gather_worker_results_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def _gather_worker_results_worker(rank: int, tmp_path: Path) -> None:
    launch = TpLaunchConfig(
        world_size=2,
        rank=rank,
        local_rank=rank,
        backend="gloo",
        init_method=f"file://{tmp_path / 'worker-gather-dist-init'}",
        device="cpu",
    )
    with TpRuntime(launch) as runtime:
        result = WorkerResult(GENERATE, rank, True, {"rank": rank})
        gathered = gather_worker_results(result, runtime)

    if rank == 0:
        assert gathered == [
            WorkerResult(GENERATE, 0, True, {"rank": 0}),
            WorkerResult(GENERATE, 1, True, {"rank": 1}),
        ]
    else:
        assert gathered is None


def test_two_rank_worker_loop_receives_shutdown(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _worker_loop_shutdown_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def _worker_loop_shutdown_worker(rank: int, tmp_path: Path) -> None:
    launch = TpLaunchConfig(
        world_size=2,
        rank=rank,
        local_rank=rank,
        backend="gloo",
        init_method=f"file://{tmp_path / 'worker-loop-dist-init'}",
        device="cpu",
    )
    with TpRuntime(launch) as runtime:
        state = WorkerState(launch)
        commands = [WorkerCommand(SHUTDOWN, {})] if rank == 0 else None
        results = run_worker_loop(state, runtime, commands)

    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].kind == SHUTDOWN
    assert state.should_shutdown is True
