from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from test_engine import _patch_tokenizer, _write_tiny_model
from tp_runtime import TpLaunchConfig, TpRuntime
from worker import GENERATE, LOAD, SHUTDOWN, WorkerCommand, WorkerResult, WorkerState, broadcast_command, execute_command, run_worker_loop


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
