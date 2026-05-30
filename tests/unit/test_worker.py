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
    POLL,
    STATUS,
    SUBMIT,
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


def test_worker_status_reports_unloaded_scheduler_state() -> None:
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    result = execute_command(state, WorkerCommand(STATUS, {}))

    assert result.ok is True
    assert result.kind == STATUS
    assert result.data["loaded"] is False
    assert result.data["loaded_model_dir"] is None
    assert result.data["should_shutdown"] is False
    assert result.data["scheduler"] == _empty_scheduler_data()


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
    assert generate_result.data["scheduler"]["request_id"] == "gen-1"
    assert generate_result.data["scheduler"]["status"] == "COMPLETED"
    assert generate_result.data["scheduler"]["queued_seconds"] >= 0
    assert generate_result.data["scheduler"]["run_seconds"] >= 0
    assert generate_result.data["runtime"]["backend"] == "gloo"
    assert generate_result.data["runtime"]["world_size"] == 1
    assert generate_result.data["runtime"]["rank"] == 0
    assert generate_result.data["runtime"]["local_rank"] == 0
    assert generate_result.data["runtime"]["device"] == "cpu"
    assert generate_result.data["model"]["layers"] == 1
    assert generate_result.data["model"]["mapped_tensors"] > 0
    assert generate_result.data["model"]["mapped_bytes"] > 0
    assert generate_result.data["load"]["loaded_tensors"] > 0
    assert generate_result.data["load"]["loaded_bytes"] > 0
    assert generate_result.data["load"]["load_seconds"] >= 0
    assert generate_result.data["timings"]["prefill_seconds"] >= 0
    assert generate_result.data["timings"]["decode_seconds"] >= 0
    assert generate_result.data["timings"]["total_seconds"] >= 0
    assert generate_result.data["throughput"]["decode_tokens_per_second"] > 0
    assert generate_result.data["throughput"]["total_tokens_per_second"] > 0
    assert generate_result.data["dispatch"]["calls"] >= 0
    assert generate_result.data["memory"]["available"] is False


def test_worker_generate_routes_through_scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    load_result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))
    generate_result = execute_command(state, WorkerCommand(GENERATE, {"prompt": "hello", "max_new_tokens": 1}))
    status_result = execute_command(state, WorkerCommand(STATUS, {}))

    assert load_result.ok is True
    assert generate_result.ok is True
    assert generate_result.data["scheduler"]["status"] == "COMPLETED"
    assert status_result.ok is True
    assert status_result.data["loaded"] is True
    assert status_result.data["scheduler"]["pending"] == 0
    assert status_result.data["scheduler"]["running"] is None
    assert status_result.data["scheduler"]["completed"] == 1
    assert status_result.data["scheduler"]["failed"] == 0
    assert status_result.data["scheduler"]["total_submitted"] == 1
    assert status_result.data["scheduler"]["total_completed"] == 1


def test_worker_load_clears_scheduler_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True
    assert execute_command(state, WorkerCommand(GENERATE, {"prompt": "hello", "max_new_tokens": 1})).ok is True
    before_reload = execute_command(state, WorkerCommand(STATUS, {}))
    assert before_reload.data["scheduler"]["completed"] == 1

    reload_result = execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)}))
    after_reload = execute_command(state, WorkerCommand(STATUS, {}))

    assert reload_result.ok is True
    assert after_reload.ok is True
    assert after_reload.data["loaded"] is True
    assert after_reload.data["scheduler"] == _empty_scheduler_data()


def test_worker_submit_requires_loaded_model() -> None:
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    result = execute_command(state, WorkerCommand(SUBMIT, {"prompt": "hello", "max_new_tokens": 1}))

    assert result.ok is False
    assert "not loaded" in (result.error or "")


def test_worker_submit_enqueues_without_generating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True
    submit_result = execute_command(state, WorkerCommand(SUBMIT, {"prompt": "hello", "max_new_tokens": 1}))
    status_result = execute_command(state, WorkerCommand(STATUS, {}))

    assert submit_result.ok is True
    assert submit_result.kind == SUBMIT
    assert submit_result.data["request_id"] == "gen-1"
    assert submit_result.data["status"] == "PENDING"
    assert submit_result.data["pending"] == 1
    assert status_result.data["scheduler"]["pending"] == 1
    assert status_result.data["scheduler"]["completed"] == 0
    assert status_result.data["scheduler"]["total_submitted"] == 1


def test_worker_submit_accepts_explicit_request_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True
    submit_result = execute_command(
        state, WorkerCommand(SUBMIT, {"prompt": "hello", "max_new_tokens": 1, "request_id": "job-1"})
    )

    assert submit_result.ok is True
    assert submit_result.data["request_id"] == "job-1"


def test_worker_poll_unknown_request_id_reports_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True
    poll_result = execute_command(state, WorkerCommand(POLL, {"request_id": "missing"}))

    assert poll_result.ok is True
    assert poll_result.kind == POLL
    assert poll_result.data["request_id"] == "missing"
    assert poll_result.data["found"] is False
    assert poll_result.data["status"] is None
    assert poll_result.data["result"] is None


def test_worker_poll_drains_pending_request_to_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True
    submit_result = execute_command(
        state, WorkerCommand(SUBMIT, {"prompt": "hello", "max_new_tokens": 2, "request_id": "job-1"})
    )
    poll_result = execute_command(state, WorkerCommand(POLL, {"request_id": "job-1"}))
    status_result = execute_command(state, WorkerCommand(STATUS, {}))

    assert submit_result.ok is True
    assert poll_result.ok is True
    assert poll_result.data["found"] is True
    assert poll_result.data["status"] == "COMPLETED"
    assert poll_result.data["error"] is None
    assert poll_result.data["queued_seconds"] >= 0
    assert poll_result.data["run_seconds"] >= 0
    result = poll_result.data["result"]
    assert result["text"].startswith("decoded:")
    assert len(result["generated_token_ids"]) == 2
    assert result["world_size"] == 1
    assert result["runtime"]["backend"] == "gloo"
    assert result["runtime"]["rank"] == 0
    assert result["runtime"]["local_rank"] == 0
    assert result["timings"]["total_seconds"] >= 0
    assert result["throughput"]["total_tokens_per_second"] > 0
    assert status_result.data["scheduler"]["completed"] == 1
    assert status_result.data["scheduler"]["pending"] == 0


def test_worker_poll_reports_failed_job_without_command_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True

    def boom(_prompt: str, _max_new_tokens: int):
        raise RuntimeError("generate kaboom")

    monkeypatch.setattr(state.session, "generate", boom)
    submit_result = execute_command(
        state, WorkerCommand(SUBMIT, {"prompt": "hello", "max_new_tokens": 1, "request_id": "job-1"})
    )
    poll_result = execute_command(state, WorkerCommand(POLL, {"request_id": "job-1"}))
    status_result = execute_command(state, WorkerCommand(STATUS, {}))

    assert submit_result.ok is True
    assert poll_result.ok is True
    assert poll_result.data["found"] is True
    assert poll_result.data["status"] == "FAILED"
    assert "generate kaboom" in poll_result.data["error"]
    assert poll_result.data["result"] is None
    assert status_result.data["scheduler"]["failed"] == 1
    assert status_result.data["scheduler"]["total_failed"] == 1


def test_worker_poll_drains_one_request_per_call_in_fifo_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))

    assert execute_command(state, WorkerCommand(LOAD, {"model_dir": str(tmp_path)})).ok is True
    execute_command(state, WorkerCommand(SUBMIT, {"prompt": "first", "max_new_tokens": 1, "request_id": "a"}))
    execute_command(state, WorkerCommand(SUBMIT, {"prompt": "second", "max_new_tokens": 1, "request_id": "b"}))

    first_poll = execute_command(state, WorkerCommand(POLL, {"request_id": "b"}))
    second_poll = execute_command(state, WorkerCommand(POLL, {"request_id": "b"}))

    assert first_poll.ok is True
    assert first_poll.data["found"] is True
    assert first_poll.data["status"] == "PENDING"
    assert first_poll.data["result"] is None
    assert second_poll.ok is True
    assert second_poll.data["status"] == "COMPLETED"
    assert second_poll.data["result"]["text"].startswith("decoded:")
    assert execute_command(state, WorkerCommand(POLL, {"request_id": "a"})).data["status"] == "COMPLETED"


def test_worker_protocol_loop_submit_poll_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))
    input_stream = _jsonl_stream(
        [
            {"id": "load", "kind": LOAD, "payload": {"model_dir": str(tmp_path)}},
            {"id": "submit", "kind": SUBMIT, "payload": {"prompt": "hello", "max_new_tokens": 2, "request_id": "job-1"}},
            {"id": "status-before", "kind": STATUS, "payload": {}},
            {"id": "poll", "kind": POLL, "payload": {"request_id": "job-1"}},
            {"id": "status-after", "kind": STATUS, "payload": {}},
            {"id": "stop", "kind": SHUTDOWN, "payload": {}},
        ]
    )
    output_stream = io.StringIO()

    with TpRuntime(state.launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = _read_jsonl(output_stream)
    assert [response["id"] for response in responses] == [
        "load",
        "submit",
        "status-before",
        "poll",
        "status-after",
        "stop",
    ]
    assert [response["kind"] for response in responses] == [LOAD, SUBMIT, STATUS, POLL, STATUS, SHUTDOWN]
    assert all(response["ok"] for response in responses)
    submit = responses[1]
    assert submit["data"]["request_id"] == "job-1"
    assert submit["data"]["status"] == "PENDING"
    assert responses[2]["data"]["scheduler"]["pending"] == 1
    assert responses[2]["data"]["scheduler"]["completed"] == 0
    poll = responses[3]
    assert poll["data"]["request_id"] == "job-1"
    assert poll["data"]["status"] == "COMPLETED"
    assert poll["data"]["result"]["text"].startswith("decoded:")
    assert responses[4]["data"]["scheduler"]["completed"] == 1
    assert responses[4]["data"]["scheduler"]["pending"] == 0
    assert responses[5]["data"]["shutdown"] is True
    assert state.should_shutdown is True


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


def test_worker_protocol_loop_status_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    state = WorkerState(TpLaunchConfig(backend="gloo", device="cpu"))
    input_stream = _jsonl_stream(
        [
            {"id": "status-before", "kind": STATUS, "payload": {}},
            {"id": "load", "kind": LOAD, "payload": {"model_dir": str(tmp_path)}},
            {"id": "gen", "kind": GENERATE, "payload": {"prompt": "hello", "max_new_tokens": 1}},
            {"id": "status-after", "kind": STATUS, "payload": {}},
            {"id": "stop", "kind": SHUTDOWN, "payload": {}},
        ]
    )
    output_stream = io.StringIO()

    with TpRuntime(state.launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = _read_jsonl(output_stream)
    assert [response["id"] for response in responses] == ["status-before", "load", "gen", "status-after", "stop"]
    assert [response["kind"] for response in responses] == [STATUS, LOAD, GENERATE, STATUS, SHUTDOWN]
    assert all(response["ok"] for response in responses)
    assert responses[0]["data"]["loaded"] is False
    assert responses[0]["data"]["scheduler"] == _empty_scheduler_data()
    assert responses[2]["data"]["scheduler"]["status"] == "COMPLETED"
    assert responses[3]["data"]["loaded"] is True
    assert responses[3]["data"]["scheduler"]["completed"] == 1
    assert responses[3]["data"]["scheduler"]["total_completed"] == 1
    assert responses[4]["data"]["shutdown"] is True
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


def test_worker_protocol_loop_reports_invalid_envelope_without_stopping() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    input_stream = _jsonl_stream(
        [
            {"id": "bad", "kind": SHUTDOWN, "payload": []},
            {"id": "stop", "kind": SHUTDOWN, "payload": {}},
        ]
    )
    output_stream = io.StringIO()

    with TpRuntime(launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = _read_jsonl(output_stream)
    assert len(responses) == 2
    assert responses[0]["id"] == "bad"
    assert responses[0]["ok"] is False
    assert responses[0]["kind"] == "UNKNOWN"
    assert "payload must be a dict" in responses[0]["error"]
    assert responses[1]["ok"] is True
    assert responses[1]["kind"] == SHUTDOWN
    assert state.should_shutdown is True


def test_worker_protocol_loop_reports_unknown_command_without_stopping() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    input_stream = _jsonl_stream(
        [
            {"id": "bogus", "kind": "BOGUS", "payload": {}},
            {"id": "stop", "kind": SHUTDOWN, "payload": {}},
        ]
    )
    output_stream = io.StringIO()

    with TpRuntime(launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = _read_jsonl(output_stream)
    assert len(responses) == 2
    assert responses[0]["id"] == "bogus"
    assert responses[0]["kind"] == "BOGUS"
    assert responses[0]["ok"] is False
    assert responses[0]["rank_results"][0]["rank"] == 0
    assert "unknown worker command" in responses[0]["rank_results"][0]["error"]
    assert responses[1]["ok"] is True
    assert responses[1]["kind"] == SHUTDOWN
    assert state.should_shutdown is True


def test_worker_protocol_loop_shutdown_on_eof() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    input_stream = io.StringIO("\n\n")
    output_stream = io.StringIO()

    with TpRuntime(launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    responses = _read_jsonl(output_stream)
    assert len(responses) == 1
    assert responses[0]["id"] is None
    assert responses[0]["kind"] == SHUTDOWN
    assert responses[0]["ok"] is True
    assert responses[0]["data"]["shutdown"] is True
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


def test_two_rank_worker_protocol_loop_load_generate_generate_shutdown(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-worker-protocol-model"
    model_dir.mkdir()
    _write_tiny_model(model_dir)
    torch.multiprocessing.spawn(
        _worker_protocol_e2e_worker,
        args=(tmp_path, model_dir),
        nprocs=2,
        join=True,
    )


def _worker_protocol_e2e_worker(rank: int, tmp_path: Path, model_dir: Path) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
        launch = _two_rank_launch(rank, tmp_path, "worker-protocol-e2e-dist-init")
        state = WorkerState(launch)
        input_stream = (
            _jsonl_stream(
                [
                    {"id": "load", "kind": LOAD, "payload": {"model_dir": str(model_dir)}},
                    {"id": "gen-1", "kind": GENERATE, "payload": {"prompt": "hello", "max_new_tokens": 2}},
                    {"id": "gen-2", "kind": GENERATE, "payload": {"prompt": "again", "max_new_tokens": 2}},
                    {"id": "stop", "kind": SHUTDOWN, "payload": {}},
                ]
            )
            if rank == 0
            else io.StringIO()
        )
        output_stream = io.StringIO()

        with TpRuntime(launch) as runtime:
            run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    assert state.should_shutdown is True
    assert state.session is None
    assert state.manifest is None
    assert state.runtime_config is None
    assert state.loaded_model_dir is None
    if rank != 0:
        assert output_stream.getvalue() == ""
        return

    responses = _read_jsonl(output_stream)
    assert [response["id"] for response in responses] == ["load", "gen-1", "gen-2", "stop"]
    assert [response["kind"] for response in responses] == [LOAD, GENERATE, GENERATE, SHUTDOWN]
    for response, kind in zip(responses, [LOAD, GENERATE, GENERATE, SHUTDOWN], strict=True):
        _assert_two_rank_response(response, kind)

    load = responses[0]
    assert load["data"]["world_size"] == 2
    assert load["data"]["rank"] == 0
    assert load["data"]["layers"] == 1
    assert load["data"]["loaded_tensors"] > 0
    assert load["data"]["loaded_bytes"] > 0

    for generate in responses[1:3]:
        assert generate["data"]["world_size"] == 2
        assert generate["data"]["rank"] == 0
        assert generate["data"]["device"] == "cpu"
        assert generate["data"]["prompt_tokens"] == 2
        assert generate["data"]["max_new_tokens"] == 2
        assert generate["data"]["all_finite"] is True
        assert len(generate["data"]["generated_token_ids"]) == 2
        assert generate["data"]["text"].startswith("decoded:")

    assert responses[3]["data"]["shutdown"] is True
    assert all(result["data"]["shutdown"] for result in responses[3]["rank_results"])


def test_two_rank_worker_protocol_loop_submit_poll_shutdown(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-worker-submit-poll-model"
    model_dir.mkdir()
    _write_tiny_model(model_dir)
    torch.multiprocessing.spawn(
        _worker_protocol_submit_poll_worker,
        args=(tmp_path, model_dir),
        nprocs=2,
        join=True,
    )


def _worker_protocol_submit_poll_worker(rank: int, tmp_path: Path, model_dir: Path) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
        launch = _two_rank_launch(rank, tmp_path, "worker-protocol-submit-poll-dist-init")
        state = WorkerState(launch)
        input_stream = (
            _jsonl_stream(
                [
                    {"id": "load", "kind": LOAD, "payload": {"model_dir": str(model_dir)}},
                    {
                        "id": "submit-envelope",
                        "kind": SUBMIT,
                        "payload": {"prompt": "hello", "max_new_tokens": 2, "request_id": "job-1"},
                    },
                    {"id": "poll-envelope", "kind": POLL, "payload": {"request_id": "job-1"}},
                    {"id": "stop", "kind": SHUTDOWN, "payload": {}},
                ]
            )
            if rank == 0
            else io.StringIO()
        )
        output_stream = io.StringIO()

        with TpRuntime(launch) as runtime:
            run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    assert state.should_shutdown is True
    assert state.session is None
    if rank != 0:
        assert output_stream.getvalue() == ""
        return

    responses = _read_jsonl(output_stream)
    assert [response["id"] for response in responses] == ["load", "submit-envelope", "poll-envelope", "stop"]
    assert [response["kind"] for response in responses] == [LOAD, SUBMIT, POLL, SHUTDOWN]
    for response, kind in zip(responses, [LOAD, SUBMIT, POLL, SHUTDOWN], strict=True):
        _assert_two_rank_response(response, kind)

    submit = responses[1]
    assert submit["data"]["request_id"] == "job-1"
    assert submit["data"]["status"] == "PENDING"
    assert submit["data"]["scheduler"]["pending"] == 1

    poll = responses[2]
    assert poll["data"]["request_id"] == "job-1"
    assert poll["data"]["status"] == "COMPLETED"
    assert poll["data"]["result"]["world_size"] == 2
    assert poll["data"]["result"]["rank"] == 0
    assert poll["data"]["result"]["runtime"]["rank"] == 0
    assert poll["data"]["result"]["timings"]["total_seconds"] >= 0
    assert poll["data"]["result"]["text"].startswith("decoded:")
    assert poll["data"]["scheduler"]["completed"] == 1
    assert responses[3]["data"]["shutdown"] is True
    assert all(result["data"]["shutdown"] for result in responses[3]["rank_results"])


def test_two_rank_worker_protocol_loop_aggregates_generate_before_load_error(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _worker_protocol_error_worker,
        args=(tmp_path,),
        nprocs=2,
        join=True,
    )


def _worker_protocol_error_worker(rank: int, tmp_path: Path) -> None:
    launch = _two_rank_launch(rank, tmp_path, "worker-protocol-error-dist-init")
    state = WorkerState(launch)
    input_stream = (
        _jsonl_stream(
            [
                {"id": "gen-before-load", "kind": GENERATE, "payload": {"prompt": "hello", "max_new_tokens": 1}},
                {"id": "stop", "kind": SHUTDOWN, "payload": {}},
            ]
        )
        if rank == 0
        else io.StringIO()
    )
    output_stream = io.StringIO()

    with TpRuntime(launch) as runtime:
        run_worker_protocol_loop(state, runtime, input_stream, output_stream)

    assert state.should_shutdown is True
    assert state.session is None
    if rank != 0:
        assert output_stream.getvalue() == ""
        return

    responses = _read_jsonl(output_stream)
    assert [response["id"] for response in responses] == ["gen-before-load", "stop"]
    generate = responses[0]
    _assert_two_rank_response(generate, GENERATE, ok=False)
    assert generate["data"] == {}
    assert "rank 0:" in generate["error"]
    assert "rank 1:" in generate["error"]
    assert all("not loaded" in result["error"] for result in generate["rank_results"])
    assert all(result["ok"] is False for result in generate["rank_results"])
    _assert_two_rank_response(responses[1], SHUTDOWN)
    assert responses[1]["data"]["shutdown"] is True


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
    launch = _two_rank_launch(rank, tmp_path, "worker-loop-dist-init")
    with TpRuntime(launch) as runtime:
        state = WorkerState(launch)
        commands = [WorkerCommand(SHUTDOWN, {})] if rank == 0 else None
        results = run_worker_loop(state, runtime, commands)

    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].kind == SHUTDOWN
    assert state.should_shutdown is True


def _two_rank_launch(rank: int, tmp_path: Path, name: str) -> TpLaunchConfig:
    return TpLaunchConfig(
        world_size=2,
        rank=rank,
        local_rank=rank,
        backend="gloo",
        init_method=f"file://{tmp_path / name}",
        device="cpu",
    )


def _jsonl_stream(commands: list[dict[str, object]]) -> io.StringIO:
    return io.StringIO("\n".join(json.dumps(command) for command in commands) + "\n")


def _read_jsonl(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def _assert_two_rank_response(response: dict[str, object], kind: str, *, ok: bool = True) -> None:
    assert response["kind"] == kind
    assert response["ok"] is ok
    rank_results = response["rank_results"]
    assert isinstance(rank_results, list)
    assert [result["rank"] for result in rank_results] == [0, 1]
    assert all(result["kind"] == kind for result in rank_results)
    assert all(result["ok"] is ok for result in rank_results)


def _empty_scheduler_data() -> dict[str, object]:
    return {
        "pending": 0,
        "running": None,
        "completed": 0,
        "failed": 0,
        "total_submitted": 0,
        "total_completed": 0,
        "total_failed": 0,
    }
