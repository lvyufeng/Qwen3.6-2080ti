from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from service import ServiceConfig, WorkerService, create_worker_http_server
from test_engine import _patch_tokenizer, _write_tiny_model
from tp_runtime import TpLaunchConfig, TpRuntime
from worker import LOAD, GENERATE, STATUS, STEP, SUBMIT, WorkerState


def test_worker_service_status_dispatches_protocol_response() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        response = service.status()

    assert response["id"] == "status"
    assert response["kind"] == STATUS
    assert response["ok"] is True
    assert response["control"] == _control_data(rank_count=1)
    assert response["data"]["loaded"] is False


def test_worker_service_load_and_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)

    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        load = service.load_model(str(tmp_path))
        generated = service.generate("hello", 2, request_id="job-1")

    assert load["kind"] == LOAD
    assert load["ok"] is True
    assert generated["id"] == "job-1"
    assert generated["kind"] == GENERATE
    assert generated["ok"] is True
    assert generated["data"]["status"] == "COMPLETED"
    assert generated["data"]["event"]["type"] == "completed"
    assert generated["data"]["text"].startswith("decoded:")


def test_worker_service_stream_generate_emits_queued_token_and_completed_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)

    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        assert service.load_model(str(tmp_path))["ok"] is True
        responses = list(service.stream_generate("hello", 2, request_id="job-1"))

    assert [response["kind"] for response in responses] == [SUBMIT, STEP, STEP]
    assert [response["data"]["event"]["type"] for response in responses] == ["queued", "token", "completed"]
    assert responses[0]["id"] == "job-1"
    assert responses[1]["id"] == "job-1"
    assert responses[2]["id"] == "job-1"
    assert responses[1]["data"]["event"]["token_ids"] == [responses[1]["data"]["progress"]["latest_token_id"]]
    assert responses[2]["data"]["result"]["text"].startswith("decoded:")
    assert responses[2]["data"]["event"]["is_terminal"] is True


def test_worker_http_healthz_and_status() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        with _running_server(service) as client:
            status, health = client("GET", "/healthz")
            assert status == 200
            assert health == {"ok": True, "service": "qwen36-worker"}

            status, response = client("GET", "/v1/status")

    assert status == 200
    assert response["kind"] == STATUS
    assert response["ok"] is True
    assert response["data"]["loaded"] is False


def test_worker_http_load_generate_and_poll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)

    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        with _running_server(service) as client:
            status, load = client("POST", "/v1/load", {"model_dir": str(tmp_path)})
            assert status == 200
            assert load["ok"] is True

            status, generated = client(
                "POST",
                "/v1/generate",
                {"prompt": "hello", "max_new_tokens": 2, "request_id": "job-1", "stream": False},
            )
            assert status == 200

            status, polled = client("GET", "/v1/requests/job-1")

    assert generated["id"] == "job-1"
    assert generated["kind"] == GENERATE
    assert generated["ok"] is True
    assert generated["data"]["status"] == "COMPLETED"
    assert generated["data"]["text"].startswith("decoded:")
    assert polled["id"] == "job-1"
    assert polled["data"]["event"]["type"] == "completed"


def test_worker_http_stream_generate_sends_sse_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tiny_model(tmp_path)
    _patch_tokenizer(monkeypatch, torch.tensor([[1, 2]]))
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)

    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        with _running_server(service) as client:
            assert client("POST", "/v1/load", {"model_dir": str(tmp_path)})[1]["ok"] is True
            status, headers, body = client(
                "POST",
                "/v1/generate",
                {"prompt": "hello", "max_new_tokens": 2, "request_id": "job-1", "stream": True},
                raw=True,
            )

    frames = _parse_sse(body.decode("utf-8"))
    assert status == 200
    assert headers["content-type"] == "text/event-stream"
    assert [event for event, _ in frames] == ["queued", "token", "completed"]
    payloads = [json.loads(data) for _, data in frames]
    assert [payload["data"]["event"]["type"] for payload in payloads] == ["queued", "token", "completed"]
    assert payloads[-1]["id"] == "job-1"
    assert payloads[-1]["data"]["result"]["text"].startswith("decoded:")


def test_worker_http_reports_client_errors() -> None:
    launch = TpLaunchConfig(backend="gloo", device="cpu")
    state = WorkerState(launch)
    with TpRuntime(launch) as runtime:
        service = WorkerService(state, runtime)
        with _running_server(service) as client:
            status, response = client("POST", "/v1/generate", {"prompt": "hello", "max_new_tokens": 0})
            assert status == 400
            assert response["ok"] is False
            assert "max_new_tokens" in response["error"]

            status, response = client("GET", "/missing")
            assert status == 404
            assert response["ok"] is False

            status, response = client("POST", "/v1/generate", "not-json")
            assert status == 400
            assert "invalid JSON" in response["error"]


@contextmanager
def _running_server(service: WorkerService) -> Iterator[Any]:
    server = create_worker_http_server(service, ServiceConfig(host="127.0.0.1", port=0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def request(method: str, path: str, payload: Any | None = None, *, raw: bool = False):
        body = None
        headers = {}
        if payload is not None:
            if isinstance(payload, str):
                body = payload.encode("utf-8")
            else:
                body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            response_body = response.read()
            status = response.status
            response_headers = {key.lower(): value for key, value in response.getheaders()}
        finally:
            conn.close()
        if raw:
            return status, response_headers, response_body
        return status, json.loads(response_body.decode("utf-8"))

    try:
        yield request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _parse_sse(text: str) -> list[tuple[str, str]]:
    frames: list[tuple[str, str]] = []
    for chunk in text.strip().split("\n\n"):
        event = "message"
        data = ""
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            if line.startswith("data: "):
                data = line[len("data: ") :]
        frames.append((event, data))
    return frames


def _control_data(*, rank_count: int) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "response_shape": "rank_aggregate",
        "streaming_ready": True,
        "rank_count": rank_count,
    }
