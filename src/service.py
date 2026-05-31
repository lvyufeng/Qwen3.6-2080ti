from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from tp_runtime import TpRuntime
from worker import LOAD, GENERATE, POLL, STATUS, STEP, SUBMIT, SHUTDOWN, WorkerCommand, WorkerState, dispatch_protocol_command


class ServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    model_dir: str | None = None


class WorkerService:
    def __init__(self, state: WorkerState, runtime: TpRuntime) -> None:
        if runtime.config.rank != 0:
            raise ServiceError("HTTP service transport must run on rank 0")
        self.state = state
        self.runtime = runtime
        self._lock = threading.RLock()

    def dispatch(self, request_id: str | None, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            response = self._dispatch_unlocked(request_id, kind, payload or {})
            return self._maybe_alias_request_id(response)

    def load_model(self, model_dir: str) -> dict[str, Any]:
        return self.dispatch("load", LOAD, {"model_dir": model_dir})

    def status(self) -> dict[str, Any]:
        return self.dispatch("status", STATUS, {})

    def generate(self, prompt: str, max_new_tokens: int, request_id: str | None = None) -> dict[str, Any]:
        payload = _generate_payload(prompt, max_new_tokens, request_id)
        return self.dispatch(request_id, GENERATE, payload)

    def poll(self, request_id: str) -> dict[str, Any]:
        return self.dispatch(request_id, POLL, {"request_id": request_id})

    def shutdown(self) -> dict[str, Any]:
        return self.dispatch("shutdown", SHUTDOWN, {})

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload = _generate_payload(prompt, max_new_tokens, request_id)

        def _iterator() -> Iterator[dict[str, Any]]:
            with self._lock:
                submit = self._maybe_alias_request_id(self._dispatch_unlocked(request_id, SUBMIT, payload))
                yield submit
                if not submit.get("ok", False):
                    return
                request_label = submit.get("id") if isinstance(submit.get("id"), str) else submit["data"].get("request_id")
                while True:
                    step = self._maybe_alias_request_id(self._dispatch_unlocked(request_label, STEP, {}))
                    yield step
                    if not step.get("ok", False) or _response_event_is_terminal_or_idle(step):
                        return

        return _iterator()

    def _dispatch_unlocked(self, request_id: str | None, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = dispatch_protocol_command(self.state, self.runtime, request_id, WorkerCommand(kind, payload))
        if response is None:
            raise ServiceError("rank 0 dispatch unexpectedly returned no response")
        return response

    def _maybe_alias_request_id(self, response: dict[str, Any]) -> dict[str, Any]:
        if response.get("id") is not None:
            return response
        data = response.get("data")
        if not isinstance(data, dict):
            return response
        request_id = data.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return response
        aliased = dict(response)
        aliased["id"] = request_id
        return aliased


def serve_worker_http(state: WorkerState, runtime: TpRuntime, config: ServiceConfig) -> None:
    if runtime.config.rank != 0:
        _run_service_peer_loop(state, runtime)
        return
    service = WorkerService(state, runtime)
    if config.model_dir is not None:
        response = service.load_model(config.model_dir)
        if not response.get("ok", False):
            raise ServiceError(str(response.get("error") or "failed to load model"))
    server = create_worker_http_server(service, config)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def create_worker_http_server(service: WorkerService, config: ServiceConfig) -> ThreadingHTTPServer:
    handler = make_worker_http_handler(service)
    return ThreadingHTTPServer((config.host, config.port), handler)


def make_worker_http_handler(service: WorkerService) -> type[BaseHTTPRequestHandler]:
    class WorkerHttpHandler(BaseHTTPRequestHandler):
        server_version = "Qwen36WorkerHTTP/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True, "service": "qwen36-worker"})
                return
            if parsed.path == "/v1/status":
                self._send_worker_response(service.status())
                return
            request_prefix = "/v1/requests/"
            if parsed.path.startswith(request_prefix):
                request_id = unquote(parsed.path[len(request_prefix) :])
                if not request_id or "/" in request_id:
                    self._send_json(HTTPStatus.NOT_FOUND, _error_payload("unknown route"))
                    return
                self._send_worker_response(service.poll(request_id))
                return
            self._send_json(HTTPStatus.NOT_FOUND, _error_payload("unknown route"))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
                if parsed.path == "/v1/load":
                    model_dir = _require_str(payload, "model_dir")
                    self._send_worker_response(service.load_model(model_dir))
                    return
                if parsed.path == "/v1/generate":
                    prompt = _require_str(payload, "prompt")
                    max_new_tokens = _require_positive_int(payload, "max_new_tokens")
                    request_id = _optional_str(payload, "request_id")
                    if bool(payload.get("stream", False)):
                        self._send_sse_stream(service.stream_generate(prompt, max_new_tokens, request_id))
                    else:
                        self._send_worker_response(service.generate(prompt, max_new_tokens, request_id))
                    return
                if parsed.path == "/v1/shutdown":
                    response = service.shutdown()
                    self._send_worker_response(response)
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                self._send_json(HTTPStatus.NOT_FOUND, _error_payload("unknown route"))
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, _error_payload(str(exc)))

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json_body(self) -> dict[str, Any]:
            length_text = self.headers.get("Content-Length")
            if length_text is None:
                return {}
            try:
                length = int(length_text)
            except ValueError as exc:
                raise ValueError("Content-Length must be an integer") from exc
            body = self.rfile.read(length)
            if not body:
                return {}
            try:
                value = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON body: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _send_worker_response(self, response: dict[str, Any]) -> None:
            self._send_json(HTTPStatus.OK, response)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _send_sse_stream(self, responses: Iterator[dict[str, Any]]) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for response in responses:
                    event = _response_event_type(response)
                    frame = f"event: {event}\ndata: {json.dumps(response, sort_keys=True)}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                return
            finally:
                close = getattr(responses, "close", None)
                if callable(close):
                    close()

    return WorkerHttpHandler


def _run_service_peer_loop(state: WorkerState, runtime: TpRuntime) -> None:
    while not state.should_shutdown:
        dispatch_protocol_command(state, runtime, None, None)


def _generate_payload(prompt: str, max_new_tokens: int, request_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string when provided")
    return value


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _error_payload(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


def _response_event_type(response: dict[str, Any]) -> str:
    data = response.get("data")
    if isinstance(data, dict):
        event = data.get("event")
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            return event["type"]
    return "message"


def _response_event_is_terminal_or_idle(response: dict[str, Any]) -> bool:
    data = response.get("data")
    if not isinstance(data, dict):
        return True
    event = data.get("event")
    if not isinstance(event, dict):
        return True
    if event.get("type") == "idle":
        return True
    return event.get("is_terminal") is True
