from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar, cast


class SchedulerError(RuntimeError):
    pass


class RequestStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    created_at: float


T = TypeVar("T")


@dataclass(frozen=True)
class ScheduledResult(Generic[T]):
    request_id: str
    status: RequestStatus
    result: T | None
    error: str | None
    queued_seconds: float
    run_seconds: float


@dataclass(frozen=True)
class SchedulerSnapshot:
    pending: int
    running: str | None
    completed: int
    failed: int
    total_submitted: int
    total_completed: int
    total_failed: int


class RequestScheduler:
    def __init__(self, *, max_history: int = 128, clock: Callable[[], float] | None = None) -> None:
        if max_history < 0:
            raise SchedulerError(f"max_history must be non-negative, got {max_history}")
        self.max_history = max_history
        self.clock = time.perf_counter if clock is None else clock
        self._pending: deque[GenerateRequest] = deque()
        self._running: GenerateRequest | None = None
        self._running_started_at: float | None = None
        self._completed: deque[ScheduledResult[object]] = deque()
        self._failed: deque[ScheduledResult[object]] = deque()
        self._result_by_id: dict[str, ScheduledResult[object]] = {}
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._next_id = 1

    def submit_generate(self, prompt: str, max_new_tokens: int, *, request_id: str | None = None) -> GenerateRequest:
        if max_new_tokens <= 0:
            raise SchedulerError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if request_id is None:
            request_id = self._next_request_id()
        elif not isinstance(request_id, str) or not request_id:
            raise SchedulerError("request_id must be a non-empty string")
        if self._has_reserved_request_id(request_id):
            raise SchedulerError(f"duplicate active request id: {request_id}")
        request = GenerateRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            created_at=self.clock(),
        )
        self._pending.append(request)
        self._total_submitted += 1
        return request

    def begin_next(self) -> GenerateRequest:
        if self._running is not None:
            raise SchedulerError(f"request is already running: {self._running.request_id}")
        try:
            request = self._pending.popleft()
        except IndexError as exc:
            raise SchedulerError("no pending request to run") from exc
        self._running = request
        self._running_started_at = self.clock()
        return request

    def running_request(self) -> GenerateRequest | None:
        return self._running

    def complete_running(self, result: T) -> ScheduledResult[T]:
        request, started_at = self._require_running()
        end = self.clock()
        scheduled = ScheduledResult(
            request_id=request.request_id,
            status=RequestStatus.COMPLETED,
            result=result,
            error=None,
            queued_seconds=_elapsed_seconds(request.created_at, started_at),
            run_seconds=_elapsed_seconds(started_at, end),
        )
        self._record_terminal(cast(ScheduledResult[object], scheduled))
        self._clear_running()
        return scheduled

    def fail_running(self, error: BaseException | str) -> ScheduledResult[object]:
        request, started_at = self._require_running()
        end = self.clock()
        scheduled = ScheduledResult(
            request_id=request.request_id,
            status=RequestStatus.FAILED,
            result=None,
            error=str(error),
            queued_seconds=_elapsed_seconds(request.created_at, started_at),
            run_seconds=_elapsed_seconds(started_at, end),
        )
        self._record_terminal(scheduled)
        self._clear_running()
        return scheduled

    def run_next(self, executor: Callable[[GenerateRequest], T], *, reraise: bool = True) -> ScheduledResult[T]:
        request = self.begin_next()
        try:
            value = executor(request)
        except Exception as exc:
            scheduled = self.fail_running(exc)
            if reraise:
                raise
            return cast(ScheduledResult[T], scheduled)
        return self.complete_running(value)

    def run_blocking_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        executor: Callable[[GenerateRequest], T],
        *,
        request_id: str | None = None,
    ) -> ScheduledResult[T]:
        self.submit_generate(prompt, max_new_tokens, request_id=request_id)
        return self.run_next(executor)

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            pending=len(self._pending),
            running=self._running.request_id if self._running is not None else None,
            completed=len(self._completed),
            failed=len(self._failed),
            total_submitted=self._total_submitted,
            total_completed=self._total_completed,
            total_failed=self._total_failed,
        )

    def clear(self) -> None:
        self._pending.clear()
        self._clear_running()
        self._completed.clear()
        self._failed.clear()
        self._result_by_id.clear()
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._next_id = 1

    def result_for(self, request_id: str) -> ScheduledResult[object] | None:
        return self._result_by_id.get(request_id)

    def is_pending(self, request_id: str) -> bool:
        return any(request.request_id == request_id for request in self._pending)

    def _next_request_id(self) -> str:
        while True:
            request_id = f"gen-{self._next_id}"
            self._next_id += 1
            if not self._has_reserved_request_id(request_id):
                return request_id

    def _has_active_request(self, request_id: str) -> bool:
        return any(request.request_id == request_id for request in self._pending) or (
            self._running is not None and self._running.request_id == request_id
        )

    def _has_reserved_request_id(self, request_id: str) -> bool:
        return self._has_active_request(request_id) or request_id in self._result_by_id

    def _record_terminal(self, scheduled: ScheduledResult[object]) -> None:
        if scheduled.status is RequestStatus.COMPLETED:
            self._completed.append(scheduled)
            self._total_completed += 1
            self._result_by_id[scheduled.request_id] = scheduled
            self._trim_history(self._completed)
            return
        if scheduled.status is RequestStatus.FAILED:
            self._failed.append(scheduled)
            self._total_failed += 1
            self._result_by_id[scheduled.request_id] = scheduled
            self._trim_history(self._failed)
            return
        raise SchedulerError(f"unexpected terminal status: {scheduled.status}")

    def _require_running(self) -> tuple[GenerateRequest, float]:
        request = self._running
        started_at = self._running_started_at
        if request is None or started_at is None:
            raise SchedulerError("no running request to complete")
        return request, started_at

    def _clear_running(self) -> None:
        self._running = None
        self._running_started_at = None

    def _trim_history(self, history: deque[ScheduledResult[object]]) -> None:
        while len(history) > self.max_history:
            removed = history.popleft()
            if self._result_by_id.get(removed.request_id) is removed:
                del self._result_by_id[removed.request_id]


def _elapsed_seconds(start: float, end: float) -> float:
    return max(0.0, end - start)
