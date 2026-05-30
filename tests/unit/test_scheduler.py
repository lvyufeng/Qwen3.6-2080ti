from __future__ import annotations

import pytest

from scheduler import RequestScheduler, RequestStatus, SchedulerError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


def test_scheduler_submit_generate_assigns_fifo_ids() -> None:
    scheduler = RequestScheduler(clock=FakeClock())

    first = scheduler.submit_generate("hello", 2)
    second = scheduler.submit_generate("again", 3)

    assert first.request_id == "gen-1"
    assert second.request_id == "gen-2"
    assert first.prompt == "hello"
    assert second.max_new_tokens == 3
    snapshot = scheduler.snapshot()
    assert snapshot.pending == 2
    assert snapshot.running is None
    assert snapshot.total_submitted == 2


def test_scheduler_run_blocking_generate_executes_request() -> None:
    scheduler = RequestScheduler(clock=FakeClock())

    result = scheduler.run_blocking_generate("hello", 2, lambda request: f"done:{request.prompt}:{request.max_new_tokens}")

    assert result.request_id == "gen-1"
    assert result.status is RequestStatus.COMPLETED
    assert result.result == "done:hello:2"
    assert result.error is None
    assert result.queued_seconds >= 0
    assert result.run_seconds >= 0
    snapshot = scheduler.snapshot()
    assert snapshot.pending == 0
    assert snapshot.running is None
    assert snapshot.completed == 1
    assert snapshot.failed == 0
    assert snapshot.total_completed == 1


def test_scheduler_run_next_is_fifo() -> None:
    scheduler = RequestScheduler(clock=FakeClock())
    seen: list[str] = []
    scheduler.submit_generate("first", 1)
    scheduler.submit_generate("second", 1)

    first = scheduler.run_next(lambda request: seen.append(request.prompt) or request.prompt)
    second = scheduler.run_next(lambda request: seen.append(request.prompt) or request.prompt)

    assert seen == ["first", "second"]
    assert first.result == "first"
    assert second.result == "second"
    assert scheduler.snapshot().completed == 2


def test_scheduler_records_failed_request_and_reraises() -> None:
    scheduler = RequestScheduler(clock=FakeClock())

    def fail(_request):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        scheduler.run_blocking_generate("hello", 1, fail)

    snapshot = scheduler.snapshot()
    assert snapshot.pending == 0
    assert snapshot.running is None
    assert snapshot.completed == 0
    assert snapshot.failed == 1
    assert snapshot.total_failed == 1


def test_scheduler_clear_resets_state() -> None:
    scheduler = RequestScheduler(clock=FakeClock())
    result = scheduler.run_blocking_generate("hello", 1, lambda request: request.prompt)
    assert scheduler.result_for(result.request_id) == result

    scheduler.clear()

    snapshot = scheduler.snapshot()
    assert snapshot.pending == 0
    assert snapshot.running is None
    assert snapshot.completed == 0
    assert snapshot.failed == 0
    assert snapshot.total_submitted == 0
    assert snapshot.total_completed == 0
    assert snapshot.total_failed == 0
    assert scheduler.result_for(result.request_id) is None
    assert scheduler.submit_generate("again", 1).request_id == "gen-1"


def test_scheduler_rejects_duplicate_active_request_id() -> None:
    scheduler = RequestScheduler(clock=FakeClock())
    scheduler.submit_generate("hello", 1, request_id="same")

    with pytest.raises(SchedulerError, match="duplicate active request id"):
        scheduler.submit_generate("again", 1, request_id="same")


def test_scheduler_rejects_duplicate_retained_request_id() -> None:
    scheduler = RequestScheduler(clock=FakeClock())
    scheduler.run_blocking_generate("hello", 1, lambda request: request.prompt, request_id="same")

    with pytest.raises(SchedulerError, match="duplicate active request id"):
        scheduler.submit_generate("again", 1, request_id="same")

    scheduler.clear()
    assert scheduler.submit_generate("again", 1, request_id="same").request_id == "same"


def test_scheduler_rejects_invalid_generate_request() -> None:
    scheduler = RequestScheduler(clock=FakeClock())

    with pytest.raises(SchedulerError, match="non-empty"):
        scheduler.submit_generate("hello", 1, request_id="")
    with pytest.raises(SchedulerError, match="positive"):
        scheduler.submit_generate("hello", 0)
    with pytest.raises(SchedulerError, match="no pending request"):
        scheduler.run_next(lambda request: request.prompt)


def test_scheduler_result_for_returns_completed_request() -> None:
    scheduler = RequestScheduler(clock=FakeClock())

    result = scheduler.run_blocking_generate("hello", 2, lambda request: request.prompt)

    assert scheduler.result_for(result.request_id) == result


def test_scheduler_run_next_can_return_failed_request_without_reraising() -> None:
    scheduler = RequestScheduler(clock=FakeClock())
    request = scheduler.submit_generate("hello", 1, request_id="fail-me")

    def fail(_request):
        raise RuntimeError("boom")

    result = scheduler.run_next(fail, reraise=False)

    assert result.request_id == request.request_id
    assert result.status is RequestStatus.FAILED
    assert result.result is None
    assert result.error == "boom"
    assert scheduler.result_for(request.request_id) == result
    snapshot = scheduler.snapshot()
    assert snapshot.pending == 0
    assert snapshot.running is None
    assert snapshot.failed == 1
    assert snapshot.total_failed == 1


def test_scheduler_result_lookup_is_trimmed_with_history() -> None:
    scheduler = RequestScheduler(max_history=1, clock=FakeClock())

    first = scheduler.run_blocking_generate("first", 1, lambda request: request.prompt, request_id="first")
    second = scheduler.run_blocking_generate("second", 1, lambda request: request.prompt, request_id="second")

    assert scheduler.result_for(first.request_id) is None
    assert scheduler.result_for(second.request_id) == second
    snapshot = scheduler.snapshot()
    assert snapshot.completed == 1
    assert snapshot.total_completed == 2


def test_scheduler_is_pending_tracks_pending_only() -> None:
    scheduler = RequestScheduler(clock=FakeClock())
    scheduler.submit_generate("hello", 1, request_id="pending")

    assert scheduler.is_pending("pending") is True
    result = scheduler.run_next(lambda request: request.prompt)

    assert scheduler.is_pending("pending") is False
    assert scheduler.result_for("pending") == result
