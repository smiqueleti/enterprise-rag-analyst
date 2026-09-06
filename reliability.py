"""Async backpressure, deadlines, and transient-failure circuit breakers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AdmissionRejected(RuntimeError):
    """Raised when the bounded request queue is full."""


class QueueTimedOut(RuntimeError):
    """Raised when a provider semaphore cannot be acquired in time."""


class RequestDeadlineExceeded(TimeoutError):
    """Raised when the original end-to-end request budget is exhausted."""


class CircuitOpen(RuntimeError):
    """Raised while a provider circuit is open."""


@dataclass
class RequestDeadline:
    seconds: float
    started: float = field(default_factory=time.perf_counter)

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.perf_counter() - self.started))

    def require(self, minimum: float = 0.0) -> float:
        remaining = self.remaining
        if remaining <= minimum:
            raise RequestDeadlineExceeded("The end-to-end request deadline was exhausted.")
        return remaining


class AdmissionController:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self._pending = 0
        self._lock = asyncio.Lock()
        self.admitted = 0
        self.rejected = 0
        self.maximum_depth = 0

    async def admit(self) -> None:
        async with self._lock:
            if self._pending >= self.maximum:
                self.rejected += 1
                raise AdmissionRejected("The service request queue is at capacity.")
            self._pending += 1
            self.admitted += 1
            self.maximum_depth = max(self.maximum_depth, self._pending)

    async def release(self) -> None:
        async with self._lock:
            self._pending = max(0, self._pending - 1)

    def snapshot(self) -> dict[str, int]:
        return {
            "pending": self._pending,
            "maximum": self.maximum,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "maximum_queue_depth": self.maximum_depth,
        }


class ProviderGate:
    def __init__(self, name: str, maximum: int, queue_timeout: float) -> None:
        self.name = name
        self.maximum = maximum
        self.queue_timeout = queue_timeout
        self._semaphore = asyncio.Semaphore(maximum)
        self.active = 0
        self.queued = 0
        self.maximum_queued = 0
        self.queue_timeouts = 0
        self.calls = 0

    async def run(
        self,
        operation: Callable[[], Awaitable[Any]],
        deadline: RequestDeadline,
        minimum_budget: float,
        observations: list[dict[str, Any]],
    ) -> Any:
        queued_at = time.perf_counter()
        self.calls += 1
        self.queued += 1
        self.maximum_queued = max(self.maximum_queued, self.queued)
        try:
            timeout = min(self.queue_timeout, deadline.require(minimum_budget))
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            except TimeoutError as error:
                self.queue_timeouts += 1
                raise QueueTimedOut(f"{self.name} provider queue timed out.") from error
        finally:
            self.queued = max(0, self.queued - 1)

        queue_wait_ms = (time.perf_counter() - queued_at) * 1000
        try:
            remaining = deadline.require(minimum_budget)
        except Exception:
            self._semaphore.release()
            raise
        self.active += 1
        provider_started = time.perf_counter()
        observation = {
            "stage": self.name,
            "queue_wait_ms": queue_wait_ms,
            "provider_execution_ms": None,
        }
        observations.append(observation)
        task = asyncio.create_task(operation())
        deferred_release = False

        def release_slot(_: asyncio.Task | None = None) -> None:
            observation["provider_execution_ms"] = (
                time.perf_counter() - provider_started
            ) * 1000
            self.active = max(0, self.active - 1)
            self._semaphore.release()

        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            return result
        except TimeoutError as error:
            deferred_release = True
            task.add_done_callback(release_slot)
            raise RequestDeadlineExceeded(
                f"The request deadline expired during {self.name}."
            ) from error
        except asyncio.CancelledError:
            deferred_release = True
            task.add_done_callback(release_slot)
            raise
        finally:
            if not deferred_release:
                release_slot()

    def snapshot(self) -> dict[str, int | float]:
        return {
            "limit": self.maximum,
            "active": self.active,
            "queued": self.queued,
            "maximum_queued": self.maximum_queued,
            "queue_timeouts": self.queue_timeouts,
            "calls": self.calls,
            "queue_timeout_rate": self.queue_timeouts / self.calls if self.calls else 0.0,
        }


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int,
        recovery_seconds: float,
        half_open_max_calls: int,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.half_open_max_calls = half_open_max_calls
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.half_open_active = 0
        self.transitions: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def before_call(self) -> None:
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.perf_counter() - self.opened_at < self.recovery_seconds:
                    raise CircuitOpen(f"{self.name} circuit is open.")
                self._transition(CircuitState.HALF_OPEN, "recovery_interval_elapsed")
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_active >= self.half_open_max_calls:
                    raise CircuitOpen(f"{self.name} circuit half-open probe is busy.")
                self.half_open_active += 1

    async def success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_active = max(0, self.half_open_active - 1)
                self._transition(CircuitState.CLOSED, "probe_succeeded")
            self.failures = 0

    async def failure(self, transient: bool) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_active = max(0, self.half_open_active - 1)
            if not transient:
                return
            self.failures += 1
            if self.state == CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.opened_at = time.perf_counter()
                self._transition(CircuitState.OPEN, "transient_failure_threshold")

    def _transition(self, state: CircuitState, reason: str) -> None:
        if state == self.state:
            return
        previous = self.state
        self.state = state
        self.transitions.append(
            {
                "provider": self.name,
                "from": previous.value,
                "to": state.value,
                "reason": reason,
                "monotonic_timestamp": time.perf_counter(),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failures": self.failures,
            "transitions": list(self.transitions),
        }
