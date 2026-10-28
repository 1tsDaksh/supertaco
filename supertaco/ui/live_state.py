"""Background execution for the Live Run wizard (worker thread, no Streamlit APIs).

Why a worker thread: the supervisor loop takes minutes, and Streamlit aborts a
running script at any ``st.*`` yield point when the client asks it to stop
(page reload, reconnect, session shutdown, idle timeout) — see
``ScriptRunner._maybe_handle_execution_control_request``. When the loop ran in the
script thread, a finished run (zip saved, ``run_succeeded`` emitted) could be torn
down before the wizard advanced, leaving the UI pinned on "running" forever.

Here the loop runs in a daemon thread that touches **no** Streamlit API. The UI
polls a :class:`LiveRun` holder (thread-safe snapshots) and advances the wizard
once ``done`` is set, so the report can never be blocked by a stuck script run.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import supertaco.loop as loop_mod
from supertaco.loop import Event, extract_loss_points

__all__ = [
    "LiveRun",
    "start_run",
    "get_run",
    "adoptable",
    "clear_run",
    "clear_all",
    "clear_finished",
]


@dataclass
class LiveRun:
    """Thread-safe holder for one run: everything the UI needs, nothing live."""

    token: str
    goal: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    done: bool = False
    error: str | None = None
    result: Any = None
    events: list[Event] = field(default_factory=list)
    term_lines: list[str] = field(default_factory=list)
    loss_points: list[float] = field(default_factory=list)
    streaming: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    # ─── worker callbacks (never touch Streamlit) ───
    def on_line(self, line: str) -> None:
        with self.lock:
            self.streaming = True
            self.term_lines.append(line[-500:])
            self.loss_points.extend(p for p in extract_loss_points(line) if p == p)

    def on_event(self, event: Event) -> None:
        with self.lock:
            self.events.append(event)
            # The loop replays the full log text in ``logs_produced``; only use it
            # when the transport never streamed lines (otherwise lines duplicate).
            if event.type == "logs_produced" and not self.streaming:
                text = event.data.get("text") or ""
                self.term_lines.extend(text.splitlines() or [""])

    # ─── UI-side snapshots ───
    def events_snapshot(self) -> list[Event]:
        with self.lock:
            return list(self.events)

    def lines_snapshot(self) -> list[str]:
        with self.lock:
            return list(self.term_lines)

    def loss_snapshot(self) -> list[float]:
        with self.lock:
            return list(self.loss_points)


_RUNS: dict[str, LiveRun] = {}
_REGISTRY_LOCK = threading.Lock()


def start_run(
    *,
    config: dict,
    retries: int,
    prompts: list[str] | None = None,
    judge_backend: str = "direct",
    goal: str | None = None,
) -> LiveRun:
    """Register a run and start its worker thread. Returns immediately."""
    run = LiveRun(token=uuid.uuid4().hex[:12], goal=goal)
    with _REGISTRY_LOCK:
        _RUNS[run.token] = run
    threading.Thread(
        target=_execute,
        args=(run, config, retries, prompts, judge_backend),
        name=f"supertaco-run-{run.token}",
        daemon=True,
    ).start()
    return run


def _execute(
    run: LiveRun,
    config: dict,
    retries: int,
    prompts: list[str] | None,
    judge_backend: str,
) -> None:
    """Run the supervisor loop off the Streamlit thread; never raise."""
    try:
        from supertaco.gpu.transport import ColabTransport

        result = loop_mod.run_training_loop(
            config,
            transport=ColabTransport(),
            on_event=run.on_event,
            on_line=run.on_line,
            llm=loop_mod.make_llm(),
            max_retries=retries,
            runs_dir="configs/runs",
            prompts=prompts,
            judge_backend=judge_backend,
            goal=run.goal or None,
        )
        run.result = result
        if not result.success:
            run.error = result.error
    except Exception as exc:  # a crash must still land on the report
        run.error = f"Run crashed: {type(exc).__name__}: {exc}"
    finally:
        run.finished = time.time()
        run.done = True


def get_run(token: str | None) -> LiveRun | None:
    if not token:
        return None
    with _REGISTRY_LOCK:
        return _RUNS.get(token)


ADOPT_FINISHED_WINDOW = 300.0  # seconds: a reload right after training still finds the report


def adoptable() -> LiveRun | None:
    """Most recent run a fresh session should re-attach to.

    An active run always qualifies. A run that finished moments ago qualifies too:
    reloading the tab at the end of a 7-minute run must land on its report rather
    than throwing the result away. Older finished runs do not — a visitor opening
    the app later starts at the goal step.
    """
    with _REGISTRY_LOCK:
        runs = sorted(_RUNS.values(), key=lambda r: r.started, reverse=True)
    for run in runs:
        if not run.done:
            return run
        if run.finished is not None and time.time() - run.finished <= ADOPT_FINISHED_WINDOW:
            return run
        break
    return None


def clear_run(token: str | None) -> None:
    if not token:
        return
    with _REGISTRY_LOCK:
        _RUNS.pop(token, None)


def clear_all() -> None:
    """Drop every registered run (session reset / test isolation)."""
    with _REGISTRY_LOCK:
        _RUNS.clear()


def clear_finished() -> None:
    """Forget runs that already finished, never an in-flight one."""
    with _REGISTRY_LOCK:
        for token in [t for t, r in _RUNS.items() if r.done]:
            _RUNS.pop(token, None)
