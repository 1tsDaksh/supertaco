"""Windows-friendly wrapper around google-colab-cli for one-shot GPU training.

Runs the deterministic lifecycle `new -> exec -> download -> stop` (the
ephemeral `colab run` command does not reliably retrieve artifacts), streams
CLI output line-by-line for live UI feedback, and always attempts `stop` so a
failed run never leaks a billable VM.

google-colab-cli officially supports Linux/macOS only; on Windows it imports
cleanly once a `termios` stub exists (only its interactive console commands
need the real module). `ensure_windows_shims` writes that stub idempotently.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

DEFAULT_CLI = Path.home() / ".venv-colab" / "Scripts" / "colab.exe"

AUTH_MARKERS = ("To authorize colab-cli", "Enter the authorization code")

_TERMIOIS_STUB = '''"""Windows stub for POSIX termios (google-colab-cli console commands only)."""

TCSANOW = 0
TCSADRAIN = 1
TCSAFLUSH = 2

IGNBRK = 0o1
BRKINT = 0o2
IGNPAR = 0o4
PARMRK = 0o10
INPCK = 0o20
ISTRIP = 0o40
INLCR = 0o100
IGNCR = 0o200
ICRNL = 0o400
IXON = 0o2000
IXOFF = 0o10000
IXANY = 0o4000
OPOST = 0o1

ECHO = 0o10
ECHOE = 0o20
ECHOK = 0o40
ECHONL = 0o100
ICANON = 0o2
ISIG = 0o1
IEXTEN = 0o100000
NOFLSH = 0o200
TOSTOP = 0o10000

CSIZE = 0o60
CS8 = 0o60
PARENB = 0o400

VMIN = 6
VTIME = 5


def _unavailable(*args, **kwargs):
    raise RuntimeError("termios is unavailable on Windows")


tcgetattr = _unavailable
tcsetattr = _unavailable
cfmakeraw = _unavailable
tcsendbreak = _unavailable
tcdrain = _unavailable
tcflush = _unavailable
tcflow = _unavailable
'''


class ColabError(RuntimeError):
    """A Colab CLI failure with a user-actionable message (never a traceback)."""


@dataclass
class TrainingOutcome:
    success: bool
    artifact: Optional[Path] = None
    error: Optional[str] = None
    timed_out: bool = False
    logs: list = field(default_factory=list)


CliArg = Union[str, Sequence[str], None]


def resolve_cli(explicit: CliArg = None) -> list[str]:
    """Build the CLI argv prefix, validating existence; env override supported."""
    if isinstance(explicit, (list, tuple)):
        return list(explicit)
    raw = explicit or os.environ.get("COLAB_CLI_EXE") or str(DEFAULT_CLI)
    path = Path(raw)
    if not path.exists():
        raise ColabError(
            f"colab CLI not found at {path}. Install it with: "
            f"python -m venv {DEFAULT_CLI.parent.parent} && "
            f"{DEFAULT_CLI.parent / 'pip'} install google-colab-cli "
            "(or set COLAB_CLI_EXE)."
        )
    if os.name == "nt":
        _ensure_termios_stub(path)
    return [str(path)]


def _ensure_termios_stub(cli_path: Path) -> None:
    """google-colab-cli imports termios at CLI import time; stub it on Windows."""
    site = cli_path.parent.parent / "Lib" / "site-packages"
    stub = site / "termios.py"
    if stub.exists() or not site.is_dir():
        return
    try:
        stub.write_text(_TERMIOIS_STUB, encoding="utf-8")
    except OSError as exc:
        raise ColabError(f"could not write termios stub at {stub}: {exc}") from exc


def _stream(
    base: list[str],
    args: list[str],
    on_line: Optional[Callable[[str], None]],
    *,
    deadline_s: float,
) -> tuple[int, bool, list[str]]:
    """Run one CLI command, forwarding output lines. Returns (rc, timed_out, lines)."""
    cmd = base + args
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,  # auth prompts must never read the server console
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise ColabError(f"failed to launch {cmd[0]}: {exc}") from exc

    got: queue.Queue[Optional[str]] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            got.put(line)
        got.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    collected: list[str] = []
    deadline = time.monotonic() + deadline_s
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            line = got.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue
        if line is None:
            break
        text = line.rstrip("\n")
        collected.append(text)
        if on_line is not None:
            on_line(text)
    if timed_out:
        proc.kill()
    rc = proc.wait()
    if any(marker in ln for ln in collected for marker in AUTH_MARKERS):
        raise ColabError(
            "Colab CLI is not authenticated. Run this once in a terminal and "
            "follow the Google URL: " + str(base[0]) + " usage"
        )
    return rc, timed_out, collected


def _last_line(lines: list[str], fallback: str) -> str:
    for ln in reversed(lines):
        if ln.strip():
            return ln.strip()[:300]
    return fallback


def run_training(
    script: Path,
    *,
    gpu: str = "T4",
    artifact_remote: str = "/content/lora_adapter.zip",
    output_dir: Path,
    on_line: Optional[Callable[[str], None]] = None,
    cli: CliArg = None,
    provision_deadline_s: float = 300,
    exec_deadline_s: float = 2700,
    exec_silence_timeout_s: int = 600,
    download_deadline_s: float = 600,
    session: Optional[str] = None,
) -> TrainingOutcome:
    """Provision a Colab VM, run `script`, download `artifact_remote`, teardown.

    Always attempts `stop` in a finally block; raises ColabError only for
    actionable setup problems (missing CLI, missing script, auth needed).
    """
    base = resolve_cli(cli)
    script = Path(script)
    if not script.exists():
        raise ColabError(f"training script not found: {script}")
    session = session or f"st-{int(time.time())}"
    log = on_line or (lambda _ln: None)
    artifact: Optional[Path] = None
    all_lines: list[str] = []

    log(f"[colab] provisioning {gpu} session '{session}' ...")
    rc, to, lines = _stream(
        base, ["new", "-s", session, "--gpu", gpu], on_line, deadline_s=provision_deadline_s
    )
    all_lines.extend(lines)
    if to:
        _, _, stop_lines = _stream(base, ["stop", "-s", session], on_line, deadline_s=120)
        all_lines.extend(stop_lines)
        return TrainingOutcome(
            False, error="timed out provisioning the GPU session", logs=list(all_lines)
        )
    if rc != 0:
        return TrainingOutcome(
            False, error=_last_line(lines, f"colab new failed (exit {rc})"), logs=list(all_lines)
        )

    try:
        log("[colab] executing training script on the VM ...")
        rc, to, lines = _stream(
            base,
            [
                "exec",
                "-s",
                session,
                "-f",
                str(script),
                "--timeout",
                str(int(exec_silence_timeout_s)),
            ],
            on_line,
            deadline_s=exec_deadline_s,
        )
        all_lines.extend(lines)
        if to:
            return TrainingOutcome(
                False,
                error=f"training exceeded {int(exec_deadline_s)}s and was stopped",
                logs=list(all_lines),
            )
        if rc != 0:
            return TrainingOutcome(
                False,
                error=_last_line(lines, f"training script failed (exit {rc})"),
                logs=list(all_lines),
            )
        # `colab exec` exits 0 even when a notebook cell raises (kernel survives),
        # so remember the training tail: a later download miss needs that context.
        exec_tail = _last_line(lines, "")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"lora_adapter_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        log(f"[colab] downloading artifact to {artifact.name} ...")
        rc, to, lines = _stream(
            base,
            ["download", "-s", session, artifact_remote, str(artifact)],
            on_line,
            deadline_s=download_deadline_s,
        )
        all_lines.extend(lines)
        if to or rc != 0 or not artifact.exists():
            if artifact.exists():
                artifact.unlink()
            err = _last_line(lines, f"artifact download failed (exit {rc}) - no {artifact_remote}")
            if exec_tail:
                err = f"{err} (training tail: {exec_tail})"
            return TrainingOutcome(False, error=err, logs=list(all_lines))
        return TrainingOutcome(True, artifact=artifact, logs=list(all_lines))
    finally:
        log("[colab] releasing the VM ...")
        try:
            _, _, stop_lines = _stream(base, ["stop", "-s", session], on_line, deadline_s=120)
            all_lines.extend(stop_lines)
        except ColabError:
            pass  # teardown is best-effort; backend reclaims idle VMs anyway
