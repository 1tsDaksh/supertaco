"""10-goal sweep runner for WS4 receipts: plan -> run -> RESULTS.md.

Usage:
    python scripts/sweep.py --plans-only           # write the 10 plan yamls, no training
    python scripts/sweep.py [--limit N] [--force]   # unattended sweep (batch-approved)
    python scripts/sweep.py --report                # render RESULTS.md from artifacts

Each goal gets a stable id sw<4hex> derived from the goal text, exported as
SUPERTACO_RUN_ID so the plan and run subprocesses stamp the same identity into
logs/llm_calls.jsonl and logs/runs/<run_id>.json.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOALS_PATH = Path(__file__).resolve().parent / "goals.json"
ARTIFACT_DIR = REPO / "logs" / "runs"
REPORT_PATH = REPO / "RESULTS.md"
# Per-MTok USD rates: fill from Token Factory pricing when available; empty keeps
# RESULTS.md tokens-only (no fabricated dollars).
RATES_PER_MTOK: dict[str, float] = {}

PLAN_RE = re.compile(r"Plan written: (\S+)")


def load_goals() -> list[str]:
    """Read the sweep goal texts; the list order is the report order."""
    data = json.loads(GOALS_PATH.read_text(encoding="utf-8"))
    goals = [entry["goal"] for entry in data]
    ids = [run_id_for(g) for g in goals]
    assert len(set(ids)) == len(goals), "duplicate run_id in goals.json"
    return goals


def run_id_for(goal: str) -> str:
    """Stable per-goal id: sw + first 4 hex of sha1(goal) (resume maps to the same run)."""
    return "sw" + hashlib.sha1(goal.encode("utf-8")).hexdigest()[:4]


def artifact_path(run_id: str) -> Path:
    """Artifact location for one run id."""
    return ARTIFACT_DIR / f"{run_id}.json"


def load_artifact(run_id: str) -> dict | None:
    """Read one run artifact; missing OR corrupt counts as not-yet-run (re-run)."""
    path = artifact_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[{run_id}] artifact unreadable ({exc}); will re-run", file=sys.stderr)
        return None


def _write_timing(run_id: str, pair_start: float) -> None:
    """Persist the runner-measured wall clock of the plan+run pair (spec: runner measures)."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"pair_s": round(time.monotonic() - pair_start, 3)}
    (ARTIFACT_DIR / f"{run_id}.timing.json").write_text(json.dumps(payload), encoding="utf-8")


def _with_timing(artifacts: list[dict]) -> list[dict]:
    """Attach runner_wall_clock_s to each artifact when its timing file exists."""
    for artifact in artifacts:
        run_id = artifact.get("run_id")
        path = ARTIFACT_DIR / f"{run_id}.timing.json"
        if run_id and path.exists():
            try:
                timing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"[{run_id}] timing unreadable ({exc}); using artifact wall_clock_s",
                    file=sys.stderr,
                )
                continue
            if isinstance(timing.get("pair_s"), (int, float)):
                artifact["runner_wall_clock_s"] = timing["pair_s"]
    return artifacts


def _run(argv: list[str], run_id: str) -> subprocess.CompletedProcess:
    """Run one CLI subprocess with SUPERTACO_RUN_ID exported (monkeypatch seam)."""
    env = dict(os.environ, SUPERTACO_RUN_ID=run_id, PYTHONUTF8="1")
    return subprocess.run(
        argv,
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def plan_goal(goal: str, run_id: str) -> Path | None:
    """`supertaco plan` in a subprocess; return the written plan path (None on failure)."""
    proc = _run([sys.executable, "-m", "supertaco.cli", "plan", goal], run_id)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(f"[{run_id}] plan FAILED: {out.strip()}", file=sys.stderr)
        return None
    match = PLAN_RE.search(out)
    if match is None:
        print(f"[{run_id}] plan produced no path: {out.strip()}", file=sys.stderr)
        return None
    line = next((ln.strip() for ln in out.splitlines() if PLAN_RE.search(ln)), match.group(0))
    print(f"[{run_id}] {line}")  # echo mode=... so a fallback planner is visible
    return Path(match.group(1))


def run_goal(plan_path: Path, run_id: str) -> int:
    """`supertaco run` with Nebius-job judging; stdout is kept beside the artifact."""
    proc = _run(
        [
            sys.executable,
            "-m",
            "supertaco.cli",
            "run",
            str(plan_path),
            "--judge-backend",
            "nebius_job",
        ],
        run_id,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_DIR / f"{run_id}.stdout.log"
    text = proc.stdout or ""
    if proc.stderr:
        text += "\n--- stderr ---\n" + proc.stderr
    log_path.write_text(text, encoding="utf-8")
    if proc.returncode != 0:
        print(f"[{run_id}] run exited {proc.returncode}; see {log_path}", file=sys.stderr)
    return proc.returncode


def _human(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def _tokens_by_key(calls: list) -> dict[str, int]:
    totals: dict[str, int] = {}
    for call in calls:
        key = str(call.get("model_key") or "unknown")
        totals[key] = totals.get(key, 0) + int(call.get("tokens") or 0)
    return totals


def _usd(tokens: dict[str, int]) -> float | None:
    """Dollar estimate only when every key is priced; None keeps the table tokens-only."""
    if not RATES_PER_MTOK or any(key not in RATES_PER_MTOK for key in tokens):
        return None
    return sum(count * RATES_PER_MTOK[key] for key, count in tokens.items()) / 1_000_000


def _model_of(artifact: dict) -> str:
    for event in reversed(artifact.get("events") or []):
        if event.get("type") == "job_launched":
            return str(event.get("data", {}).get("payload", {}).get("model", "?"))
    return "?"


def _judge_cell(artifact: dict) -> str:
    ev = artifact.get("eval") or {}
    if not ev:
        return "-"
    modes = ev.get("modes") or []
    mode = modes[0] if modes else "?"
    suffix = " ..." if len(modes) > 1 else ""
    return f"{artifact.get('judge_backend') or 'direct'} · {mode}{suffix}"


def _wall(artifact: dict) -> float:
    """Runner-measured wall clock (whole plan+run pair) when present, else the loop's."""
    runner_wall = artifact.get("runner_wall_clock_s")
    if isinstance(runner_wall, (int, float)):
        return float(runner_wall)
    return float(artifact.get("wall_clock_s") or 0.0)


def render_report(artifacts: list[dict], goals: list[str]) -> str:
    """RESULTS.md: verify-by-run_id header, one row per artifact, totals, not-run list."""
    by_id = {a.get("run_id"): a for a in artifacts}
    ordered = [by_id[run_id_for(g)] for g in goals if run_id_for(g) in by_id]

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    success_count = sum(1 for a in ordered if a.get("success"))
    lines = [
        "# SuperTaco sweep results",
        "",
        f"**Date:** {today} - **judge backend:** nebius_job - "
        f"**runs recorded:** {len(ordered)} - **successful:** {success_count}",
        "",
        "**How to verify:** every row cites its `run_id` - "
        '`grep -c \'"run_id": "<run_id>"\' logs/llm_calls.jsonl` counts that run\'s audit '
        "lines (plan-phase calls share the id, so the count is >= the artifact's "
        "`llm_calls` length and > 0), and `logs/runs/<run_id>.json` holds its full "
        "artifact (schema v1).",
        "",
        "| run_id | goal | model | attempts | healed | base avg | ft avg | delta |"
        " judge | tokens | wall-clock |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals_by_key: dict[str, int] = {}
    total_attempts = 0
    total_wall = 0.0
    for a in ordered:
        tokens = _tokens_by_key(a.get("llm_calls") or [])
        for key, value in tokens.items():
            totals_by_key[key] = totals_by_key.get(key, 0) + value
        attempts = int(a.get("attempts") or 0)
        total_attempts += attempts
        total_wall += _wall(a)
        ev = a.get("eval") or {}
        base_avg = ev.get("base_avg")
        ft_avg = ev.get("ft_avg")
        delta = ev.get("delta")
        goal_text = str(a.get("goal") or "?").replace("|", "\\|")
        tokens_cell = (
            " · ".join(f"{k}:{_human(v)}" for k, v in sorted(tokens.items())) if tokens else "-"
        )
        lines.append(
            f"| `{a.get('run_id')}` | {goal_text} | {_model_of(a)} | {attempts} | "
            f"{'yes' if a.get('healed') else 'no'} | "
            f"{f'{base_avg:.2f}' if isinstance(base_avg, (int, float)) else '-'} | "
            f"{f'{ft_avg:.2f}' if isinstance(ft_avg, (int, float)) else '-'} | "
            f"{f'{delta:+.2f}' if isinstance(delta, (int, float)) else '-'} | "
            f"{_judge_cell(a)} | {tokens_cell} | {_wall(a):.0f}s |"
        )
    totals_cell = (
        " · ".join(f"{k}:{_human(v)}" for k, v in sorted(totals_by_key.items()))
        if totals_by_key
        else "-"
    )
    lines.append(
        f"| **totals** | {len(ordered)} runs | | {total_attempts} | {success_count} ok | | | | | "
        f"{totals_cell} | {total_wall:.0f}s |"
    )
    ran_ids = {a.get("run_id") for a in ordered}
    missing = [g for g in goals if run_id_for(g) not in ran_ids]
    if missing:
        lines += [
            "",
            f"**Not run:** {len(missing)} goal(s) - " + "; ".join(f'"{m}"' for m in missing),
        ]
    rates = _usd(totals_by_key)
    if rates is not None:
        lines += ["", f"**Estimated cost:** ${rates:.2f} (rates table filled)"]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: --report renders, --plans-only stops after plans, else sweeps."""
    parser = argparse.ArgumentParser(description="SuperTaco 10-goal sweep (WS4)")
    parser.add_argument("--plans-only", action="store_true", help="write plan yamls and stop")
    parser.add_argument("--report", action="store_true", help="render RESULTS.md and exit")
    parser.add_argument("--limit", type=int, default=None, help="run at most N pending goals")
    parser.add_argument("--force", action="store_true", help="re-run goals that already succeeded")
    args = parser.parse_args(argv)

    goals = load_goals()

    if args.report:
        loaded = [a for g in goals if (a := load_artifact(run_id_for(g))) is not None]
        REPORT_PATH.write_text(render_report(_with_timing(loaded), goals), encoding="utf-8")
        print(f"wrote {REPORT_PATH}")
        return 0

    if args.plans_only:
        failures = 0
        for goal in goals:
            if plan_goal(goal, run_id_for(goal)) is None:
                failures += 1
        print(f"plans written: {len(goals) - failures}/{len(goals)}")
        return 1 if failures else 0

    pending: list[str] = []
    for goal in goals:
        rid = run_id_for(goal)
        art = load_artifact(rid)
        if args.force or art is None or not art.get("success"):
            pending.append(goal)
        else:
            print(f"[{rid}] already successful - skip (use --force to rerun)")
    if args.limit is not None:
        pending = pending[: max(args.limit, 0)]

    failures = 0
    for goal in pending:
        rid = run_id_for(goal)
        print(f"[{rid}] planning: {goal}")
        pair_start = time.monotonic()
        plan_path = plan_goal(goal, rid)
        if plan_path is None:
            failures += 1
            continue
        print(f"[{rid}] running {plan_path}")
        run_rc = run_goal(plan_path, rid)
        _write_timing(rid, pair_start)
        if run_rc != 0:
            failures += 1

    loaded = [a for g in goals if (a := load_artifact(run_id_for(g))) is not None]
    REPORT_PATH.write_text(render_report(_with_timing(loaded), goals), encoding="utf-8")
    print(f"sweep done - failures: {failures}; report: {REPORT_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
