# AGENTS.md — Core Behavioral Rules

> Read this file at the start of every session. It defines how you work, not just what you build.

## Mission
You are the AI coding agent for **SuperTaco** — an AI-supervised fine-tuning sandbox built for the
Nebius x NVIDIA Global AI Hackathon. Your job is to build, test, and ship code that lets an agent
launch, monitor, diagnose, and repair LLM fine-tuning jobs running on Nebius AI Cloud.

## Non-negotiable rules

1. **Never commit secrets.** API keys, Nebius tokens, and Tavily keys live in `.env` (gitignored) only.
   If you need a value, read it from `os.environ` / `pydantic-settings`.
2. **Never spend real resources without asking.** Colab T4 runs are free but take ~5-15 min each;
   Token Factory judge calls are the only metered resource. Tests fake the transport and the LLM;
   only a documented, user-approved live smoke touches either.
3. **Never modify `configs/base/`**. These are reference configs. Create variants in `configs/runs/`.
4. **Test before you claim.** If you say something works, you ran the test. No "this should work."
5. **Small diffs.** One logical change per edit. No drive-by refactors of files you were not asked to touch.
6. **If a task is ambiguous, ask — do not guess.** Especially for cost-bearing actions
   (launching jobs, deploying endpoints).

## Workflow for every task

1. Read `PROJECT.md` for current state and active task.
2. Check `STACK.md` and `CONVENTIONS.md` before writing any code.
3. Make the change.
4. Run the relevant tests / lint (commands below).
5. Update `PROJECT.md` (what changed, what's next).

## Testing & validation commands

```bash
# Setup (first time)
pip install -e .
pip install --group dev

# Lint + typecheck — run before every commit
python -m ruff check .
python -m ruff format --check supertaco tests
python -m mypy supertaco/

# Unit tests (transport and LLM faked — no network, no credits spent)
python -m pytest tests/ -q

# Real supervisor loop on the default T4 config (free GPU, ~15 min, spends judge tokens - ask first)
python -m supertaco.cli run configs/defaults/colab_t4.yaml
```

## Style guardrails

- Python 3.11+, strict type hints, `from __future__ import annotations` not needed on 3.11+.
- All public functions have docstrings (Google style). Private helpers docstring only if non-obvious.
- Pydantic v2 models for all config and API payloads. No raw dicts crossing module boundaries.
- Structured logging via `structlog` — never `print()` in library code.
- Errors: raise typed exceptions (`supertaco.errors.JobFailedError`) — no bare `except`, no swallowed exceptions.
- Supervisor-loop code must be deterministic given a fixed seed; log every LLM call
  (model, tokens, latency) to `logs/llm_calls.jsonl`.

## Cost guardrails (hackathon credits are finite)

- Default to the cheapest model that can do the job: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` for classification,
  `nvidia/nemotron-3-super-120b-a12b` for patching, `nvidia/Nemotron-3-Ultra-550b-a55b` ONLY when confidence < threshold
  or playbook has no matching failure mode.
- Every loop iteration (launch → diagnose → relaunch) is capped at **3 retries**, then hard stop
  with a user-facing summary.
- Long training runs (> 1 GPU-hour) require explicit human approval — log the approval in the run record.
