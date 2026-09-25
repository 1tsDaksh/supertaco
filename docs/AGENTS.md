# AGENTS.md — Core Behavioral Rules

> Read this file at the start of every session. It defines how you work, not just what you build.

## Mission
You are the AI coding agent for **SuperTaco** — an AI-supervised fine-tuning sandbox built for the
Nebius x NVIDIA Global AI Hackathon. Your job is to build, test, and ship code that lets an agent
launch, monitor, diagnose, and repair LLM fine-tuning jobs running on Nebius AI Cloud.

## Non-negotiable rules

1. **Never commit secrets.** API keys, Nebius tokens, and Tavily keys live in `.env` (gitignored) only.
   If you need a value, read it from `os.environ` / `pydantic-settings`.
2. **Never run a real Nebius job without a `--dry-run` flag available.** Real GPU runs cost money
   ($100 AI Cloud credit budget). Every job launcher must support dry-run mode that logs the exact
   payload it *would* send.
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
python -m ruff format --check .
python -m mypy supertaco/

# Unit tests (mocked Nebius API — no credits spent)
python -m pytest tests/ -m "not integration"

# Integration test (hits real Nebius API — costs money, needs explicit flag)
python -m pytest tests/ -m integration

# Dry-run the agent loop on a deliberately broken config (free)
python -m supertaco.cli run configs/runs/20260923_191506_NAN_LOSS_broken_model.yaml --dry-run

# Full smoke test (real job, single GPU, < 60 min, requires approval)
python -m supertaco.cli run configs/runs/20260923_191506_NAN_LOSS_broken_model.yaml
```

## Style guardrails

- Python 3.11+, strict type hints, `from __future__ import annotations` not needed on 3.11+.
- All public functions have docstrings (Google style). Private helpers docstring only if non-obvious.
- Pydantic v2 models for all config and API payloads. No raw dicts crossing module boundaries.
- Structured logging via `structlog` — never `print()` in library code.
- Errors: raise typed exceptions (`supertaco.errors.JobFailedError`) — no bare `except`, no swallowed exceptions.
- Agent-facing code (LangGraph nodes) must be deterministic given a fixed seed; log every LLM call
  (model, tokens, latency) to `logs/llm_calls.jsonl`.

## Cost guardrails (hackathon credits are finite)

- Default to the cheapest model that can do the job: `nvidia/nemotron-3-nano` for classification,
  `nvidia/nemotron-3-super` for patching, `nvidia/nemotron-3-ultra` ONLY when confidence < threshold
  or playbook has no matching failure mode.
- Every loop iteration (launch → diagnose → relaunch) is capped at **3 retries**, then hard stop
  with a user-facing summary.
- Long training runs (> 1 GPU-hour) require explicit human approval — log the approval in the run record.
