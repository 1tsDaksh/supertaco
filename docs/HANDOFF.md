# HANDOFF.md — SuperTaco

> Project handoff document. Read this fully before touching the codebase.
> Last updated: 2026-09-24 · Branch: `main` · Repo: https://github.com/1tsDaksh/supertaco

---

## 1. Project Overview

**SuperTaco** (renamed from *SandboxTune*) — an AI-supervised fine-tuning sandbox built for the
**Nebius x NVIDIA Global AI Hackathon** (deadline: 2026-10-30).

A user submits a dataset + goal; an agent writes the training config, launches the job on
Nebius AI Cloud (Serverless Jobs), monitors logs/metrics, diagnoses failures using an 8-mode
playbook, patches the config, and retries (max 3) until a checkpoint is produced, evaluated
against a fixed prompt suite (Nemotron judge), and deployed to a Serverless Endpoint for
before/after comparison.

**Stack:** Python 3.11 · event-driven runner (`supertaco.runner`) · Nemotron via Nebius Token Factory (nano → super →
ultra escalation) · Tavily (error-context search) · Streamlit (dashboard) · LLaMA-Factory + LoRA
(training, on Nebius AI Cloud).

---

## 2. Current Status

| Item | Status |
|---|---|
| Package install & imports | ✅ Operational (`pip install -e .`) |
| Dashboard | ✅ Live event-driven panels (dry-run pipeline + real Nemotron) |
| Agent loop (monitor → classify → patch → relaunch) | ✅ Working (event-driven runner) |
| 8 playbook failure modes | ✅ All tested |
| 3-retry cap (4th attempt → `run_failed`, error `MaxRetriesExceeded`) | ✅ Enforced |
| LLM call logging (model/tokens/latency) | ✅ Working |
| Dry-run mode (no GPU credits consumed) | ✅ Verified |
| Eval harness (Nemotron judge, 0–10, model comparison) | ✅ Working |
| 8 broken-config fixtures | ✅ Generated |
| Git working tree | ✅ Clean (local commits pending push) |

---

## 3. Quick Start

```bash
# 1. Clone & install
git clone https://github.com/1tsDaksh/supertaco.git && cd supertaco
pip install -e .

# 2. Configure environment
cp .env.example .env   # then fill in keys (see §6)

# 3. Run the dashboard
python -m streamlit run supertaco/ui/dashboard.py --server.headless true --server.port 8510
# → http://localhost:8510
```

Quality gates before committing:

```bash
pytest tests/ -m "not integration"   # unit tests (mocked API, no credits)
ruff check . && ruff format --check .
mypy supertaco/
```

---

## 4. Repository Structure

```
supertaco/                    # Python package
├── cli.py                    # `supertaco run <config> [--dry-run]`
├── settings.py               # pydantic-settings, .env loading
├── errors.py                 # JobFailedError, MaxRetriesExceeded
├── nebius/
│   ├── jobs.py               # Serverless Job SDK wrapper
│   ├── endpoints.py          # Serverless Endpoint deploy/undeploy
│   └── storage.py            # log/checkpoint object storage helpers
├── agent/
│   ├── playbook.py           # 8 failure modes: detection signals + typed fixes
│   ├── llm.py                # Nemotron client (nano→super→ultra escalation)
│   ├── graph.py              # supervisor nodes (legacy; runner drives the loop)
│   └── tools.py              # @tool functions (patch_config, launch_job, tavily_lookup)
├── eval/
│   ├── harness.py            # Fixed prompt suite runner
│   └── judges.py             # Nemotron-judge scoring
├── ui/
│   └── dashboard.py          # Streamlit frontend (event-driven, port 8510)
├── scripts/
│   └── make_broken_configs.py # 8 broken configs generator
├── configs/
│   ├── base/                 # Reference configs — NEVER MODIFY
│   └── runs/                 # 8 generated broken configs (.yaml)
├── docs/                     # HANDOFF, AGENTS, CONVENTIONS, SPEC, STACK, PROJECT
├── .env / .env.example
├── LICENSE                   # Apache 2.0
├── pyproject.toml            # name="supertaco"
└── README.md                 # Includes Nemotron/Token Factory/Jobs/Tavily sections
```

---

## 5. Architecture & Data Flow

```
Dashboard (Streamlit :8510)
  │ submit config / watch logs / agent timeline / before-after eval
  ▼
Agent loop (supertaco.runner, event-driven)
  monitor → classify (nemotron nano) → playbook patch (Nemotron displays reasoning) → relaunch (≤3)
  │   known failure → playbook fix · every LLM call logged (mode + fallback)
  ▼
Nebius AI Cloud ── Serverless Job (LLaMA-Factory + LoRA) → checkpoint
              └── Serverless Endpoint (serve fine-tuned model) → eval harness → judge
```

**Design invariants (do not break):**
1. Max 3 relaunches; a 4th attempt is not made — exhaustion emits `run_failed`
   (error `MaxRetriesExceeded`) and returns `success=False` instead of raising.
2. Every LLM call logged to `logs/llm_calls.jsonl` (model, mode=real|fallback, tokens, latency, error).
3. Every patch creates a NEW file in `configs/runs/` — configs are immutable.
4. Dry-run mode never makes a real Nebius network call (client wrapper may be
   constructed; `launch_job(dry_run=True)` is payload-only).

---

## 6. Environment Configuration

Settings load from `.env` (gitignored). Required variables:

| Variable | Purpose |
|---|---|
| `NEBIUS_API_KEY` | Token Factory + AI Cloud auth |
| `TAVILY_API_KEY` | Error-context search |
| `NEBIUS_PROJECT_ID` | Nebius project |
| `TOKEN_FACTORY_BASE_URL` | OpenAI-compatible endpoint (`https://api.token.factory.nvidia.com/v1`) |
| `SANDBOXTUNE_ENV` | `dev` \| `prod` (prod blocks dry-run bypasses) |

> **Note:** the `SANDBOXTUNE_ENV` variable name predates the rename and is kept for backward
> compatibility — do not rename without updating `settings.py` and `.env.example` together.

---

## 7. Git State

- URL: https://github.com/1tsDaksch/supertaco.git · Branch: `main` · Tree: clean

| Commit | Message |
|---|---|
| `1c77884` | rename: sandboxtune -> supertaco |
| `b1c06a5` | feat: initial SandboxTune repo |

---

## 8. ⚠️ Security Action Required — Key Rotation

Live API keys for **Nebius** and **Tavily** were exposed in plaintext outside the repository
(pasted in a chat/handoff). Before any further sharing or publication:

1. **Rotate both keys** in the Nebius console and Tavily dashboard.
2. Update `.env` locally; confirm `.env` remains gitignored and no key exists in git history:
   `git log -p -- .env` and `git grep -r "NEBIUS_API_KEY=" $(git rev-list --all)`.
3. Never paste `.env` contents into chats, tickets, or docs — share `.env.example` only.

---

## 9. How to Continue Development

**Immediate next steps (hackathon runway, ~5 weeks left):**

1. **Gate 1 — Real GPU run:** dry-run mode is verified, but no real training job has run on
   Nebius AI Cloud yet. Launch one LoRA smoke job, verify logs → checkpoint → endpoint → eval
   all work with real infrastructure.
2. **Agent loop on real jobs:** currently validated against fixtures; test full
   monitor → classify → patch → relaunch against a live NaN-loss and divergence run.
3. **Demo script + video (≤3 min):** rehearse the broken-config-heal live demo; record.
4. **Submission artifacts:** README polish (Nemotron/Token Factory usage callouts), required
   written feedback on Nebius/NVIDIA tools (also competes for 10 × $100 feedback awards),
   Devpost submission with city selected.

**Known gaps / not yet built:**
- No real Nebius AI Cloud job has run (dry-run + fixtures only).
- Dashboard consumes `supertaco.runner` (typed-event seam) plus `supertaco.eval`
  for the judge panel; it no longer duplicates pipeline logic.

**Rules for contributors:**
- Read `AGENTS.md`, `CONVENTIONS.md`, and `SPEC.md` before coding (if present in repo).
- Never modify `configs/base/` — fork into `configs/runs/`.
- One logical change per commit; conventional commits (`feat(agent): ...`).
- Cost-bearing actions (real jobs, endpoint deploys > 1 GPU-hour) require explicit approval.
