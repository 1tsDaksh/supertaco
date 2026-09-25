# STACK.md — Exact Versions & Dependencies

**Package manager: pip (Python). Prefer deps declared in pyproject.toml; avoid ad-hoc installs.**

## Language & runtime
- Python **3.11+** (pin: `>=3.11`)

## Core dependencies (pyproject.toml)
| Package | Version constraint | Why |
|---|---|---|
| langgraph | ^0.6 | Agent supervisor state machine |
| langchain-core | ^0.3 | Tool/chain plumbing (no langchain full package) |
| openai | ^1.5x | Client for Nebius Token Factory (OpenAI-compatible endpoint) |
| nebius | latest stable | Nebius AI Cloud SDK (jobs, endpoints, object storage) |
| pydantic | ^2.9 | Config + payload models |
| pydantic-settings | ^2.6 | Env-based settings |
| structlog | ^24.4 | Structured logging |
| streamlit | ^1.4x | Dashboard UI |
| datasets | ^3.x | HF dataset loading for eval suite |
| tavily-python | ^0.5 | Error-context search (prize category) |

## Model endpoints (Nebius Token Factory, OpenAI-compatible)
| Role | Model ID | Use |
|---|---|---|
| Fast classify | `nvidia/nemotron-3-nano` | Log triage, failure classification |
| Patch/draft | `nvidia/nemotron-3-super` | Config patches, fix proposals |
| Deep reason | `nvidia/nemotron-3-ultra` | Low-confidence or novel failures ONLY |

Cost routing rule: start at nano; escalate to super on low confidence; ultra only when playbook
misses. Log every escalation.

## Training stack (runs on Nebius AI Cloud, NOT in this repo's deps)
- LLaMA-Factory (pinned via container image tag, not pip) + LoRA
- Single GPU per run (L40S/H100), target < 60 min per run
- Base model: small open model, e.g. Llama-3.1-8B or Nemotron variant

## Tooling versions
- ruff ^0.8 (lint + format)
- mypy ^1.13 (strict mode on `supertaco/`)
- pytest ^8.3 + pytest-asyncio ^0.24

## Environment variables (.env, gitignored)
```
NEBIUS_API_KEY=            # AI Cloud + Token Factory
NEBIUS_PROJECT_ID=
TAVILY_API_KEY=
TOKEN_FACTORY_BASE_URL=    # OpenAI-compatible base URL
SANDBOXTUNE_ENV=dev        # dev | prod — prod blocks dry-run-only bypasses
```

## Deprecated / do-not-use list
- Do not use `langchain` (monolith) — only `langchain-core`.
- Do not use `transformers` Trainer directly — LLaMA-Factory only.
- Do not use `requests` — use `httpx` if needed.
- Do not use conda or poetry anywhere in the repo.
