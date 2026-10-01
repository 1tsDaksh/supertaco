# STACK.md — Exact Versions & Dependencies

**Package manager: pip (Python). Prefer deps declared in pyproject.toml; avoid ad-hoc installs.**

## Language & runtime
- Python **3.11+** (pin: `>=3.11`)

## Core dependencies (pyproject.toml)
| Package | Version constraint | Why |
|---|---|---|
| openai | >=1.5 | Client for Nebius Token Factory (OpenAI-compatible endpoint) |
| pydantic | >=2.9 | Config + payload models |
| pydantic-settings | >=2.6 | Env-based settings |
| structlog | >=24.4 | Structured logging |
| streamlit | >=1.4 | Dashboard UI |
| datasets | >=3.0 | HF dataset loading for eval suite |
| plotly | >=5.0 | Dashboard charts |
| pyyaml | >=6.0 | YAML config parsing (dashboard) |

## Model endpoints (Nebius Token Factory, OpenAI-compatible)
| Role | Model ID | Use |
|---|---|---|
| Fast classify | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Log triage, failure classification |
| Patch/draft | `nvidia/nemotron-3-super-120b-a12b` | Config patches, fix proposals |
| Deep reason | `nvidia/Nemotron-3-Ultra-550b-a55b` | Low-confidence or novel failures ONLY |

Cost routing rule: start at nano; escalate to super on low confidence; ultra only when playbook
misses. Log every escalation.

## Training stack (runs on a Google Colab free T4 via `colab/train_lora.py`, NOT in this repo's deps)
- PEFT/LoRA via `peft` + `transformers` `Trainer` (installed at runtime inside the Colab VM)
- Single T4 GPU per run (free tier), target < 60 min per run
- Base model: small open model, e.g. `Qwen/Qwen2.5-0.5B-Instruct` or a Nemotron variant

## Tooling versions
- ruff ^0.8 (lint + format)
- mypy ^1.13 (on `supertaco/`)
- pytest ^8.3 + pytest-asyncio ^0.24

## Environment variables (.env, gitignored)
```
NEBIUS_API_KEY=            # AI Cloud + Token Factory
NEBIUS_PROJECT_ID=
TOKEN_FACTORY_BASE_URL=    # OpenAI-compatible base URL
SANDBOXTUNE_ENV=dev        # dev | prod
```

## Deprecated / do-not-use list
- Do not use `langchain` (monolith) or `langchain-core` — neither is a dependency.
- Do not use LLaMA-Factory — training runs in the Colab VM via `colab/train_lora.py` (PEFT +
  transformers, installed at runtime, not repo deps).
- Do not use `requests` or `httpx` — neither is a dependency; the `openai` client covers HTTP.
- Do not use conda or poetry anywhere in the repo.
