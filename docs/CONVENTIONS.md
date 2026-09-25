# CONVENTIONS.md — Structure, Naming, Formatting

## Folder structure
```
sandboxtune/               # src package
  cli.py                   # entry point: `sandboxtune run <config> [--dry-run]`
  settings.py              # pydantic-settings, env loading
  errors.py                # typed exceptions
  nebius/
    jobs.py                # Serverless Job launch/stop/status (SDK wrapper)
    endpoints.py           # Serverless Endpoint deploy/undeploy
    storage.py             # log/checkpoint object storage helpers
  agent/
    graph.py               # LangGraph supervisor state machine
    playbook.py            # 8 failure modes: detection signals + fixes (typed, data-driven)
    llm.py                 # Nemotron client + cost-routing escalation logic
    tools.py               # @tool functions (patch_config, launch_job, tavily_lookup)
  eval/
    harness.py             # fixed prompt suite runner + judge
    judges.py              # Nemotron-judge scoring
  ui/
    dashboard.py           # Streamlit app
configs/
  base/                    # REFERENCE CONFIGS — NEVER MODIFY
  runs/                    # generated/run configs (gitignored except examples)
tests/
  fixtures/                # real (sanitized) log snippets per failure mode
scripts/
  make_broken_configs.py   # generates the broken configs for demos/tests
logs/                      # gitignored; llm_calls.jsonl lives here
docs/                      # hackathon notes, feedback draft, demo script
```

## Naming rules
- Modules & functions: `snake_case`. Classes: `PascalCase`. Constants: `UPPER_SNAKE`.
- Config files: `runs/<timestamp>_<slug>.yaml` — never overwrite; new file per attempt.
- Agent nodes in the LangGraph: verb phrases — `classify_failure`, `propose_patch`, `launch_job`.
- Failure modes in the playbook: `UPPER_SNAKE` keys, e.g. `NAN_LOSS`, `OOM`, `LOSS_DIVERGENCE`,
  `LOSS_PLATEAU`, `EVAL_REGRESSION`, `TOKENIZER_MISMATCH`, `DATALOADER_STALL`, `GRADIENT_EXPLOSION`.

## Formatting
- ruff defaults, line length **100**.
- One blank line between logically distinct blocks; two between top-level defs.
- YAML configs: 2-space indent, alphabetized keys, comments explaining *why*, not *what*.

## Never modify
- `configs/base/` — reference configs. Fork into `configs/runs/`.
- `tests/fixtures/` — append new fixtures; never edit existing ones (they pin agent behavior).
- `LICENSE` and the hackathon-required sections of `README.md`
  (license callout, Nemotron/Token Factory usage section).

## Git discipline
- Conventional commits: `feat(agent): add OOM failure mode`, `fix(eval): judge timeout`.
- No commit may contain real API keys, run outputs > 10 MB, or `configs/runs/` (except `*_example.yaml`).
- Every PR/commit message ends with what was tested and how (one line).

## Agent-loop invariants (tests enforce these)
1. Max 3 relaunches per run record; 4th attempt must raise `MaxRetriesExceeded`.
2. Every LLM call writes one line to `logs/llm_calls.jsonl` (model, tokens, latency, escalation reason).
3. Every config patch produces a NEW file in `configs/runs/`; the previous file is never edited.
4. Dry-run mode never instantiates a real Nebius client.
