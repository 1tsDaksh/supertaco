# PROJECT.md — SuperTaco

## Summary (one paragraph)
SuperTaco is an AI-supervised fine-tuning sandbox: a user submits a dataset + goal, and an agent
writes the training config, launches the job on Nebius AI Cloud (Serverless Jobs), watches the logs
and metrics, diagnoses failures (NaN loss, OOM, divergence, eval regression), patches the config,
and retries — up to 3 times — until a checkpoint is produced, evaluated against a fixed prompt
suite, and deployed to a Serverless Endpoint for side-by-side before/after comparison. Built on
Nemotron models (via Nebius Token Factory) for the agent brain, Tavily for error-context lookup,
and LLaMA-Factory + LoRA for the actual training.

## Current state
**Phase 1 — manual pipeline** (in progress)

- [x] Repo scaffolded, license (Apache 2.0), .env template
- [x] LLaMA-Factory LoRA smoke config for a small model (single GPU, < 60 min)
- [ ] Manual training run end-to-end on Nebius AI Cloud via SDK
- [ ] Checkpoint → Serverless Endpoint deployment (manual)
- [ ] Before/after eval harness (5 fixed prompts + judge)
- [ ] **Gate: demo-able manual pipeline before starting the agent**

## Active task
**NOW:** Get the first real LoRA training job running on Nebius AI Cloud using the Python SDK,
verify logs stream to object storage, and confirm the endpoint deploys.

## Recent changes (newest first)
- 2026-09-28: runner/cli/dashboard: single shared `make_llm` factory through the runner seam; run_failed timeline shows attempts + last config path; stale events cleared on launch (final review fixes).
- 2026-09-28: Functional dashboard: shared event-emitting runner, config-derived dry-run logs, real Nemotron calls with fallback, real judge eval, CLI rewired.
- 2026-09-28: dashboard: clear stale eval results on launch; fixture loader hardened (BOM, YAMLError, non-dict); eval-gating + banner count pins (plan Task 9 review fixes).
- 2026-09-27: dashboard: fixture loader, Nemotron fallback banner, LLM call table, real judge eval suite, dry-run checkbox wired (plan Task 9).
- 2026-09-27: dashboard: clear stale run_result on launch; ConfigurationError re-renders timeline; validation/crash no-traceback pins (plan Task 8 review fixes).
- 2026-09-24: dashboard: live event-driven launch with timeline, log stream, loss chart (plan Task 8).
- 2026-09-24: cli: clean SystemExit messages for config/validation errors; failure-path pins (plan Task 7 review fixes).
- 2026-09-24: cli: run subcommand rewired to shared runner; F821 crashes resolved (plan Task 7).
- 2026-09-24: eval: numerator-first parse for X/10 judge replies; suite slice + hash-mode pins (plan Task 6 review fixes).
- 2026-09-24: eval: Nemotron judge with hash fallback + clamping; 5-prompt suite with regression flag (plan Task 6).
- 2026-09-24: runner: invariant/failure_key/gate test pins, uniform run_failed payload (plan Task 5 review fixes).
- 2026-09-24: runner: single event-emitting run path with healing loop, retry cap, config patching (plan Task 5).
- 2026-09-24: llm: JSONL write failure now warns; tests isolated from real audit log (plan Task 4 review fixes).
- 2026-09-24: llm: real Token Factory calls, circuit-breaker fallback, JSONL audit (plan Task 4).
- 2026-09-24: simlogs: healthy header carries config; fixture glob skips _patched; precedence + boundary tests pin rule table (plan Task 3 review fixes).
- 2026-09-24: simlogs: deterministic config-derived dry-run log generator with healing property (plan Task 3).
- 2026-09-24: Playbook eval-regression detector tightened to per-line match; tests hardened (plan Task 2 review).
- 2026-09-24: Playbook: EVAL_REGRESSION detects log text; ConfigurationError import fixed (plan Task 2).
- 2026-09-22: Wrote agent docs (AGENTS/PROJECT/STACK/CONVENTIONS/SPEC). Repo scaffolded.

## Up next (rough order)
1. Manual training job + endpoint deploy (this week)
2. Agent supervisor loop with 8-failure-mode playbook (LangGraph)
3. Streamlit dashboard: live logs, loss curves, retry timeline, eval panel
4. Tavily integration for unknown-error doc lookup
5. Demo video script + README polish + submission feedback write-up

## Constraints to remember
- Deadline: **2026-10-30**. Fully online hackathon.
- Budget: $50 Token Factory credits + $100 AI Cloud credits. Design for it.
- Must use >= 1 NVIDIA open-source model (Nemotron) and run on Nebius infra. Non-negotiable.
- Submission: public repo + open license + <= 3 min demo video with audio + written feedback on
  Nebius/NVIDIA tools (required — also competes for 10x $100 feedback awards).
