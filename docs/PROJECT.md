# PROJECT.md — SandboxTune

## Summary (one paragraph)
SandboxTune is an AI-supervised fine-tuning sandbox: a user submits a dataset + goal, and an agent
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
