# SPEC.md — Functional Requirements & User Stories

## Product summary
An AI-supervised fine-tuning sandbox. User supplies (dataset, base model, goal). The agent writes
the config, launches training on Nebius AI Cloud, monitors it, auto-heals failures using a
data-driven playbook, then evaluates and deploys the checkpoint — showing before/after results.

## User stories & requirements

### US-1: Submit a fine-tune job (MVP)
- As a user, I paste a dataset (or pick a demo dataset) and describe the goal in plain language,
  so I don't need to know hyperparameters.
- AC: UI generates an initial training config; user can inspect/edit it before launching.
- AC: Launch creates a Nebius Serverless Job; job ID and live status appear in the dashboard.

### US-2: Live monitoring (MVP)
- As a user, I watch logs, loss curves, and GPU metrics stream in real time, so I trust what's happening.
- AC: Dashboard shows loss curve + learning rate + GPU utilization per step.
- AC: Each agent decision (classify → patch → relaunch) appears as a timeline entry with reasoning.

### US-3: Automatic failure diagnosis & repair (MVP — the differentiator)
- As a user, I submit a deliberately broken config and watch the agent catch, diagnose, fix, and
  retry it, so I see the supervision loop working.
- AC: Playbook covers 8 failure modes (see below) with typed detection signals.
- AC: Novel/unknown errors trigger a Tavily doc lookup; the fetched context is shown and logged.
- AC: Hard cap of 3 retries, then a clear user-facing failure summary (what failed, what was tried).
- AC: Every patch creates a new config file; full audit trail preserved.

### US-4: Evaluation & before/after comparison (MVP)
- As a user, I see the base model and fine-tuned model answer the same fixed prompt suite,
  side by side, with judge scores, so I know the fine-tune actually helped.
- AC: 5–10 fixed prompts per demo dataset; Nemotron judge scores both models.
- AC: Regression vs. baseline is flagged loudly (eval regression failure mode).

### US-5: Deployment (MVP)
- As a user, I get a callable endpoint for my fine-tuned model (Serverless Endpoint), so the
  result is usable, not just a checkpoint file.
- AC: Deploy button + auto-undeploy to conserve credits.

### US-6 (stretch): Natural-language iteration
- "Make it follow instructions more strictly" → agent translates to concrete config deltas
  (e.g., raise LoRA rank, adjust epochs) and re-runs.

## Playbook v1 — 8 failure modes
| Key | Detection signal | Default fix |
|---|---|---|
| NAN_LOSS | loss == NaN in first N steps | lower LR 10x; check data preprocessing |
| OOM | CUDA OOM in traceback | halve batch size; enable gradient checkpointing |
| LOSS_DIVERGENCE | loss increases > threshold over window | lower LR; add warmup; clip grads |
| LOSS_PLATEAU | no improvement over K evals | raise LR slightly or adjust schedule; check data mix |
| EVAL_REGRESSION | post-train eval < baseline | lower LoRA rank/alpha; fewer epochs; stronger regularization |
| TOKENIZER_MISMATCH | template/tokenizer errors in logs | fix chat template; verify dataset formatting |
| DATALOADER_STALL | no step progress in T minutes | reduce num_workers / check corrupted shards |
| GRADIENT_EXPLOSION | grad_norm spikes | clip grad norm; lower LR |

## Out of scope (explicitly NOT building)
- Multi-user auth, billing, multi-GPU/distributed training
- Full fine-tuning (LoRA only)
- Arbitrary base-model support (one base model for the demo)
- Real container-security hardening (Token Factory Sandboxes are the isolation story)

## Technical requirements
- Everything runs on Nebius Token Factory / Nebius AI Cloud; >= 1 NVIDIA open-source model (Nemotron).
- All job launches support `--dry-run`.
- Submission artifacts: public repo (Apache 2.0), <= 3 min demo video, written Nebius/NVIDIA feedback.
