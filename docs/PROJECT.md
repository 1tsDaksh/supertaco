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

- 2026-09-29: playbook: case-insensitive torch-style nan detection + real healthy-log fixture; loop LOSS_RE case-insensitivity (plan Task 10 + booked fixes).
- 2026-09-29: colab template: model.eval() before eval generation; strengthen phase-2 pins (plan Task 9 review fixes).
- 2026-09-29: colab template: on-VM base/ft answer generation + RESPONSES_JSON block (plan Task 9).
- 2026-09-29: tests: pin failure-path and per-stream log passthrough in adapter (plan Task 8 review fixes).
- 2026-09-29: transport: ColabTransport (render -> run_training -> AttemptResult) + adapter log passthrough (plan Task 8).
- 2026-09-29: transport: reject config/prompt values containing template markers; pin rendered-script validity (plan Task 7 review fixes).
- 2026-09-29: transport: marker regions in colab template + deterministic render_script (plan Task 7).
- 2026-09-29: tests: transport whitelist unit tests; trim redundant loop validation test (plan Task 6 corrected).
- 2026-09-29: tests: pin pre-launch model validation (unknown / T4-blocked / missing-key honest failures) (plan Task 6).
- 2026-09-29: tests: pin artifact propagation and run_failed payload keys (plan Task 5 review fixes).
- 2026-09-29: tests: pin EVAL_REGRESSION heal within shared relaunch budget + honest missing-block evidence (plan Task 5 + review fixes).
- 2026-09-29: tests: pin relaunch config delivery, patch_proposed event, failure loss points; FakeLLM patch logging (plan Task 4 review fixes).
- 2026-09-29: loop/tests: pin nan-patch-retry + budget ledger, evidence paths, classify mode; harden error-path loss extraction (plan Task 4 + review fixes).
- 2026-09-29: loop: never-raise guards (non-dict config, negative retries), ledger/eval evidence on fail-fast paths, responses KeyError guard (plan Task 3 review fixes).
- 2026-09-29: loop: run_training_loop - attempts, events, eval gate, fail-fast infra errors; T4 model whitelist pre-launch (plan Task 3).
- 2026-09-29: loop: module skeleton - Event/LoopResult/AttemptResult, moved validate/_write/extract helpers (plan Task 2).
- 2026-09-29: eval: split_responses hardening - catch RecursionError, cover non-string payload values (plan Task 1 review fixes).
- 2026-09-29: eval: split_responses marker parser for on-VM response blocks (plan Task 1).
- 2026-09-29: docs: add colab-real-loop implementation plan - 14 tasks (split_responses parser, loop module, transport seam, VM phase 2, dashboard/CLI rework, simulation+Nebius deletions, live smoke approval gate).
- 2026-09-29: spec: colab-real-loop design approved (dashboard config -> Colab T4 attempts -> Token Factory supervisor/judge -> attempt ledger report; simulation + Nebius stubs to be removed; supersedes 2026-09-28 nebius autoheal spec).
- 2026-09-29: FIRST FULL LIVE COLAB RUN SUCCEEDED (run #4, session st-1790695167): T4 provisioned, LoRA trained 32 steps (loss 1.92->1.32, SuperTaco step/loss lines streamed), lora_adapter zip (6.2 MB, adapter_model.safetensors+tokenizer) downloaded to colab/output/, VM released; run #3 provisioning timed out transiently (colab assign API read timeout) - retry worked.
- 2026-09-29: colab script fix: remove_unused_columns=False so the chat-template collator keeps the `messages` column (Trainer signature-stripping raised ValueError in run #2); plus torchao fix from run #1.
- 2026-09-29: colab live run #1 reached real T4 but died in get_peft_model (Colab ships torchao 0.10.0, peft requires >0.16 and raises): ensure_packages now removes stale torchao; adapter error on download-miss now includes the training output tail (exec exits 0 even when a cell raises).
- 2026-09-29: dashboard: real Colab T4 training panel — supertaco/gpu/colab.py adapter (new->exec->download->stop, streamed loss lines, auth/setup hints, termios shim) + colab/train_lora.py script + AppTests; colab/README rewritten for CLI flow; kaggle/ removed (dropped per user).
- 2026-09-28: switch free-GPU smoke run to Google Colab (colab/train_lora notebook, T4, loss lines + adapter zip download): Kaggle path blocked at account level (sessions have no internet despite enable_internet=true; diagnosed via diag kernel).
- 2026-09-28: add free Kaggle GPU smoke-run path: kaggle/train_lora notebook (Qwen2.5-0.5B LoRA on 256 alpaca rows, SuperTaco loss-line format, adapter to /kaggle/working) + push/run docs; kernel zephyr0706/supertaco-lora-smoke-run pushed.
- 2026-09-28: fix eval regression signal: handcrafted per-prompt mock answers (base mediocre / healthy expert / unhealthy rambling) replace anemic 2-line templates - real Nemotron judge now separates tiers (healthy ft 9.6 vs base 6.8, flag OFF; unhealthy ft 1.2, flag ON, verified live on Token Factory).
- 2026-09-28: access: milestone 0 done — jobs list 200 via WSL CLI profile (federation, project-e00w64x9pr00th10a6x8sv); Sky DNS hijack (90.207.238.183) fixed in Windows+WSL hosts; spec §1/§8 updated (real-gpu-autoheal plan Task 1).
- 2026-09-28: fix eval judge hash-fallback: `max_tokens` 16→512 (Nemotron-3 reasoning ate the whole budget → empty content → unparseable → hash); live suite all `real`, dashboard warning gone.
- 2026-09-28: fix Token Factory base URL (old host was NXDOMAIN) and real Nemotron model IDs from the TF catalog; live API check verified `mode=real` (230 tokens, verdict OOM).
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
