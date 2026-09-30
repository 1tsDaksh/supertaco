> **SUPERSEDED (2026-09-29):** This Nebius autoheal design was never executed; the approved
> replacement is `docs/superpowers/specs/2026-09-29-colab-real-loop-design.md`.

# Real GPU Auto-Heal Loop — Design Spec

> Date: 2026-09-28 · Status: approved (design sections approved in chat; pending written-spec review)
> Scope: replace the dummy eval + flag-only behavior with real GPU training, real judge eval,
> and automatic config healing on Nebius AI Cloud Serverless Jobs (HANDOFF Gate 1, minus endpoints)

---

## 1. Problem

The functional-dashboard spec (2026-09-24) made every panel live but kept two layers fake:

| Layer | Today |
|---|---|
| Eval responses | `build_responses` fabricates template strings ("Fine-tuned answer to '...'"); the judge scores dummy text |
| Regression flag | Only renders a red banner; nothing feeds `regression_flagged` back into the heal loop (spec 5.4 put eval outside the runner, post-success, button-triggered) |
| Dry-run flag correctness | The strict judge floors both template answers at 0–1 (live evidence: base avg 0.2, ft avg 0.0 on a **healthy** config), so `avg_ft < avg_base × 0.9` flags **every successful run** — a false positive on the happy path |
| Real Nebius jobs | `NebiusJobClient.launch_job(dry_run=False)` raises `NotImplementedError`; `get_status`/`stop_job` too; no real job has ever run (HANDOFF Gate 1) |

User direction (2026-09-28): "our point is to fix the problem not flag them", and "we need to
actually working no dummy or no fakes".

### Verified access facts (probes run 2026-09-28)

- **Jobs access (verified 2026-09-28, milestone 0 complete):** the jobs-capable credential is
  the **user federation profile** (`nebius` CLI profile `default`, Google login, tenant
  `tenant-e00v26mtwqtan81xc4`, parent project `project-e00w64x9pr00th10a6x8sv` =
  `default-project-eu-north1`). `nebius ai job list` returns HTTP 200/empty — prerequisite P1
  satisfied. The `NEBIUS_API_KEY` AI Studio key (SA `sa-api-key-aiproject-…`, project
  `aiproject-e00b52k8gpcc7zs1me`) still has **no jobs rights** (403) and is NOT used for jobs;
  it remains the Token Factory/judge credential.
- **DNS hijack (verified + fixed 2026-09-28):** Sky broadband poisons `*.nebius.cloud` (and
  `kali.org`) to `90.207.238.183` (Sky block server — proven by apt errors referencing
  `block.isp.sky.com`). Fix is permanent: entries in Windows `hosts` AND WSL `/etc/hosts`
  (with `generateHosts = false`) covering `api.`, `api.eu-north1.`, `storage.eu-north1.`,
  `auth.nebius.com`, `apps.msp.api.`, `cpl.iam.api.` + six more route names (all →
  `91.210.70.243`). Prerequisite P2 satisfied; preflight (§5.1) re-checks every run.
- The **CLI-subprocess transport** (`wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius …`)
  is the chosen real-jobs transport (§5.1): only credential path with jobs permission, and its
  `ai job create/get/get-by-name/logs/cancel` verbs pin the request contract without guessing
  REST field names.

## 2. Decisions (clarifiers resolved with user)

| # | Question | Decision |
|---|---|---|
| D1 | What happens on eval regression | **Auto-heal loop**: regression is fed back into the runner as `failure_key=EVAL_REGRESSION` → patch → retrain, within the existing initial + ≤3 relaunch budget; a banner only appears when the budget is exhausted |
| D2 | Fidelity | **Real only** — no dummy responses, no simulated tiers, no injected fake evaluators in the real path ("no dummy or no fakes") |
| D3 | Packaging | **One spec** covering the full real loop (jobs API + training + eval + auto-heal), executed milestone-wise in one plan |
| D4 | Endpoints | **Out of scope** this effort; checkpoint → endpoint deployment stays a later Gate-1 task |
| D5 | Architecture | **Approach 1**: one monolithic train+eval Serverless Job per attempt; base/ft responses emitted as a delimited JSON block in job logs (no File Storage/IAM plumbing) |
| D6 | Dry-run regression signal | **Config-health ground truth** (`classify_config_failure(final_config) is not None` ⇒ flagged) — kills the false positive; dry-run eval panel stays report-only, auto-eval-loop lives only in real mode |
| D7 | Training stack | Qwen2.5-1.5B-Instruct (non-gated, LoRA-ready, fast) · `yahma/alpaca-cleaned` slice · `max_steps ≈ 60` · L40S preset — ≈10–20 min/attempt, ≤4 attempts, ~$5–15 per full demo run against the $100 credit budget |
| D8 | CLI/dashboard mode | Simulation stays the **default**; `--real` CLI flag and a **Simulation / Real GPU** sidebar toggle opt in. `--dry-run` flag always available (AGENTS rule 2); cost-bearing runs ask first (rule 6). The default stays simulation despite AGENTS examples implying bare `run` = real, because rule 6 outranks example style |
| D9 | Real-mode fallback policy | **Never silently fall back to heuristics**: infra failures (403/DNS/API) → `run_failed` with an actionable message. (The judge's hash fallback banner from the previous spec is unchanged and stays visible.) |

## 3. Out of scope

- Endpoint deployment / side-by-side serving comparison (D4)
- Tavily real API (remains mocked)
- Real-log coverage for **all 8** failure keys — real mode guarantees detection for the
  classes reproducible on GPU in a smoke run (**OOM, NaN/divergence, eval regression**);
  remaining keys stay dry-run-calibrated and are extended opportunistically during the
  calibration milestone (§7 fixture capture)
- Repo-wide lint/mypy debt (unchanged policy: touched files clean)
- Changing simulation-mode behavior beyond the D6 flag rule (fixtures, generator, healing
  property all stay as-is and green)

## 4. Architecture (Approach 1)

```
Dashboard [Simulation | Real GPU]         CLI (supertaco run <cfg> [--real | --dry-run])
        │  config dict + mode flag                │  config path + mode flag
        └──────────────────┬──────────────────────┘
                           ▼
                 supertaco/runner.py — ONE run path, same 10-event catalog
                           │
              ┌────────────┴─────────────┐
              ▼ dry_run=True (default)   ▼ dry_run=False (real)
     sim log generator (unchanged)   NebiusJobClient (REAL)
                                     launch → poll status → stream logs
                                     real-log detect → patch → relaunch
                                     job emits ###RESPONSES_JSON### … ###END_RESPONSES_JSON###
                                     → judge (Token Factory, real)
                                       → regression? EVAL_REGRESSION patch → relaunch
```

### Run semantics — real mode (one "Launch Job" click with Real GPU on)

1. **Preflight**: DNS resolves `api.nebius.cloud` to a serving IP, key authenticates, jobs
   permission probe passes. Any failure → `run_failed` with an actionable message; no launch.
2. Launch: config → container spec (image, command, env, L40S preset, timeout) →
   `job_launched` event (payload = real request body, job ID).
3. Poll: status transitions (provisioning → running → succeeded/failed) with logs streamed;
   `logs_produced` events carry real log text + parsed loss points (loss chart works).
4. `detect_failure(real_logs)` via playbook extended with real phrasings; on failure the
   existing classify → propose → `patch_written` (new YAML in `configs/runs/`, invariant 3)
   → relaunch cycle runs with **real** jobs, within the same initial + ≤3 budget.
5. Clean logs ⇒ the job's trailing phase has merged the LoRA adapter and generated answers
   for the 5 fixed prompts with **base** and **fine-tuned** weights, emitted in logs as:
   `###RESPONSES_JSON###` / one JSON object `{"base": {prompt: answer}, "fine_tuned": {...}}`
   / `###END_RESPONSES_JSON###`. Runner extracts it.
6. Eval gate: `run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=judge_client)` — **real** judge.
   Regression (`avg_ft < avg_base × 0.9`) ⇒ treated as a failed attempt:
   `failure_detected(EVAL_REGRESSION)` → patch (halve lora_r/alpha, −1 epoch) → relaunch.
   Eval-regression relaunches **consume the same shared relaunch budget** as log-failure
   relaunches (initial + ≤3 total, D1).
   Pass ⇒ `run_succeeded`; `RunResult.eval_results` populated.
7. Budget exhausted ⇒ `run_failed` (never raises; banner only here — "fix, not flag" D1).

### Run semantics — dry-run mode (unchanged except D6)

- Simulation loop exactly as today (spec 2026-09-24 §4), including the manual eval button.
- The eval panel's regression flag becomes `classify_config_failure(final_config) is not
  None` (ground truth) instead of the judge-ratio rule — healthy runs can no longer be
  falsely flagged; judge scores remain displayed as report-only data.

### Event catalog

Unchanged — the as-built 10-type table in the 2026-09-24 spec §4 remains valid (real mode
reuses `job_launched`, `logs_produced`, `failure_detected(EVAL_REGRESSION)`, `patch_written`,
`retry_scheduled`, `run_succeeded`, `run_failed`). `RunResult` gains `eval_results: dict | None`.

## 5. Components

### 5.1 `supertaco/nebius/jobs.py` — real client (replaces `NotImplementedError`)

- `launch_job(config, dry_run=False)` — build the Serverless Jobs create request
  (`parent_id=project`, container image/command, env, GPU preset, timeout); `dry_run=True`
  returns the payload with zero network (AGENTS rule 2 invariant kept).
- `get_status(job_id)`, `get_logs(job_id)`, `stop_job(job_id)` — real implementations.
- Transport: `nebius` SDK 0.6.14 (already installed) preferred; httpx-REST fallback if the
  SDK rejects API-key auth. The exact request contract is pinned in the milestone-1 live
  probe task — the spec deliberately does not guess field names.
- Pydantic v2 models for request/response payloads (AGENTS style rule).

### 5.2 Config → training-argument mapping

| Config key | LLaMA-Factory argument | Notes |
|---|---|---|
| `model` | `model_name_or_path` | default `Qwen/Qwen2.5-1.5B-Instruct` |
| `learning_rate` | `learning_rate` | playbook knob |
| `batch_size` | `per_device_train_batch_size` | OOM trigger knob |
| `num_workers` | `dataloader_num_workers` | stall knob |
| `num_epochs` | `num_train_epochs` | overfit/regression knob |
| `lora_r` / `lora_alpha` | `lora_rank` / `lora_alpha` | regression fix knobs |
| `gradient_clip_norm` | `max_grad_norm` | divergence/explosion fix knob |
| `chat_template` | `template` | tokenizer-mismatch knob |
| — | `dataset=alpaca_slice`, `max_steps≈60`, `cutoff_len` | fixed smoke envelope |

### 5.3 Real-log detection (`playbook.py` / `simlogs.py`)

- Playbook detection lambdas are extended (not replaced) to also match real LLaMA-Factory
  phrasings: real OOM already matches (`torch.cuda.OutOfMemoryError`); NaN adds
  `loss=nan`/`loss: nan` variants; divergence adds rising `loss=` series analysis on real
  lines. Calibration uses **captured fixtures from the real smoke run** (§7) — no invented
  log formats.
- Simulation generators/detectors are untouched (all existing tests stay green).

### 5.4 `supertaco/runner.py` — real branch + eval gate

- `run(config, *, max_retries=3, dry_run=True, on_event, log_fn=generate_logs,
  job_client=None, llm=None) -> RunResult` — `job_client` injectable for tests (same seam
  style as `make_llm`).
- Real branch: preflight → launch/poll/logs via `job_client` → detect → patch → relaunch;
  on clean logs: extract responses JSON → eval gate (§4 step 6). `RunResult` gains
  `eval_results`.
- Responses extraction is strict: missing or unparseable block after a successful job ⇒
  eval skipped + `RunResult.eval_results=None` + visible note (fetch failure ≠ model
  failure; §6).

### 5.5 `supertaco/eval/` — dual-mode signal

- `run_eval_suite` unchanged (takes response dicts).
- Regression decision is mode-aware:
  - real mode: `avg_ft < avg_base × 0.9` on real judge scores (loop-driving signal, D1);
  - dry-run: `classify_config_failure(final_config) is not None` (ground truth, D6).
- `build_responses` + template path remain for dry-run display.

### 5.6 `supertaco/ui/dashboard.py` — mode toggle, loop results

- Sidebar: **Simulation / Real GPU** selector; Real shows preflight status and the cost
  note; Simulation unchanged.
- Real mode: eval panel renders `run_result.eval_results` after the loop (auto), manual
  button hidden; the timeline shows the healing (`failure_detected(EVAL_REGRESSION)` →
  `patch_written` → `run_succeeded`) — fix is visible, not a flag.
- Dry-run mode: eval panel/button exactly as today, with the D6 flag rule.

### 5.7 `supertaco/cli.py` — `--real` flag

- `run <cfg> [--real | --dry-run]`; default stays simulation (D8). Real path asks for no
  extra input but the cost-bearing launch requires prior user approval per AGENTS rule 6
  (approval is logged in the run record).
- Exit codes unchanged.

### 5.8 `scripts/live_smoke.py` — cost-bearing live harness (manual, never in pytest)

- Preflight probe (DNS/auth/permission) → one launch (nvidia-smi → then the train+eval job)
  → prints job ID, status transitions, first/last log lines, responses JSON extraction.
- Each invocation is individually approved by the user and logged in `docs/PROJECT.md`.

## 6. Error handling

| Condition | Behavior |
|---|---|
| Preflight fail (DNS / auth / 403) | `run_failed` before any launch, actionable message (hosts-file hint / console-IAM hint); no heuristic fallback (D9) |
| Jobs API error mid-run (poll/logs) | `run_failed` with API error; real mode never simulates a continuation |
| Job crashes (container error, OOM at runtime) | Real logs carry the traceback → normal `detect_failure` → patch → relaunch |
| Poll timeout (no terminal state in 90 min) | `stop_job` attempted, `run_failed` with last logs; run record notes spend |
| Responses JSON missing/unparseable after success | Eval skipped, `eval_results=None`, visible note; run still counts as succeeded (fetch ≠ model failure) |
| Judge HTTP failure | Existing hash-fallback + visible banner (previous spec D3/D4 semantics kept) |
| Regression still present at budget end | `run_failed` with `failure_key=EVAL_REGRESSION` — the only moment a flag renders |
| Cost control | Each GPU run individually approved + logged (AGENTS rule 6); ≤1 GPU-hour per run, ≤4 runs per heal cycle |

## 7. Testing

Offline by default (AGENTS: `-m "not integration"`); live paths marked `integration` or kept
in `scripts/live_smoke.py` (approval-gated, never pytest).

1. **`tests/test_nebius_jobs.py`** — mocked HTTP/SDK: request body shape, auth header,
   status transitions, log paging, `stop_job` on timeout, error mapping to typed exceptions.
2. **`tests/test_real_detectors.py`** — detectors against **committed real-log fixtures**
   captured in milestone 2 (real LLaMA-Factory OOM traceback, real `loss=nan`, real clean
   run); plus the existing 37 simulated-log tests untouched.
3. **`tests/test_runner_real.py`** — injected fake `job_client` (seam, not a fake product):
   happy path with eval gate pass; eval regression → patch → relaunch → pass; budget
   exhaustion → `run_failed(EVAL_REGRESSION)`; preflight failure path; responses-JSON
   extraction edge cases.
4. **`tests/test_eval_signal.py`** — D6: healthy config ⇒ no flag even with judge scores at
   the 0–1 floor (the reported bug pinned); unhealthy config ⇒ flag; real-rule unit tests.
5. **Dashboard AppTest** — mode toggle wiring (Simulation path unchanged; Real path mocked
   at the runner seam); eval panel renders `run_result.eval_results` in real mode.
6. **Gates per task**: `pytest tests/ -m "not integration"` green; `ruff check` +
   `ruff format --check` clean on touched files; mypy on new modules (matching previous
   spec's policy).
7. **Live integration (approval-gated)**: milestone smoke/drill/demo runs via
   `scripts/live_smoke.py` / `pytest -m integration`; each run logged with spend.

## 8. Milestones (single plan, executed in order)

| # | Milestone | GPU spend |
|---|---|---|
| 0 | **Prerequisites**: ✅ DONE 2026-09-28 — federation profile with jobs 200; Windows+WSL hosts entries (Sky hijack) | none |
| 1 | Jobs API contract probe → real `NebiusJobClient` + unit tests | none (contract probe may be free GETs) |
| 2 | Image + recipe → **first live smoke** (tiny train → responses JSON) + capture real-log fixtures | ~1 run |
| 3 | Real-log detector calibration → **live failure drill** (real OOM/NaN → patch → clean rerun) | ~2 runs |
| 4 | Eval gate + D6 dry-run fix + offline tests | none |
| 5 | Full loop demo run → dashboard/CLI/docs → final gates | ≤4 runs |

Total ≈ **$5–15** against the $100 budget, every run approval-gated.

## 9. Documentation updates (at implementation time)

- **HANDOFF**: Gate 1 status (jobs/training/eval real; endpoints still open), invariant
  table updated for real mode (invariants 1–3 kept verbatim; invariant 4 reworded to cover
  both modes), access/DNS facts from §1.
- **PROJECT.md**: per-task recent-changes lines (existing rule); milestone status.
- **STACK.md / README**: `--real` usage, mode toggle, cost guardrails, model/dataset/GPU.
- **AGENTS.md examples**: align `run <cfg>` examples with the simulation-default (D8).
- **`.env.example`**: no new keys required (same `NEBIUS_API_KEY` + `NEBIUS_PROJECT_ID`);
  document that jobs need a jobs-capable credential.

## 10. Success criteria

- One `--real` run on a broken config: real OOM/NaN appears in **real** logs → classified →
  patched YAML written → retrained → clean → real base/ft responses judged → pass, with the
  timeline showing the whole heal (or a bounded `run_failed` after 3 relaunches — never a
  bare flag)
- An overfit-prone real recipe: eval regression detected → `EVAL_REGRESSION` patch →
  retrained → eval passes (or budgeted fail) — "fix, not flag" demonstrated end-to-end
- Dry-run: healthy runs show **no** regression banner (false positive pinned by a test);
  simulation demo unchanged and green
- All offline gates green; zero GPU spend without explicit per-run approval; total spend
  logged against the $100 budget

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Jobs API contract/permissions unknown until prereq 0 | Milestone 1 starts with a live probe; transport has SDK + REST fallback |
| LLaMA-Factory image not pullable in Serverless Jobs | Try official `hiyouga/llamafactory` image first; fallback: Nebius PyTorch catalog image + `pip install llamafactory` in the container command |
| HF model download per run (~3 GB, 1–2 min) | Accepted for v1; File Storage cache only if it measurably hurts |
| Real OOM not reproducible with a 1.5B model | Milestone 3 drill calibrates batch/`cutoff_len` (documented seed, not guesswork) |
| Judge may not separate base vs ft on real outputs | Milestone 5 adjusts recipe (overfit / degraded-training seeds); regression rule and fixtures validated before the demo run |
| DNS poisoning returns (DHCP/hosts churn) | Preflight re-checks resolution every run; run record logs resolved IP |
