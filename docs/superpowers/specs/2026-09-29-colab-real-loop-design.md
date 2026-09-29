# Colab Real Training Loop — Design Spec

> Date: 2026-09-29 · Status: approved (design parts validated + approved in chat; pending written-spec review)
> Supersedes: `2026-09-28-real-gpu-autoheal-design.md` (Nebius jobs transport — abandoned)
> Scope: the project becomes one real pipeline — dashboard config → Colab T4 training
> attempts → Token Factory supervisor (classify/patch/judge) → detailed report.

---

## 1. Problem

Two parallel systems exist today, both wrong for the demo:

| Layer | Today |
|---|---|
| Dry-run loop (`runner.py`) | Supervisor loop works but over **simulated** logs (`simlogs.generate_logs`); eval scores template strings from `build_responses` |
| Colab button (`render_gpu_panel`) | Real T4 training but **hardcoded** config, single-shot, no supervisor, no eval |
| Nebius path (`supertaco/nebius/`) | Stubs that raise `NotImplementedError`; the 2026-09-28 plan around them is dead |

User direction (2026-09-29): the dashboard should run **one config → Colab training** with the
**supervisor loop backed by Token Factory**, ending in a **detailed evaluation and a report of
every change made**. Old machinery is removed.

## 2. Decisions (resolved with user)

| # | Question | Decision |
|---|---|---|
| D1 | Transport | **Google Colab T4** via the proven `supertaco/gpu/colab.py` adapter; Nebius jobs removed entirely |
| D2 | Config source | Dashboard-owned config (YAML path + sidebar knobs + model select); `model` resolved through a **T4 whitelist**, invalid models fail **before** provisioning |
| D3 | Cleanup | **Full removal** of the simulation loop, broken-fixture configs, and Nebius stubs; tests rewritten to fake the transport, not the loop |
| D4 | Judge | **Token Factory Nemotron only** (`judges.py`/`make_llm` unchanged); hash-mode banner stays as the honest degraded signal |
| D5 | Report | **Full attempt ledger** (per attempt: failure key, verdict, config diff, loss curve) + judge table + final config + adapter download |
| D6 | Architecture | **Approach B**: fresh `supertaco/loop.py`; cons neutralized by reusing playbook, judges, config validation, patch writer, timeline |
| D7 | Eval responses | Generated **on the VM** (base model + adapter, 5 fixed prompts) and emitted as a delimited JSON block in logs |
| D8 | Budget | Initial attempt + ≤3 relaunches; log-failures and eval-regressions **share** the budget |
| D9 | Fallbacks | **None**: infra failures (auth, provisioning, exec, download) surface as `run_failed` with the real message |
| D10 | Dataset | Fixed 256-row `yahma/alpaca-cleaned` slice (T4-sized); not configurable this iteration |

## 3. Out of scope

- Nebius jobs/endpoints/storage (deleted, not replaced)
- Multiple datasets, HF_TOKEN/gated models, pushing adapters to the Hub
- Async/background dashboard execution (blocking button handler, as today — it streams fine)
- Endpoint deployment of the trained adapter
- Repo-wide lint/mypy debt outside touched files

## 4. Architecture

```
Dashboard (config editor)                    Token Factory (Nemotron)
        │ config dict                              ▲ classify / propose_patch / judge
        ▼                                          │
supertaco/loop.py  run_training_loop() ────────────┘
        │  transport seam: AttemptTransport.run_attempt(config, prompts, on_line)
        ▼
supertaco/gpu/colab.py  (existing adapter, unchanged contract)
        │  new → exec(rendered script) → download(on success) → stop
        ▼
colab/train_lora.py  (rendered per attempt)
        │  phase 1: train, print `step N loss X`
        │  phase 2: generate base+adapter answers for 5 prompts
        │           print ###RESPONSES_JSON### {…} ###END_RESPONSES_JSON###
        ▼
        artifact zip pulled on success → colab/output/
```

### Loop semantics (one button click)

1. **Validate**: `validate_config` (moved from runner) + `resolve_model` whitelist. Failure →
   `run_failed`, no session provisioned.
2. **Attempt** (≤4 total): `job_launched` (payload `{"config": current, "model": hf_id,
   "session": name}`) → transport streams lines via `on_line` (live panel) → returns logs +
   artifact. One aggregated `logs_produced` event per attempt (text + `extract_loss_points`).
3. **Detect**: responses block stripped first (`split_responses`), then `detect_failure` on
   real logs (playbook extended with real phrasings: lowercase `loss nan`, CUDA OOM text,
   plateau/divergence over `step N loss X` series). Transport errors (auth/provision/exec/
   download) → `run_failed` immediately, consuming an attempt.
4. **Classify + patch**: `classified` (Nemotron verdict, mode, diverged) → `patch_proposed`
   (Nemotron reasoning) → `apply_default_fix` → YAML written to `configs/runs/` →
   `patch_written` (before/after) → `retry_scheduled`.
5. **Healthy logs** → `run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=judge_client)` on the
   block from the logs. Regression ⇒ treat as failed attempt: `failure_detected`
   (`EVAL_REGRESSION`) → classify → playbook fix (halve lora_r/alpha, −1 epoch) → relaunch.
6. **Pass** → `run_succeeded`; adapter already downloaded; `LoopResult` populated.
7. **Budget exhausted** → `run_failed` (never raises), ledger still rendered.

## 5. Components

### 5.1 `supertaco/loop.py` (new, replaces `runner.py`)

```python
def run_training_loop(
    config: dict,
    *,
    transport: AttemptTransport,      # injected; real = ColabTransport, tests = fake
    llm,                              # Token Factory client (make_llm)
    on_event: OnEvent | None = None,
    max_retries: int = 3,
    runs_dir: str = "configs/runs",
) -> LoopResult: ...
```

- `LoopResult` keeps the dashboard-facing field names (`success`, `attempts`,
  `configs_written`, `eval_results`, `llm_calls`, `final_config`, `failure_key`, `error`)
  plus `attempt_ledger: list[AttemptRecord]` and `artifact: Path | None`.
- `AttemptRecord`: `attempt`, `failure_key`, `verdict`, `loss_points`, `config_before`,
  `config_after` (None on the final successful attempt).
- Moved-in helpers: `validate_config`, `_write_config` (patch YAML writer), `extract_loss_points`.
- Event types unchanged: `run_started, job_launched, logs_produced, failure_detected,
  classified, patch_proposed, patch_written, retry_scheduled, run_succeeded, run_failed`.

### 5.2 Config rendering + whitelist

- `resolve_model(name: str) -> str` — whitelist map, case-insensitive keys:
  `qwen2.5-0.5b`, `qwen2.5-1.5b` (+ optional `qwen2.5-3b`, `tinyllama-1.1b`). Unlisted or
  oversized (e.g. `llama-3-8b`) → `ConfigurationError("model X is not T4-viable; pick from: …")`
  **before** any provisioning.
- `render_script(config, prompts) -> Path` — pure: takes `colab/train_lora.py` template
  (CFG dict + prompts marked regions), substitutes values, writes `configs/runs/
  <timestamp>_attemptN.py` (gitignored). No CLI-arg guessing.

### 5.3 Transport contract

```python
class AttemptTransport(Protocol):
    def run_attempt(self, config: dict, prompts: list[str], on_line) -> AttemptResult: ...

@dataclass
class AttemptResult:
    logs: str                 # full streamed output incl. responses block
    artifact: Path | None     # set whenever the VM produced one (training completed)
```

The adapter is tiny (~6 MB), so the transport **always attempts the download before `stop`**
(the VM is gone afterwards — a download decision can't wait for the loop to see the logs).
The **loop** only surfaces `artifact` in `LoopResult` on a passing run; failed-attempt
artifacts are discarded.

`ColabTransport` wraps `run_training` (existing adapter); each attempt provisions a fresh VM
(state never leaks). On failure the VM is already stopped by the adapter's `finally`.

### 5.4 Responses block parser (new in `harness.py`)

`###RESPONSES_JSON###\n{"base": {p: a}, "fine_tuned": {p: a}}\n###END_RESPONSES_JSON###`;
`split_responses(log_text) -> (train_logs, responses|None)` — tolerant: unterminated block or
bad JSON → `(logs, None)`; the block is stripped before playbook detection. (Spec'ing carried
over from the abandoned Nebius plan; the code itself is new.)

### 5.5 VM script (`colab/train_lora.py`, rendered)

Phase 1 = today's proven training (torchao removal, `remove_unused_columns=False`).
Phase 2 = load adapter over base, generate answers for the 5 prompts with both weights
(`max_new_tokens=150`, greedy), print the block, zip adapter.

### 5.6 Judge / eval

`run_eval_suite` unchanged (×0.9 rule, per-prompt `base:real / ft:real` modes).
`build_responses` + BASE/HEALTHY/UNHEALTHY tiers **deleted** — responses are real from the VM.
Nemotron classify/patch calls stay exactly as today (`llm.classify_failure`/`propose_patch`).

## 6. Dashboard rework

- **Sidebar**: YAML path input (default `configs/defaults/colab_t4.yaml`, a new tracked sane
  default) + model select (whitelist) + knob inputs (learning_rate, lora_r, lora_alpha,
  num_epochs, batch_size) that overlay the YAML. Fixture loader **removed**.
- **One button**: 🚀 Fine-tune on Colab (supervisor loop). Replaces: dry-run Launch button,
  standalone eval button, single-shot GPU panel button.
- **Timeline**: `_render_events` wording adapted (`job_launched` → "launch `<model>` on T4,
  attempt N"); mangled emoji arrows fixed during the rewrite.
- **Charts**: per-attempt loss curves from `logs_produced` events (existing `_loss_figure`),
  plus the live stream slot during the run.
- **Report section** (after run): attempt-ledger table (attempt, failure key, verdict, config
  diff), judge table (prompt, base, ft, mode, Δ), regression verdict, final config YAML, adapter
  download button, LLM call table (existing).

## 7. Deletions

`supertaco/runner.py`, `supertaco/agent/simlogs.py`, `supertaco/agent/graph.py`,
`supertaco/agent/tools.py`, `supertaco/nebius/` (jobs/endpoints/storage), `scripts/
make_broken_configs.py`, `configs/runs/*_broken_model.yaml` fixtures, old spec superseded
(`2026-09-28-real-gpu-autoheal-design.md` gets a "superseded" banner; its plan file likewise).

Tests deleted/rewritten: `test_runner.py`, `test_simlogs.py`, sim parts of `test_cli.py` and
`test_dashboard.py`, `build_responses` pins in `test_judge.py`. Kept: `test_gpu_colab.py`,
judge real/fallback tests, playbook tests, adapter lifecycle tests.

## 8. Testing

- **Loop tests** (`tests/test_loop.py`): fake transport returning scripted logs/responses +
  `FakeLLM`. Cases: healthy first attempt; NaN logs → patch → heal; eval regression →
  EVAL_REGRESSION patch → heal; budget exhausted; transport error → `run_failed`, no fake
  fallback; `job_launched` payload keys; event order pinned.
- **Render/whitelist tests**: substitution correctness, pre-launch rejection, no provisioning on
  invalid model.
- **Parser tests**: split/tolerant cases (block stripped, unterminated, bad JSON).
- **AppTest** (`test_dashboard.py` rewritten): success flow renders ledger + judge table +
  download; failure flow shows `run_failed` without traceback; invalid model blocked pre-launch.
- **Gates**: `python -m pytest tests/ -q`, ruff check/format on touched files.
- **Live smoke** (final): one real Colab run from the dashboard, ~4–6 min/attempt.

## 9. Error handling

| Case | Behavior |
|---|---|
| Invalid/unviable model | `ConfigurationError` pre-launch; timeline shows `run_failed`, no VM |
| Colab CLI missing/not authenticated | adapter's `ColabError` message (actionable) → `run_failed` |
| Provisioning timeout (transient) | `run_failed` "timed out provisioning…"; user re-clicks |
| Exec cell exception (rc 0) | download-miss error now carries training tail (adapter, existing) |
| Judge unreachable | hash-mode banner; scores marked `hash`; regression rule still applies to flagged runs |
| Budget exhausted | `run_failed` + full ledger; no exception |

## 10. Project docs impact

- `AGENTS.md` rule 2 (dry-run) was Nebius-credit-specific; with Nebius removed and Colab free,
  it is rewritten to "Colab runs are free-tier; each supervisor run provisions ≤4 short VMs".
- `docs/superpowers/plans/2026-09-28-real-gpu-autoheal.md` + old spec: marked superseded.
- `PROJECT.md`: per-commit log lines as usual.

## 11. Milestones (for the implementation plan)

| # | Milestone |
|---|---|
| 1 | `loop.py` + LoopResult/AttemptRecord + fake-transport loop tests green |
| 2 | `resolve_model` + `render_script` + template markers in `colab/train_lora.py` |
| 3 | `split_responses` parser + script phase 2 (on-VM generation) + eval gate in loop |
| 4 | Dashboard rework (sidebar, single button, timeline wording, report section) |
| 5 | Deletions + superseded banners + AGENTS.md rule update |
| 6 | Live smoke: one real dashboard run end-to-end |
