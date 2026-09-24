# Functional Dashboard — Design Spec

> Date: 2026-09-24 · Status: approved (design sections 1–4 signed off by user)
> Scope: make the SuperTaco dashboard actually run the agent pipeline instead of simulating it

---

## 1. Problem

Every layer of the current pipeline is simulated, so the dashboard is decorative:

| Layer | Today |
|---|---|
| Dashboard "Launch Job" | Invents a `sim_` ID; timeline/loss/LLM panels never receive real data |
| Agent loop (`run_agent_loop`, `run_agent_graph`) | Structure exists, but monitor gets no logs; graph fakes `sim_job_N` |
| Nebius job client | Dry-run payload only; real launch/status raise `NotImplementedError` |
| Nemotron client | Heuristic responses; never calls Token Factory |
| Eval harness / judges | `random.uniform` scores; hash-based mock scoring |
| CLI | Crashes (undefined `NemotronClient`, `run_agent_loop`) |

## 2. Decisions (clarifiers resolved with user)

| # | Question | Decision |
|---|---|---|
| D1 | Depth of "actually run" | **A + B**: real agent loop in Nebius dry-run mode **and** real Nemotron calls via Token Factory |
| D2 | Dry-run log source | **Config-derived generator**: logs are a deterministic function of config values; broken-config fixtures pre-loaded in the demo menu |
| D3 | LLM failure behavior | **Auto-fallback with visible indicator**: API error → heuristic response, persistent banner, per-call `real\|fallback` label |
| D4 | Eval panel | **Real judge, structured mock responses**: deterministic templates scored by real Nemotron judge (with same fallback) |
| D5 | CLI | **Fix it in this effort**; CLI and dashboard share one run path |
| D6 | Architecture | **Approach 3**: single `supertaco.runner` module emitting typed events; both UIs consume it |

## 3. Out of scope

- Real Nebius AI Cloud GPU jobs, endpoint deploy, real training logs (handoff Gate 1; cost-bearing, needs explicit approval)
- Real checkpoint evaluation (requires Gate 1)
- Tavily real API (remains mocked; unknown-failure path not exercised by the generator)
- Repo-wide lint/mypy debt (42 pre-existing ruff errors outside touched files) — reported separately, not swept in
- `logs/` cleanup, run-history persistence across browser refreshes (session-state only)

## 4. Architecture (Approach 3)

```
Dashboard (Streamlit)                     CLI (supertaco run <cfg> --dry-run)
     │  config dict + options                   │  config path + flags
     └──────────────────┬───────────────────────┘
                        ▼
            supertaco/runner.py  ── ONE run path, emits typed events via callback
                        │
        ┌───────────────┼─────────────────────┬──────────────────┐
        ▼               ▼                     ▼                  ▼
  sim log generator  NemotronClient     NebiusJobClient      playbook
  (config →          REAL Token Factory dry-run payload     detect_failure /
   deterministic     HTTP via openai    only (no network)    apply_default_fix
   log lines/attempt SDK, auto-fallback
                      to heuristics)
```

### Run semantics (one "Launch Job" click)

1. Runner validates config (non-empty, has `learning_rate`) → emits `job_launched`
   (dry-run payload, zero network) → generator emits training logs derived from config.
2. Playbook `detect_failure(logs)`; healthy logs → run **succeeds**.
3. On failure: **real Nemotron nano classify** (verbatim in timeline; API error →
   fallback indicator) → playbook applies the typed fix → **new YAML written to
   `configs/runs/`** (handoff invariant 3 kept) → `patch_written` event carries
   before/after diff.
4. Logs regenerated from the patched config → healthy → **healed**.
5. Retry budget per handoff invariant 1: **initial launch + ≤3 relaunches; if the
   4th attempt's logs still fail → `MaxRetriesExceeded`** → `run_failed` event
   (not a crash).
6. **Patch authority = playbook** (deterministic healing; demo cannot flail).
   Real Nemotron super/ultra output is displayed as the model's reasoning beside
   each patch; nano-vs-playbook classification divergence is flagged visually but
   never blocks. Every LLM call appends to `logs/llm_calls.jsonl`
   (handoff invariant 2 — not actually implemented today).

### Event catalog

`run_started · job_launched(attempt, payload) · logs_produced(attempt, text) ·
failure_detected(attempt, key) · llm_call(model, mode=real|fallback, latency,
tokens, error?) · patch_proposed(reason) · patch_written(path, failure_key, diff) ·
retry_scheduled(next_attempt) · run_succeeded(final_config_path) ·
run_failed(error, attempts) · eval_scored(prompt, base, fine_tuned)`

Delivery: `on_event` callback. Dashboard appends to session state and updates
`st.empty()` placeholders for live rendering; CLI prints one line per event.

## 5. Components

### 5.1 `supertaco/agent/simlogs.py` (new) — deterministic log generator

Pure function `generate_logs(config) -> str`. Missing keys read as `None`.
Precedence rules — verified against all 8 fixtures in `configs/runs/` and all 5
sidebar demo configs, and constructed so **every playbook fix crosses its own
trigger threshold** (see §7 healing property):

| # | Trigger | Emitted failure | Why the playbook fix heals it |
|---|---|---|---|
| 1 | `batch_size > 32` | OOM traceback | `//2`: 64→32 (healthy) |
| 2 | `num_workers > 16` | stall warnings | `//2`: 20→10 (healthy) |
| 3 | `chat_template` ∉ `{None, "llama-3"}` | tokenizer/template error | fix sets `llama-3` |
| 4 | `lr ≥ 0.05` and `clip is None` and `epochs ≥ 8` | rising loss series | fix adds `clip=1.0` |
| 5 | `lora_r ≥ 32` and `epochs ≥ 10` | eval-regression lines | fix drops epochs to 9 |
| 6 | `lr ≥ 0.05` and `clip is None` and `template == "llama-3"` | grad_norm spikes | fix sets `clip=0.5` |
| 7 | `lr ≥ 0.05` and `template in {None, ""}` | `loss = NaN` lines | fix drops lr ×10 → 0.01 |
| 8 | `lr < 1.5e-6` | flat loss series | fix raises lr ×1.5 → 1.5e-6 |
| — | none match | healthy decreasing-loss logs | — |

Notes:
- Fixtures disambiguate the three `lr ≥ 0.05` modes: NAN_LOSS has `template=None`,
  GRADIENT_EXPLOSION has `template="llama-3"` + `epochs=3`, LOSS_DIVERGENCE has
  `epochs=10`. Sidebar demos (which omit `clip`/`template` keys) resolve correctly
  under the same rules (verified case-by-case during design).
- Multi-trigger configs heal **sequentially** (precedence order), still within the
  3-relaunch cap (worst fixture chain ≈ 2 retries; OOM 64 needs only 1 since the
  rule is `> 32`).
- Each mode emits a distinct loss curve (rising / flat / spiky / NaN / decay) so
  the loss chart tells the story.

### 5.2 `supertaco/runner.py` (new) — orchestration + events

```python
run(config, *, max_retries=3, dry_run=True, on_event, log_fn=generate_logs,
    llm=None, judge=None) -> RunResult
```

- Owns launch→generate→detect→classify→patch→relaunch loop and retry cap.
- `log_fn` injectable for tests (default: real generator).
- Emits the event catalog (§4); writes patched configs to `configs/runs/`
  (timestamped, never overwrites — invariant 3).
- Validates up front: config non-empty dict with `learning_rate`.

### 5.3 `supertaco/agent/llm.py` — real Token Factory calls

- `_call()` uses the already-installed `openai` SDK against
  `TOKEN_FACTORY_BASE_URL` with `nvidia/nemotron-3-{nano,super,ultra}` (README IDs),
  30s timeout.
- **Circuit breaker**: after the first failure in a run, later calls skip HTTP and
  go straight to fallback — dead API costs one timeout, not ten.
- Fallback → existing `_simulate_response`; every call records
  `mode=real|fallback`, latency, tokens (API `usage`), error string.
- `log_call` additionally appends JSONL to `logs/llm_calls.jsonl` (invariant 2);
  write failure → warning only, never kills a run.

### 5.4 `supertaco/eval/` — real judge, mock responses

- Template responses for the 5 fixed prompts (`PROMPT_SUITE_SIZE`): base variant
  vs fine-tuned variant (degraded if the final config is still unhealthy).
- `judges.py` scores each pair with a real Nemotron rubric call (same fallback +
  circuit breaker), clamps 0–10, sets regression flag (existing `avg_ft <
  avg_base * 0.9` rule).
- Eval runs only after a completed run; each score independent.

### 5.5 `supertaco/ui/dashboard.py` — panels go live

- Launch handler calls `runner.run()` synchronously; `st.empty()` placeholders
  stream events (timeline, logs) as they arrive.
- **Timeline**: real events — verbatim Nemotron reasoning, failure badges,
  config before/after diff.
- **Loss chart**: per-attempt series parsed from generated `loss:` lines
  (replaces empty traces).
- **LLM-calls table**: model, tokens, latency, real/fallback, escalation reason.
- **Sidebar**: persistent fallback-warning banner; fixture loader dropdown reading
  `configs/runs/*.yaml` (alongside existing demo configs); key-presence captions kept.
- **Dry-run checkbox relabeled** to state honestly that Nebius jobs stay dry-run
  until Gate 1.
- **Eval panel**: real scores after run completion; disabled with hint otherwise.
- Settings: keeps `_load_env()` (env-var precedence over `.env`), then imports
  package settings/clients through the runner seam only.
- **Import surface (explicit)**: dashboard imports `supertaco.runner` (run path)
  and `supertaco.eval.harness` / `supertaco.eval.judges` (eval panel) — no other
  package internals, and no duplicated pipeline logic.
- **Invariant change**: dashboard drops "self-contained by duplication" for one
  narrow import (`supertaco.runner` + helpers). Documented in HANDOFF §5.

### 5.6 `supertaco/cli.py` — fixed + rewired

- Adds the two missing names by rewiring to `runner.run()` with a printing event
  callback (fixes both F821 crashes).
- Exit codes: 0 success, 1 `MaxRetriesExceeded`/error (unchanged semantics).

### 5.7 Support fixes (on the run path)

- `playbook.py`: import missing `ConfigurationError` (F821 → would crash on
  unknown failure key).
- `judges.py`: import missing `List` (F821).

## 6. Error handling

| Condition | Behavior |
|---|---|
| LLM call exception | Per-call fallback + event; **circuit breaker** after 1st failure; sidebar banner "Nemotron unavailable — heuristic fallback (N calls)" |
| nano vs playbook divergence | Timeline flag; playbook still governs fix |
| `MaxRetriesExceeded` | `run_failed` event → red timeline entry, attempts + last config path; no stack trace |
| Exception escaping `runner.run()` | Dashboard wraps call → `st.error` + render completed events; never a Streamlit traceback |
| Empty config / no `learning_rate` | Validated before run; clear message; no run starts |
| Editor YAML typo | Existing behavior: inline parse error, previous config used |
| `llm_calls.jsonl` write failure | Warning only |
| Eval before any run | Button disabled with hint |
| Concurrent launches | Impossible: synchronous inside one button handler; rerun resets placeholders |
| Dry-run networking | Zero Nebius HTTP (client wrapper may be constructed; `launch_job(dry_run=True)` never hits network) — invariant 4 reworded accordingly |

## 7. Testing

All free / offline; `tests/` created (none exists today).

1. **`tests/test_simlogs.py`** — each fixture → expected key; each sidebar demo →
   expected key; healthy config → no failure; **healing property** (fixture →
   `apply_default_fix` → regenerated logs healthy — the demo-terminates proof);
   determinism (byte-identical output); loss-series shape per mode.
2. **`tests/test_runner.py`** — event sequences for success / heal / cap paths
   (cap tested via injected always-failing `log_fn`; 4-attempt semantics);
   `patch_written` files land in `configs/runs/` with differing keys; circuit
   breaker + fallback with raising client.
3. **`tests/test_llm.py`, `tests/test_judge.py`** — mocked `openai` client:
   real-mode token/latency capture; error-mode fallback + JSONL append; judge
   clamp 0–10 + independent fallback.
4. **AppTest integration** (extends existing suite): launch NAN_LOSS fixture →
   `failure_detected → patch_written → run_succeeded`, config file exists, LLM
   table populated, loss chart renders; deterministic offline by overriding
   `TOKEN_FACTORY_BASE_URL` to unreachable localhost (env precedence over
   `.env` already supported). CLI smoke: in-process `cli.main()` with patched
   client → prints events, exit 0.
5. **Gates**: `pytest tests/ -m "not integration"`; `ruff check` + `ruff format
   --check` on touched files; `mypy` on the two new modules only. Optional live
   check: **one** real nano call to prove `mode=real` — ask user first (costs
   credits, however tiny).

## 8. Documentation updates (at implementation time)

- **HANDOFF §5**: replace "dashboard self-contained" with "dashboard consumes
  `supertaco.runner` event seam"; reword invariant 4 to "dry-run never makes real
  Nebius network calls"; keep invariants 1–3 unchanged.
- **HANDOFF §4/§9**: remove the "dashboard does not import the package" gap;
  CLI status honest (fixed); `mypy src/` wrong-path note stays as known debt.
- **HANDOFF §2 status table + PROJECT.md recent changes**: updated per AGENTS
  workflow when implementation lands.
- **AGENTS.md**: unchanged (its rules already cover this work).

## 9. Success criteria

- One click on a broken fixture runs detect → real Nemotron call(s) → patch file
  in `configs/runs/` → relaunch → heal, with every panel updating live from real
  events
- Pull the Ethernet cable mid-run → banner + fallback labels, run still completes
- `supertaco run <fixture> --dry-run` works from the terminal (same runner)
- All tests + ruff green on touched files; no real GPU/network spend without
  asking
