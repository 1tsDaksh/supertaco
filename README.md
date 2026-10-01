# SuperTaco

AI-supervised fine-tuning sandbox for the Nebius x NVIDIA Global AI Hackathon.

Give it a config - base model, hyperparameters, prompts - and SuperTaco runs the whole loop:
LoRA training on a **free Google Colab T4**, live log streaming, automatic failure diagnosis
(NaN loss, OOM, divergence, eval regression, ...), config patching, relaunch (up to 3 times),
then a before/after comparison scored by an NVIDIA **Nemotron** judge on **Nebius Token
Factory**. Every attempt, event, and score is surfaced in a Streamlit attempt-ledger
dashboard.

## How it works

```
dashboard / CLI  (supertaco run configs/defaults/colab_t4.yaml)
      |
      v
supertaco.loop - supervisor loop (initial attempt + up to 3 relaunches)
      |
      v
ColabTransport  -> renders colab/train_lora.py with config + eval prompts baked in
      |            google-colab-cli: new (fresh T4) -> exec -> download -> stop
      |            streams "step N loss X" lines into the timeline/loss chart
      |            artifact: colab/output/lora_adapter*.zip
      v
on-VM eval phase: base + fine-tuned answers for a fixed 5-prompt suite
      |
      v
Nemotron judge on Nebius Token Factory (0-10 per answer, interleaved base -> ft)
      |
      +-- avg(ft) >= 0.9 * avg(base)  -> run_succeeded (attempt-ledger report)
      |
      +-- regression or train failure -> 8-mode playbook detects the failure
                                          -> Nemotron classifies (verdict + divergence flag)
                                          -> deterministic playbook fix written to a NEW
                                             config in configs/runs/ -> relaunch
```

Failure handling:

- **8 playbook failure modes**: `NAN_LOSS`, `OOM`, `GRADIENT_EXPLOSION`, `LOSS_DIVERGENCE`,
  `LOSS_PLATEAU`, `EVAL_REGRESSION`, `TOKENIZER_MISMATCH`, `DATALOADER_STALL`.
- **Retry budget**: initial attempt + max 3 relaunches (4 attempts total). Exhaustion ends the
  run with `run_failed: MaxRetriesExceeded` - a 4th relaunch is never attempted.
- **Fail fast**: validation or infrastructure errors end the run immediately with an
  actionable `run_failed`; the VM is always stopped, no silent fallbacks or orphan sessions.
- **Immutable configs**: every patch creates a new file - the original config is never edited.

## Quick start

Prerequisites: Python 3.11+, a Google account (one-time Colab auth), a Nebius Token Factory
API key.

```bash
git clone https://github.com/1tsDaksh/supertaco.git && cd supertaco
pip install -e .
pip install --group dev

cp .env.example .env    # set NEBIUS_API_KEY (judge + classifier calls)

# one-time Colab auth: run `colab usage` in the google-colab-cli venv
# and open the Google URL - details in colab/README.md

# CLI: real supervisor loop on the default T4 config
# (free GPU, ~10-15 min; spends judge tokens - ask first)
supertaco run configs/defaults/colab_t4.yaml

# Dashboard: config editor, live timeline + loss chart, attempt-ledger report
streamlit run supertaco/ui/dashboard.py --server.headless true --server.port 8510
```

## Configuration

`configs/defaults/colab_t4.yaml`:

```yaml
model: qwen2.5-0.5b     # T4-viable whitelist: qwen2.5-0.5b | qwen2.5-1.5b | qwen2.5-3b | tinyllama-1.1b
learning_rate: 0.0002
batch_size: 4
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
num_epochs: 1
chat_template: llama-3
```

- Training data: a 256-row slice of `yahma/alpaca-cleaned` (see `colab/train_lora.py`).
- Generated artifacts: attempt scripts and patched configs in `configs/runs/`, adapter zips in
  `colab/output/`.
- `configs/base/` is reference-only - never modify it.
- Models outside the T4 whitelist are rejected before any GPU is provisioned.

## Nemotron / Token Factory usage

| Role | Model | When |
|---|---|---|
| Failure classification | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | every detected failure |
| Patch proposal + judge | `nvidia/nemotron-3-super-120b-a12b` | proposal each retry; 10 judge calls per run (5 prompts x base/ft) |
| Deep reason | `nvidia/Nemotron-3-Ultra-550b-a55b` | reserved for novel failures the playbook misses |

- Cost routing starts cheap and escalates only when needed. **Every** call is logged to
  `logs/llm_calls.jsonl` (model, mode=`real`/`fallback`, tokens, latency, error) for audit.
- On API errors the client trips a circuit breaker and falls back to simulated/hash scores,
  clearly marked `fallback` - never silently mixed with real results.
- Budget: **$50 Token Factory credits** - the only metered resource (Colab T4 training is
  free). Live runs require explicit human approval.

## Testing & validation

```bash
python -m pytest tests/ -q                # 84 tests - transport and LLM faked, no network/credits
python -m ruff check .                    # lint
python -m ruff format --check supertaco tests
python -m mypy supertaco/                 # 13 pre-existing findings (deferred)
```

## Documentation

- [docs/AGENTS.md](docs/AGENTS.md) - behavioral rules, commands, cost guardrails
- [docs/PROJECT.md](docs/PROJECT.md) - current state + changelog
- [docs/HANDOFF.md](docs/HANDOFF.md) - architecture, invariants, live-run history, known issues
- [docs/SPEC.md](docs/SPEC.md), [docs/CONVENTIONS.md](docs/CONVENTIONS.md),
  [docs/STACK.md](docs/STACK.md)
- [colab/README.md](colab/README.md) - Colab adapter details + manual browser fallback

## Status

The real-loop implementation plan (14 tasks) is complete: live smoke on 2026-10-01 passed end
to end (exit 0, `run_started -> run_succeeded` on attempt 1, judge mode `real`, VM stopped).
Next up: demo video + submission write-up (deadline 2026-10-30).

## License

Apache 2.0 - see [LICENSE](LICENSE).
