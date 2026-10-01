# SandboxTune

AI-supervised fine-tuning sandbox for the Nebius x NVIDIA Global AI Hackathon.

## Product summary
An AI-supervised fine-tuning sandbox. User supplies (dataset, base model, goal). The agent writes
the config, launches training on a free Colab T4 (via `colab/train_lora.py`), monitors it,
auto-heals failures using a data-driven playbook, then evaluates and deploys the checkpoint —
showing before/after results.

## License
See [LICENSE](LICENSE) for Apache 2.0 licensing terms.

## Nemotron / Token Factory usage
This project uses NVIDIA Nemotron models (via Nebius Token Factory) for the agent brain:
- Fast classification: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`
- Patch/draft: `nvidia/nemotron-3-super-120b-a12b`
- Deep reason: `nvidia/Nemotron-3-Ultra-550b-a55b`

Cost routing starts at nano and escalates to super/ultra only when needed. All LLM calls are logged to `logs/llm_calls.jsonl` for audit and cost tracking.

## Token Factory / AI Cloud credits
- Budget: $50 Token Factory credits + $100 AI Cloud credits
- Production runs require explicit human approval

## Usage
```bash
# Install dependencies
uv sync

# Run the real supervisor loop on the default T4 config
# (free Colab GPU, ~15 min; spends judge tokens - ask first)
supertaco run configs/defaults/colab_t4.yaml
```

Lint/typecheck/test gates: see the "Testing & validation commands" section in
[docs/AGENTS.md](docs/AGENTS.md).