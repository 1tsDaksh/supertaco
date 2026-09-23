# SandboxTune

AI-supervised fine-tuning sandbox for the Nebius x NVIDIA Global AI Hackathon.

## Product summary
An AI-supervised fine-tuning sandbox. User supplies (dataset, base model, goal). The agent writes
the config, launches training on Nebius AI Cloud, monitors it, auto-heals failures using a
data-driven playbook, then evaluates and deploys the checkpoint — showing before/after results.

## License
See [LICENSE](LICENSE) for Apache 2.0 licensing terms.

## Nemotron / Token Factory usage
This project uses NVIDIA Nemotron models (via Nebius Token Factory) for the agent brain:
- Fast classification: `nvidia/nemotron-3-nano`
- Patch/draft: `nvidia/nemotron-3-super`
- Deep reason: `nvidia/nemotron-3-ultra`

Cost routing starts at nano and escalates to super/ultra only when needed. All LLM calls are logged to `logs/llm_calls.jsonl` for audit and cost tracking.

## Token Factory / AI Cloud credits
- Budget: $50 Token Factory credits + $100 AI Cloud credits
- All job launches support `--dry-run` mode
- Production runs require explicit human approval

## Usage
```bash
# Install dependencies
uv sync

# Run a fine-tuning job (dry-run first)
sandboxtune run configs/runs/broken_nan_loss.yaml --dry-run

# Run with real Nebius job (requires API key)
sandboxtune run configs/runs/smoke_test.yaml
```