# Kaggle GPU smoke run (free)

One-off LoRA fine-tune on a Kaggle P100 to validate the manual Phase-1
training loop. The agent/autoheal path stays on Nebius; this is the
zero-cost way to prove the training config end-to-end and produce a real
LoRA checkpoint.

## Prerequisites (one time)

1. Kaggle account with **phone verification** done (required for GPU quota,
   ~30 h/week of P100/T4).
2. Kaggle CLI authenticated. On this machine the token is already at
   `~/.kaggle/access_token` for the venv at `C:\Users\Zephyr\.venv-kaggle`.
   Elsewhere:
   ```bash
   pip install kaggle
   export KAGGLE_API_TOKEN=...   # or write it to ~/.kaggle/access_token
   ```
   Never commit the token — `~/.kaggle` lives outside this repo.

## Run it

```bash
# from the repo root; KAGGLE_EXE points at the isolated venv CLI
$env:KAGGLE_EXE = "C:\Users\Zephyr\.venv-kaggle\Scripts\kaggle.exe"
$k = $env:KAGGLE_EXE

# 1. put your username in kaggle/train_lora/kernel-metadata.json  ("id" field)
(Get-Content kaggle/train_lora/kernel-metadata.json) -replace 'USERNAME', '<your-kaggle-username>' | Set-Content kaggle/train_lora/kernel-metadata.json

# 2. push (creates/updates the private kernel and starts it on a GPU)
& $k kernels push -p kaggle/train_lora/

# 3. watch progress (status: complete / error / running)
& $k kernels status <your-kaggle-username>/supertaco-lora-smoke

# 4. when complete, download the LoRA adapter output
& $k kernels output <your-kaggle-username>/supertaco-lora-smoke -p kaggle/output
```

## What the notebook does

- Installs `peft`/`datasets`/`accelerate` (torch + transformers are
  preinstalled on the Kaggle image).
- LoRA-finetunes `Qwen/Qwen2.5-0.5B-Instruct` on a 256-row slice of
  `yahma/alpaca-cleaned`, with knobs mirroring the SuperTaco config schema
  (`learning_rate`, `lora_r`, `lora_alpha`, `num_epochs`, `batch_size`).
- Prints `step N loss X` lines in the same format the simulator emits, so
  `supertaco.agent.simlogs.extract_loss_points` can parse real logs.
- Saves the adapter + tokenizer to `/kaggle/working/lora_adapter/`, which
  Kaggle exposes as kernel output.

Runtime is ~10-15 minutes including installs. If the run errors, check the
kernel page on kaggle.com for the full log.
