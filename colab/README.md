# Google Colab GPU smoke run (free) — RECOMMENDED PATH

The Kaggle path is currently blocked: this account's Kaggle sessions have no
outbound internet (API accepts `enable_internet: true`, sandbox DNS is still
blocked — see `kaggle/README.md`). Colab's free T4 has full internet and works
today.

## Run it (3 steps, ~10 min)

1. Open https://colab.research.google.com → **Upload** → pick
   `colab/train_lora.ipynb` from this repo.
2. **Runtime → Change runtime type → GPU (T4)**.
3. **Runtime → Run all**.

## What it does

- Ensures `peft`/`datasets`/`accelerate` are fresh (Colab preinstalls
  torch/transformers).
- LoRA-finetunes `Qwen/Qwen2.5-0.5B-Instruct` on a 256-row slice of
  `yahma/alpaca-cleaned`, with knobs mirroring the SuperTaco config schema
  (`learning_rate`, `lora_r`, `lora_alpha`, `num_epochs`, `batch_size`).
- Prints `step N loss X` lines in the same format the simulator emits, so
  `supertaco.agent.simlogs.extract_loss_points` can parse real logs.
- Saves the adapter + tokenizer to `/content/lora_adapter/`, zips it, and
  triggers a browser download of `lora_adapter.zip` at the end.

## Free-tier notes

- T4 quota is ~hours/day; this run needs ~10 min.
- If the runtime disconnects, just Run all again — nothing is lost until the
  final save.
