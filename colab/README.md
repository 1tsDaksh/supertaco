# Google Colab GPU training (free T4) — dashboard-driven

Kaggle was dropped (its sandbox has no outbound internet despite
`enable_internet: true`). Colab's free T4 has full internet, so this is the
real-GPU smoke path.

## How it runs (from the dashboard, no manual steps)

1. One-time auth: run `C:\Users\Zephyr\.venv-colab\Scripts\colab.exe usage`,
   open the Google URL, paste the code back into the terminal.
2. Start the Streamlit dashboard, scroll to
   **"Real GPU Training (Google Colab, free T4)"**, press
   **Train on Colab (T4)**.
3. The adapter (`supertaco/gpu/colab.py`) drives `google-colab-cli`:
   `new` (provision T4) -> `exec colab/train_lora.py` (streams `step N loss X`
   lines into the live chart) -> `download` to `colab/output/lora_adapter.zip`
   -> `stop`. The zip is offered as a dashboard download button.

The CLI lives in the `~/.venv-colab` venv (official Linux/macOS tool; Windows
works via a tiny `termios` stub — see `supertaco/gpu/colab.py`).

## What the training script does

- Ensures `peft`/`datasets`/`accelerate` are fresh (Colab preinstalls
  torch/transformers).
- LoRA-finetunes `Qwen/Qwen2.5-0.5B-Instruct` on a 256-row slice of
  `yahma/alpaca-cleaned`, with knobs mirroring the SuperTaco config schema
  (`learning_rate`, `lora_r`, `lora_alpha`, `num_epochs`, `batch_size`).
- Prints `step N loss X` lines in the same format the simulator emits, so
  `supertaco.loop.extract_loss_points` parses real logs.
- Saves the adapter + tokenizer to `/content/lora_adapter/` and zips it.

## Manual fallback (browser)

1. Open https://colab.research.google.com -> **Upload** -> pick
   `colab/train_lora.ipynb`.
2. **Runtime -> Change runtime type -> GPU (T4)**.
3. **Runtime -> Run all** (downloads `lora_adapter.zip` in the browser).

## Free-tier notes

- T4 quota is ~hours/day; this run needs ~10 min.
- `exec` has a silence timeout: quiet stretches longer than the timeout are
  treated as a hung session (the notebook prints every step to avoid this).
- If the runtime disconnects, rerun — nothing is lost until the final save.
