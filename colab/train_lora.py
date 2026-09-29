"""LoRA smoke run for Google Colab - executed remotely via google-colab-cli.

Same SuperTaco-shaped knobs as the dashboard configs; prints `step N loss X`
log lines (simulator format) and packs the adapter to /content/lora_adapter.zip
for `colab download`.
"""

import os
import shutil
import subprocess
import sys


def ensure_packages() -> None:
    import importlib.util

    missing = [p for p in ("peft", "datasets", "accelerate") if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def main() -> None:
    ensure_packages()

    import datasets
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    CFG = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "num_epochs": 1,
        "batch_size": 4,
        "grad_accum": 2,
        "max_rows": 256,
    }
    print("config:", CFG, flush=True)

    assert torch.cuda.is_available(), "no GPU on the Colab VM"
    print("gpu:", torch.cuda.get_device_name(0), flush=True)

    def to_messages(rec):
        user = rec["instruction"].strip()
        if rec.get("input"):
            user = user + "\n\n" + rec["input"].strip()
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": rec["output"].strip()},
            ]
        }

    ds = datasets.load_dataset("yahma/alpaca-cleaned", split="train")
    ds = ds.select(range(CFG["max_rows"])).map(to_messages)

    tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate(batch):
        texts = [tokenizer.apply_chat_template(b["messages"], tokenize=False) for b in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc["labels"] = enc["input_ids"].clone()
        enc["labels"][enc["labels"] == tokenizer.pad_token_id] = -100
        return enc

    print(f"dataset rows: {len(ds)}", flush=True)

    class LossPrinter(TrainerCallback):
        """Emit SuperTaco-format log lines: step N loss X."""

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                print(f"step {state.global_step} loss {logs['loss']:.4f}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        CFG["model"], torch_dtype=torch.float16, device_map="cuda"
    )
    lora = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="/content/out",
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["learning_rate"],
        num_train_epochs=CFG["num_epochs"],
        fp16=True,
        logging_steps=4,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
        callbacks=[LossPrinter()],
    )
    trainer.train()
    print("training finished", flush=True)

    out = "/content/lora_adapter"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    zip_path = shutil.make_archive("/content/lora_adapter", "zip", out)
    print("artifact:", zip_path, flush=True)


if __name__ == "__main__":
    main()
