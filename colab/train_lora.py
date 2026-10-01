"""LoRA smoke run for Google Colab - executed remotely via google-colab-cli.

Same SuperTaco-shaped knobs as the dashboard configs; prints `step N loss X`
log lines (simulator format) and packs the adapter to /content/lora_adapter.zip
for `colab download`.
"""

import json
import os
import shutil
import subprocess
import sys


def ensure_packages() -> None:
    import importlib.util

    missing = [p for p in ("peft", "datasets", "accelerate") if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    # Colab images ship torchao <0.16 and peft's is_torchao_available() *raises*
    # (instead of returning False) for too-old versions; the plain Linear LoRA
    # path we use never needs torchao, so drop it rather than fight the pin.
    if importlib.util.find_spec("torchao") is not None:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False
        )


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

    # >>> SUPER_TACO_CFG >>>
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
    # <<< SUPER_TACO_CFG <<<
    # >>> SUPER_TACO_PROMPTS >>>
    EVAL_PROMPTS = [
        "Explain what LoRA fine-tuning is in two sentences.",
        "Write a Python function that loads a Hugging Face dataset.",
        "Summarize why gradient clipping prevents divergence.",
        "How does a chat template differ from a tokenizer?",
        "Give one reason eval scores can regress after training.",
    ]
    # <<< SUPER_TACO_PROMPTS <<<
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
    ds = ds.select(range(CFG.get("max_rows", 256))).map(to_messages)

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
        gradient_accumulation_steps=CFG.get("grad_accum", 2),
        learning_rate=CFG["learning_rate"],
        num_train_epochs=CFG["num_epochs"],
        fp16=True,
        logging_steps=4,
        save_strategy="no",
        report_to=[],
        # our collator consumes the raw `messages` column (chat template);
        # the Trainer's signature-based column stripping would remove it
        remove_unused_columns=False,
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

    model.eval()

    # ---- phase 2: base vs fine-tuned answers for the judge ----
    print("phase 2: generating eval answers", flush=True)
    chat_inputs = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in EVAL_PROMPTS
    ]

    def _gen(prompt_text: str) -> str:
        enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out_ids = model.generate(
            **enc, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(
            out_ids[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    base_answers: dict = {}
    with model.disable_adapter():
        for p, t in zip(EVAL_PROMPTS, chat_inputs, strict=True):
            base_answers[p] = _gen(t)
    ft_answers: dict = {}
    for p, t in zip(EVAL_PROMPTS, chat_inputs, strict=True):
        ft_answers[p] = _gen(t)
    print("###RESPONSES_JSON###", flush=True)
    print(json.dumps({"base": base_answers, "fine_tuned": ft_answers}), flush=True)
    print("###END_RESPONSES_JSON###", flush=True)

    out = "/content/lora_adapter"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    zip_path = shutil.make_archive("/content/lora_adapter", "zip", out)
    print("artifact:", zip_path, flush=True)


if __name__ == "__main__":
    main()
